"""Bounded epoch-level semantic feedback agent for SkillEvoReg.

The agent can inspect every Train trajectory through paginated, read-only MCP
calls. It cannot read the workspace, validation data, or test data. The agent
submits typed guidance through a controller-validated MCP tool. A strict
final-message JSON parser remains as a compatibility fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from functools import partial
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from skillopt.utils.json_utils import extract_json


class NextEpochGuidanceInput(TypedDict):
    keep: Annotated[list[str], "Behaviors or rules the next epoch should preserve."]
    revise: Annotated[list[str], "Behaviors or rules the next epoch should revise."]
    avoid: Annotated[list[str], "Behaviors or rules the next epoch should avoid."]
    strategy: Annotated[str, "Non-empty general strategy for the next epoch."]


class EpochFeedbackSubmissionInput(TypedDict):
    failure_modes: Annotated[
        list[dict[str, Any]],
        "General failure modes grounded in the inspected Train trajectories.",
    ]
    successful_patterns: Annotated[
        list[dict[str, Any]],
        "General patterns that succeeded and should be preserved.",
    ]
    regularization_diagnosis: Annotated[
        str, "Diagnosis of complexity, dropout, and deterministic Regularization behavior."
    ]
    ccv_response: Annotated[
        str,
        "Advisory CCV interpretation; controller-verified facts remain authoritative.",
    ]
    next_epoch_guidance: NextEpochGuidanceInput


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if hasattr(value, "__dict__"):
        return _safe_json(vars(value))
    return str(value)


def _trajectory_summary(row: dict, index: int) -> dict:
    """Compact every trajectory for scalable all-example coverage."""
    keys = (
        "epoch_evidence_id", "id", "ok", "hard", "soft", "phase",
        "fail_reason", "error", "task_type", "instruction_type",
        "task_description", "n_turns", "n_cases", "n_pass",
    )
    summary = {key: row.get(key) for key in keys if key in row}
    summary["trajectory_index"] = index
    for key, value in list(summary.items()):
        if isinstance(value, str) and len(value) > 1600:
            summary[key] = value[:1600] + "...[clipped]"
    cases = row.get("cases")
    if isinstance(cases, list):
        summary["cases"] = _bounded(cases, depth=1)
    return summary


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """Environment-agnostic clipping; retain evidence, drop repeated prompts."""
    if depth > 6:
        return "...[depth clipped]..."
    if isinstance(value, str):
        return value if len(value) <= 6000 else value[:3000] + "\n...[clipped]...\n" + value[-3000:]
    if isinstance(value, list):
        return [_bounded(v, depth=depth + 1) for v in value[:100]]
    if isinstance(value, dict):
        ignored = {"target_system_prompt", "system_prompt", "skill_content"}
        return {
            str(k): _bounded(v, depth=depth + 1)
            for k, v in value.items()
            if str(k) not in ignored
        }
    return value


def _hard_pass(result: Any) -> bool:
    """Interpret only the environment verifier's hard outcome."""
    return isinstance(result, dict) and bool(result.get("hard"))


def _ccv_verifier_facts(ccv: Any) -> dict:
    """Build an immutable, compact view of CCV's verifier-confirmed facts.

    Attack-agent hypotheses are not outcomes. Only an ``update_regression``
    record with an old-pass/candidate-fail hard-verifier pair is exposed as a
    confirmed regression. Non-confirming records retain only identifiers and
    statuses so their narratives cannot be mistaken for verifier evidence.
    """
    ccv = ccv if isinstance(ccv, dict) else {}
    records = ccv.get("attack_records")
    records = records if isinstance(records, list) else []
    confirmed: list[dict] = []
    excluded: list[dict] = []
    status_counts: dict[str, int] = {}
    for raw in records:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        old_result = raw.get("old_result")
        candidate_result = raw.get("candidate_result")
        old_pass = _hard_pass(old_result)
        candidate_pass = _hard_pass(candidate_result)
        is_confirmed = status == "update_regression" and old_pass and not candidate_pass
        identity = {
            "attack_id": raw.get("attack_id"),
            "source_id": raw.get("source_id"),
            "status": status,
        }
        if not is_confirmed:
            excluded.append(identity)
            continue
        pattern = raw.get("pattern") if isinstance(raw.get("pattern"), dict) else {}
        generation = raw.get("generation") if isinstance(raw.get("generation"), dict) else {}
        causal = raw.get("causal_validation") if isinstance(raw.get("causal_validation"), dict) else {}
        confirmed.append({
            **identity,
            "verifier_outcome": {
                "old_pass": True,
                "candidate_pass": False,
                "candidate_clean_pass": _hard_pass(raw.get("candidate_clean_result")),
                "candidate_failure": _bounded((candidate_result or {}).get("fail_reason", "")),
                "candidate_cases": _bounded((candidate_result or {}).get("cases", [])),
            },
            "mutation": {
                "modified_locations": _bounded(generation.get("modified_locations", [])),
                "operations": _bounded(raw.get("mutation_operations", [])),
            },
            "agent_context": {
                "skill_delta_evidence": _bounded(pattern.get("delta_evidence", [])),
                "true_invariant": _bounded(pattern.get("true_invariant", "")),
                "minimal_intervention": _bounded(pattern.get("minimal_intervention", "")),
                "causal_validation": _bounded(causal),
            },
        })
    return {
        "status": ccv.get("status"),
        "generated_attacks": int(ccv.get("generated_attacks", len(records)) or 0),
        "paired_evaluated_attacks": int(ccv.get("paired_evaluated_attacks", 0) or 0),
        "confirmed_regression_count": len(confirmed),
        "confirmed_regressions": confirmed,
        "non_confirming_records": excluded,
        "record_status_counts": status_counts,
        "fact_policy": (
            "Only status=update_regression with verifier old_pass=true and "
            "candidate_pass=false is a confirmed regression. Non-confirming "
            "attack narratives are deliberately unavailable."
        ),
    }


def _feedback_transaction(transaction: Any, ccv_facts: dict) -> dict:
    """Remove raw CCV records from the transaction's duplicate data path."""
    transaction = transaction if isinstance(transaction, dict) else {}
    sanitized = {
        str(key): _bounded(value)
        for key, value in transaction.items()
        if str(key) != "ccv"
    }
    sanitized["scap_verifier_facts"] = ccv_facts
    return sanitized


def _controller_ccv_response(facts: dict) -> str:
    """Controller-owned text passed to the next epoch; never model-authored."""
    confirmed = facts.get("confirmed_regressions") or []
    if not confirmed:
        return (
            "No verifier-confirmed CCV regression. Non-confirming attacks "
            "must not be treated as evidence about the candidate skill."
        )
    rows = []
    for row in confirmed:
        outcome = row.get("verifier_outcome") or {}
        rows.append({
            "attack_id": row.get("attack_id"),
            "source_id": row.get("source_id"),
            "status": "verifier_confirmed_update_regression",
            "old_pass": outcome.get("old_pass"),
            "candidate_pass": outcome.get("candidate_pass"),
            "candidate_clean_pass": outcome.get("candidate_clean_pass"),
            "candidate_failure": outcome.get("candidate_failure"),
            "modified_locations": (row.get("mutation") or {}).get("modified_locations", []),
            "skill_delta_evidence": (row.get("agent_context") or {}).get("skill_delta_evidence", []),
        })
    return "Verifier-confirmed CCV regressions: " + json.dumps(
        rows, ensure_ascii=False, sort_keys=True
    )


class EpochFeedbackRuntime:
    def __init__(self, evidence: dict, *, out_dir: str, page_size: int) -> None:
        self.evidence = evidence
        self.out_dir = out_dir
        self.page_size = max(1, int(page_size))
        self.trajectories = list(evidence.get("train_trajectories", []))
        self.step_records = list(evidence.get("step_records", []))
        self.ccv_facts = _ccv_verifier_facts(evidence.get("ccv"))
        self.transaction = _feedback_transaction(
            evidence.get("transaction"), self.ccv_facts
        )
        self.read_trajectory_indices: set[int] = set()
        self.submitted: dict | None = None

    def overview(self) -> dict:
        return _bounded({
            "epoch": self.evidence.get("epoch"),
            "transaction": self.transaction,
            "actionable_guard_feedback": self.evidence.get(
                "actionable_guard_feedback"
            ),
            "skill_delta": self.evidence.get("skill_delta"),
            "complexity": self.evidence.get("complexity"),
            "scap_verifier_facts": self.ccv_facts,
            "dropout_summary": self.evidence.get("dropout_summary"),
            "trajectory_count": len(self.trajectories),
            "trajectory_pages": (len(self.trajectories) + self.page_size - 1) // self.page_size,
            "step_record_count": len(self.step_records),
            "step_record_pages": (len(self.step_records) + self.page_size - 1) // self.page_size,
            "page_size": self.page_size,
            "provenance": {
                "train_only": True,
                "validation_examples": False,
                "test_access": False,
            },
        })

    def page(self, rows: list[dict], page: int, *, trajectories: bool) -> dict:
        page = max(1, int(page))
        start = (page - 1) * self.page_size
        end = min(len(rows), start + self.page_size)
        if trajectories:
            self.read_trajectory_indices.update(range(start, end))
            items = [_trajectory_summary(row, index) for index, row in enumerate(rows[start:end], start=start)]
        else:
            items = [_bounded(row) for row in rows[start:end]]
        return {
            "page": page,
            "page_size": self.page_size,
            "total": len(rows),
            "has_more": end < len(rows),
            "items": items,
        }

    def inspect_trajectory(self, evidence_id: str) -> dict:
        for index, row in enumerate(self.trajectories):
            if str(row.get("epoch_evidence_id", "")) == evidence_id:
                return {
                    "trajectory_index": index,
                    "trajectory": _bounded(row),
                }
        return {"status": "not_found", "epoch_evidence_id": evidence_id}

    def submit(self, payload: dict) -> dict:
        unread_indices = sorted(
            set(range(len(self.trajectories))) - self.read_trajectory_indices
        )
        if unread_indices:
            unread_pages = sorted({
                (index // self.page_size) + 1 for index in unread_indices
            })
            return {
                "status": "rejected_incomplete_coverage",
                "unread_trajectories": len(unread_indices),
                "unread_pages": unread_pages,
                "instruction": (
                    "Read every page listed in unread_pages, then call "
                    "submit_feedback again with the same completed analysis."
                ),
            }
        required = {"failure_modes", "successful_patterns", "next_epoch_guidance"}
        missing = sorted(required - set(payload))
        if missing:
            return {"status": "rejected_invalid_schema", "missing": missing}
        guidance = payload.get("next_epoch_guidance")
        if not isinstance(guidance, dict) or not str(guidance.get("strategy", "")).strip():
            return {"status": "rejected_invalid_guidance"}
        bounded_payload = _bounded(payload)
        # The model may interpret facts, but it does not own the factual CCV
        # response consumed by the next epoch.
        bounded_payload.pop("scap_response", None)
        self.submitted = {
            "status": "ok",
            "author": "epoch_feedback_agent",
            "evidence_coverage": {
                "train_trajectories_read": len(self.read_trajectory_indices),
                "train_trajectories_total": len(self.trajectories),
                "validation_examples_read": 0,
                "test_examples_read": 0,
            },
            **bounded_payload,
            "verified_ccv_facts": self.ccv_facts,
            "scap_response": _controller_ccv_response(self.ccv_facts),
        }
        return {"status": "accepted", "coverage": self.submitted["evidence_coverage"]}


async def _run_agent(runtime: EpochFeedbackRuntime, *, model: str, effort: str,
                     max_turns: int, timeout: int, max_thinking_tokens: int,
                     base_url: str, api_key_env: str) -> tuple[dict, str]:
    try:
        import anyio
        from claude_agent_sdk import (
            AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock,
            ThinkingBlock, ToolUseBlock, create_sdk_mcp_server, query, tool,
        )
    except ImportError as exc:
        return {}, f"Claude Agent SDK unavailable: {exc}"

    def result(payload: dict, *, error: bool = False) -> dict:
        row = {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]}
        if error:
            row["is_error"] = True
        return row

    @tool("read_epoch_overview", "Read skill delta, Regularization/CCV summaries, budgets, and evidence counts.", {})
    async def read_epoch_overview(args: dict[str, Any]) -> dict:
        del args
        return result(runtime.overview())

    @tool("read_train_trajectories", "Read one 1-indexed page of Train execution trajectories.", {"page": int})
    async def read_train_trajectories(args: dict[str, Any]) -> dict:
        return result(runtime.page(runtime.trajectories, int(args["page"]), trajectories=True))

    @tool("inspect_train_trajectory", "Inspect one full bounded Train trajectory selected from the paginated summary index.", {"epoch_evidence_id": str})
    async def inspect_train_trajectory(args: dict[str, Any]) -> dict:
        return result(runtime.inspect_trajectory(str(args["epoch_evidence_id"])))

    @tool("read_step_records", "Read one 1-indexed page of candidate edits and deterministic Regularization telemetry.", {"page": int})
    async def read_step_records(args: dict[str, Any]) -> dict:
        return result(runtime.page(runtime.step_records, int(args["page"]), trajectories=False))

    @tool(
        "submit_feedback",
        "Submit the completed epoch analysis using the controller-required schema. "
        "If rejected, follow the returned instruction and submit again.",
        EpochFeedbackSubmissionInput,
    )
    async def submit_feedback(args: dict[str, Any]) -> dict:
        return result(runtime.submit(dict(args)))

    server = create_sdk_mcp_server(
        name="epoch_feedback", version="1.2.0",
        tools=[
            read_epoch_overview,
            read_train_trajectories,
            inspect_train_trajectory,
            read_step_records,
            submit_feedback,
        ],
    )
    api_key = str(os.environ.get(api_key_env, "") or "")
    if not api_key:
        return {}, f"missing API key environment variable: {api_key_env}"
    env = {
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
        "ANTHROPIC_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "PYTHONUTF8": "1",
    }
    system = (
        "You are EvoSkillGuard's bounded Feedback Analyst Agent. Diagnose how the epoch's "
        "skill updates interacted with Train execution evidence, complexity, dropout, and SCAP. "
        "You have no validation examples and no test access. Read every compact Train trajectory "
        "page, then inspect full trajectories selectively when their summaries need clarification. "
        "Prefer general causal guidance over copying examples or prescribing exact text. Use step "
        "records when useful. SCAP facts are controller-filtered and immutable: only entries in "
        "scap_verifier_facts.confirmed_regressions may be described as regressions. Never promote "
        "a non-confirming record, attack hypothesis, or agent narrative to a confirmed result. "
        "The controller field actionable_guard_feedback is authoritative. Produce revise/avoid "
        "guidance only for modules listed in its abnormal_modules. Do not introduce complexity "
        "compression when complexity is absent, and do not introduce SCAP repair when scap is "
        "absent. Validation rejection alone is not a repair diagnosis. "
        "Your scap_response is advisory and will be replaced by the controller. "
        "After reading every Train trajectory page, you MUST call submit_feedback. Its typed "
        "arguments are the authoritative output schema. If the controller rejects the submission, "
        "follow the returned instruction and call submit_feedback again. Do not invent alternate "
        "field names or place the required fields under a wrapper object. After an accepted "
        "submission, end with a brief confirmation."
    )
    options = ClaudeAgentOptions(
        cwd=runtime.out_dir, model=model, fallback_model=None,
        max_turns=max(1, int(max_turns)), tools=[],
        allowed_tools=[
            "mcp__epoch_feedback__read_epoch_overview",
            "mcp__epoch_feedback__read_train_trajectories",
            "mcp__epoch_feedback__inspect_train_trajectory",
            "mcp__epoch_feedback__read_step_records",
            "mcp__epoch_feedback__submit_feedback",
        ],
        disallowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebSearch", "WebFetch"],
        permission_mode="bypassPermissions", mcp_servers={"epoch_feedback": server},
        strict_mcp_config=True, env=env, setting_sources=["user"],
        effort=effort if effort in {"low", "medium", "high", "xhigh", "max"} else "medium",
        max_thinking_tokens=max(1024, int(max_thinking_tokens)), system_prompt=system,
    )
    events_path = Path(runtime.out_dir) / "agent_events.jsonl"
    summary: dict[str, Any] = {"status": "unknown", "model": model}
    started = time.perf_counter()
    error = ""
    try:
        with anyio.fail_after(max(1, int(timeout))):
            async for message in query(
                prompt=(
                    "Analyze this completed training epoch. Begin with read_epoch_overview, "
                    "read every Train trajectory page, inspect details as needed, then call "
                    "submit_feedback with the completed analysis. If submission is rejected, "
                    "fix the reported issue and submit again before ending."
                ),
                options=options,
            ):
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "type": type(message).__name__, "payload": _safe_json(message),
                    }, ensure_ascii=False) + "\n")
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            print(f"    [Feedback agent] tool={block.name}")
                        elif isinstance(block, TextBlock):
                            print(f"    [Feedback agent] {block.text[:200]}")
                        elif isinstance(block, ThinkingBlock):
                            print("    [Feedback agent] thinking")
                elif isinstance(message, ResultMessage):
                    summary.update({
                        "status": "error" if message.is_error else "success",
                        "subtype": message.subtype, "num_turns": message.num_turns,
                        "total_cost_usd": message.total_cost_usd,
                        "usage": _safe_json(message.usage), "result": message.result,
                        "errors": message.errors,
                        "stop_reason": getattr(message, "stop_reason", None),
                        "terminal_reason": getattr(message, "terminal_reason", None),
                    })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        summary.update({"status": "error", "controller_error": error})
    summary["wall_time_s"] = round(time.perf_counter() - started, 3)
    (Path(runtime.out_dir) / "agent_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary, error


def _run_codex_agent(
    runtime: EpochFeedbackRuntime,
    *,
    model: str,
    timeout: int,
    effort: str,
    provider: str,
) -> tuple[dict, str]:
    """Run the same bounded analysis through Harbor or native Codex.

    Evidence is materialized read-only by convention in the isolated trial
    workspace.  A deterministic receipt script must read every trajectory page
    before the controller accepts ``submission.json``.
    """
    from skillopt.model.codex_harness import (
        prepare_workspace,
        run_regularization_codex_exec,
        run_target_exec,
    )

    normalized_provider = str(provider).strip().lower()
    work_dir = Path(runtime.out_dir) / normalized_provider
    receipt_script = (
        "import hashlib, json, pathlib\n"
        "root = pathlib.Path('evidence')\n"
        "files = sorted(root.glob('*.json'))\n"
        "rows = []\n"
        "for path in files:\n"
        "    data = path.read_bytes()\n"
        "    json.loads(data)\n"
        "    rows.append({'file': path.name, 'sha256': hashlib.sha256(data).hexdigest()})\n"
        "pathlib.Path('evidence_receipt.json').write_text(json.dumps({'files': rows}, indent=2))\n"
        "print(json.dumps({'read_files': len(rows), 'files': [r['file'] for r in rows]}))\n"
    )
    task_text = (
        "Analyze one completed SkillEvoReg training epoch using only the files under `evidence/`.\n"
        "First run `python read_all_evidence.py`, then inspect `evidence_bundle.json` or the individual JSON pages. "
        "You have no validation or test examples. Treat only controller-confirmed CCV records as regressions.\n\n"
        "Write `submission.json` with exactly these top-level fields:\n"
        "- failure_modes: list of grounded, general failure-mode objects\n"
        "- successful_patterns: list of grounded, general pattern objects\n"
        "- regularization_diagnosis: string\n"
        "- ccv_response: string (advisory; controller will replace it)\n"
        "- next_epoch_guidance: {keep: list[str], revise: list[str], avoid: list[str], strategy: non-empty str}\n"
        "Do not copy training examples into the guidance and do not access anything outside this workspace.\n"
    )
    prepare_workspace(
        work_dir=str(work_dir),
        skill_md="",
        task_text=task_text,
        extra_files={"read_all_evidence.py": receipt_script},
    )
    # prepare_workspace intentionally recreates the workspace, so evidence must
    # be materialized afterwards.  Writing it before this call silently deletes
    # every page and can make an empty receipt look like full coverage.
    pages_dir = work_dir / "evidence"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "overview.json").write_text(
        json.dumps(runtime.overview(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for start in range(0, len(runtime.trajectories), runtime.page_size):
        page = (start // runtime.page_size) + 1
        name = f"trajectories_page_{page:03d}.json"
        payload = runtime.page(runtime.trajectories, page, trajectories=False)
        (pages_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for start in range(0, len(runtime.step_records), runtime.page_size):
        page = (start // runtime.page_size) + 1
        name = f"steps_page_{page:03d}.json"
        payload = runtime.page(runtime.step_records, page, trajectories=False)
        (pages_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    expected_receipt = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(pages_dir.glob("*.json"))
    }
    started = time.perf_counter()
    try:
        prompt = (
            "Read task.md. Run `python read_all_evidence.py`, inspect all evidence, "
            "and write the required valid JSON object to submission.json."
        )
        if normalized_provider == "harbor_codex":
            final, raw = run_target_exec(
                work_dir=str(work_dir), prompt=prompt, model=model,
                timeout=timeout, allow_file_edits=True,
            )
        elif normalized_provider == "codex_exec":
            final, raw = run_regularization_codex_exec(
                work_dir=str(work_dir), prompt=prompt, model=model,
                timeout=timeout, reasoning_effort=effort,
            )
        else:
            raise ValueError(f"unknown Codex feedback provider: {provider}")
        receipt = work_dir / "evidence_receipt.json"
        submission = work_dir / "submission.json"
        if not receipt.is_file():
            raise RuntimeError("Codex did not run the mandatory all-evidence receipt script")
        receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
        receipt_rows = receipt_payload.get("files") if isinstance(receipt_payload, dict) else None
        if not isinstance(receipt_rows, list):
            raise RuntimeError("Codex evidence receipt has no files list")
        actual_receipt = {
            str(row.get("file")): str(row.get("sha256"))
            for row in receipt_rows if isinstance(row, dict)
        }
        if actual_receipt != expected_receipt:
            raise RuntimeError(
                "Codex evidence receipt does not exactly cover the materialized evidence pages"
            )
        if submission.is_file():
            payload = json.loads(submission.read_text(encoding="utf-8"))
        else:
            payload = extract_json(final)
        if not isinstance(payload, dict):
            raise RuntimeError("Codex feedback submission is not a JSON object")
        runtime.read_trajectory_indices.update(range(len(runtime.trajectories)))
        validation = runtime.submit(payload)
        if validation.get("status") != "accepted":
            raise RuntimeError("controller rejected submission: " + json.dumps(validation))
        summary = {
            "status": "success",
            "model": model,
            "provider": normalized_provider,
            "result": final,
            "raw_chars": len(raw),
            "wall_time_s": round(time.perf_counter() - started, 3),
        }
        return summary, ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {
            "status": "error",
            "model": model,
            "provider": normalized_provider,
            "controller_error": error,
            "wall_time_s": round(time.perf_counter() - started, 3),
        }, error


def run_epoch_feedback_agent(*, evidence: dict, out_dir: str, model: str,
                             effort: str = "medium", max_turns: int = 24,
                             timeout: int = 1200, max_thinking_tokens: int = 8192,
                             page_size: int = 10, retries: int = 1,
                             provider: str = "claude_agent_sdk",
                             base_url: str = "https://yibuapi.com",
                             api_key_env: str = "ANTHROPIC_API_KEY") -> dict:
    """Run a read-only Feedback Analyst Agent with validated final-JSON retry."""
    import anyio

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    ccv_facts = _ccv_verifier_facts(evidence.get("ccv"))
    (root / "evidence_manifest.json").write_text(json.dumps({
        "epoch": evidence.get("epoch"),
        "trajectory_count": len(evidence.get("train_trajectories", [])),
        "step_record_count": len(evidence.get("step_records", [])),
        "transaction": _feedback_transaction(evidence.get("transaction"), ccv_facts),
        "scap_verifier_facts": ccv_facts,
        "actionable_guard_feedback": evidence.get(
            "actionable_guard_feedback"
        ),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    attempts: list[dict] = []
    feedback: dict | None = None
    for attempt in range(max(0, int(retries)) + 1):
        attempt_dir = root / f"attempt_{attempt + 1:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        runtime = EpochFeedbackRuntime(
            evidence, out_dir=str(attempt_dir), page_size=page_size
        )
        normalized_provider = str(provider).strip().lower()
        if normalized_provider in {"harbor_codex", "codex_exec"}:
            summary, transport_error = _run_codex_agent(
                runtime, model=model, timeout=timeout, effort=effort,
                provider=normalized_provider,
            )
        else:
            summary, transport_error = anyio.run(partial(
                _run_agent, runtime, model=model, effort=effort, max_turns=max_turns,
                timeout=timeout, max_thinking_tokens=max_thinking_tokens,
                base_url=base_url, api_key_env=api_key_env,
            ))
        protocol_error = ""
        payload = None
        raw_result = str(summary.get("result") or "").strip()
        # A validated tool submission is authoritative. The final assistant
        # message is only a human-readable acknowledgement and need not repeat
        # the full payload as free-form JSON.
        if runtime.submitted is not None:
            feedback = runtime.submitted
        elif summary.get("status") != "success":
            protocol_error = transport_error or str(
                summary.get("controller_error") or summary.get("errors")
                or "agent_session_failed"
            )
        elif not raw_result:
            protocol_error = (
                "agent_session_succeeded_without_final_json: "
                f"stop_reason={summary.get('stop_reason')!r}"
            )
        else:
            try:
                payload = extract_json(raw_result)
            except Exception as exc:
                protocol_error = f"invalid_final_json: {type(exc).__name__}: {exc}"
            if not isinstance(payload, dict):
                protocol_error = protocol_error or "invalid_final_json_object"
        if feedback is None and isinstance(payload, dict) and not protocol_error:
            validation = runtime.submit(payload)
            if validation.get("status") != "accepted":
                protocol_error = "controller_validation_failed: " + json.dumps(
                    validation, ensure_ascii=False
                )
            else:
                feedback = runtime.submitted
        attempt_row = {
            "attempt": attempt + 1,
            "status": "ok" if feedback is not None else "protocol_incomplete",
            "error": protocol_error,
            "coverage": {
                "read": len(runtime.read_trajectory_indices),
                "total": len(runtime.trajectories),
            },
            "summary": summary,
        }
        attempts.append(attempt_row)
        (attempt_dir / "protocol_result.json").write_text(
            json.dumps(attempt_row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if feedback is not None:
            break

    if feedback is None:
        last = attempts[-1] if attempts else {}
        feedback = {
            "status": "protocol_incomplete",
            "author": "epoch_feedback_agent",
            "error": last.get("error") or "feedback_agent_exhausted_retries",
            "evidence_coverage": {
                "train_trajectories_read": (last.get("coverage") or {}).get("read", 0),
                "train_trajectories_total": len(evidence.get("train_trajectories", [])),
                "validation_examples_read": 0,
                "test_examples_read": 0,
            },
            "next_epoch_guidance": {
                "keep": [], "revise": [], "avoid": [],
                "strategy": "No validated semantic guidance is available.",
            },
        }
    feedback["agent_attempts"] = attempts
    (root / "feedback.json").write_text(
        json.dumps(feedback, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return feedback
