"""Delta-conditioned, verifier-backed CCV for LiveMathematicianBench.

The provider keeps the released question, choices, and correct label unchanged.
It may only attach one bounded, non-instructional prior-scratch-work note through
the environment's diagnostic channel.  The ordinary rollout and public MCQ
verifier then make a paired old/candidate decision on the exact same input.
"""
from __future__ import annotations

import difflib
import json
import os
import random
import re
from typing import Any

from skillopt.model import chat_optimizer
from skillopt.model.azure_openai import chat_with_deployment
from skillopt.utils import extract_json


PATTERN_SYSTEM = """\
You are CCV, auditing an evolving mathematical multiple-choice skill for
update-induced overfitting.

Find only assumptions introduced or strengthened by S_new that are not required
by LiveMathematicianBench. Each hypothesis must be testable by adding one short
piece of plausible but fallible prior scratch work while leaving the released
question, choices, and correct answer unchanged. The note may contain a tempting
incorrect intermediate claim, analogy, or heuristic, but must not instruct the
solver, reveal an answer label, alter the problem, or exploit the evaluator.
Return no_attack rather than inventing a weakness.

Return JSON only:
{
  "decision": "patterns" | "no_attack",
  "patterns": [{
    "pattern_id": "p1",
    "source_id": "one supplied source id",
    "delta_evidence": "changed skill behavior",
    "incidental_assumption": "falsifiable update-induced assumption",
    "minimal_intervention": "bounded prior scratch-work note",
    "predicted_failure": "specific wrong-choice behavior",
    "confidence": 0.0
  }]
}
"""


ATTACK_SYSTEM = """\
Generate one minimal LiveMathematicianBench counterexample for the supplied
hypothesis and source item. Return only JSON. The attack may add one plausible,
self-contained fragment of prior mathematical scratch work. It must not address
the solver, contain commands or prompt language, mention skills or instructions,
use XML answer tags, name an answer label, change the question or choices, or
claim the task is broken. It should target the update-induced assumption while
leaving the original problem fully solvable.

Schema:
{
  "decision": "attack" | "no_attack",
  "scratch_note": "the bounded fallible scratch-work fragment",
  "preserved_invariant": "why the released problem and correct choice are unchanged",
  "expected_candidate_failure": "specific predicted failure"
}
"""


_FORBIDDEN = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "developer message",
    "assistant message",
    "follow these instructions",
    "do not answer",
    "you are chatgpt",
    "language model",
    "the skill",
    "choose option",
    "answer is",
    "correct answer",
    "<answer>",
    "</answer>",
)


def _merge_usage(total: dict[str, int], usage: dict | None) -> None:
    for key, value in (usage or {}).items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + int(value)


def _call_attacker(
    *,
    system: str,
    user: str,
    model: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    retries: int,
    timeout: int,
) -> tuple[str, dict]:
    kwargs = {
        "system": system,
        "user": user,
        "max_completion_tokens": max_completion_tokens,
        "retries": retries,
        "stage": "livemathbench_ccv_attacker",
        "reasoning_effort": reasoning_effort or None,
        "timeout": timeout,
    }
    if model:
        return chat_with_deployment(deployment=model, **kwargs)
    return chat_optimizer(**kwargs)


def _semantic_diff(old_skill: str, candidate_skill: str, limit: int = 16000) -> str:
    text = "\n".join(
        difflib.unified_diff(
            old_skill.splitlines(),
            candidate_skill.splitlines(),
            fromfile="S_old",
            tofile="S_new",
            lineterm="",
            n=3,
        )
    )
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def _compact_source(item: dict, text_limit: int = 6000) -> dict:
    def clipped(value: object) -> str:
        text = str(value or "")
        return text if len(text) <= text_limit else text[:text_limit] + "...[truncated]"

    return {
        "id": str(item.get("id", "")),
        "question": clipped(item.get("question", "")),
        "choices": [
            {"label": str(choice.get("label", "")), "text": clipped(choice.get("text", ""))}
            for choice in item.get("choices", [])
        ],
        "theorem_type": list(item.get("theorem_type", []) or []),
    }


def _compact_result(row: dict) -> dict:
    return {
        "id": str(row.get("id", "")),
        "hard": row.get("hard"),
        "soft": row.get("soft"),
        "predicted_label": str(row.get("predicted_label", ""))[:100],
        "phase": str(row.get("phase", "")),
        "fail_reason": str(row.get("fail_reason", ""))[:500],
        "agent_ok": row.get("agent_ok"),
    }


def _is_transport_failure(row: dict) -> bool:
    return not bool(row.get("agent_ok", True)) or str(row.get("phase", "")) in {
        "error",
        "timeout",
    }


def _validate_scratch_note(payload: dict, choice_labels: list[str]) -> str:
    if payload.get("decision") != "attack":
        raise ValueError("generator returned no_attack")
    text = str(payload.get("scratch_note", "")).strip()
    if not 40 <= len(text) <= 1600:
        raise ValueError("scratch_note must contain 40..1600 characters")
    lowered = text.casefold()
    forbidden = [token for token in _FORBIDDEN if token in lowered]
    if forbidden:
        raise ValueError(f"forbidden prompt-like content: {forbidden[0]}")
    if re.search(r"\b(prompt|instruction|assistant|chatbot|respond|output)\b", lowered):
        raise ValueError("scratch_note contains prompt-oriented vocabulary")
    # Reject explicit answer-label steering such as "option B" or "choice C".
    labels = "|".join(re.escape(label) for label in choice_labels if label)
    if labels and re.search(rf"\b(?:option|choice|answer)\s*(?:{labels})\b", text, re.IGNORECASE):
        raise ValueError("scratch_note explicitly names an answer label")
    return text


def _write_result(out_dir: str, result: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ccv_result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def run_livemathbench_ccv(
    adapter,
    *,
    old_skill: str,
    candidate_skill: str,
    step: int,
    seed: int,
    out_dir: str,
    source_split: str,
    source_size: int,
    min_qualified_sources: int = 1,
    qualification_passes: int = 1,
    max_patterns: int,
    attacks_per_pattern: int,
    max_attacks: int,
    history_items: int,
    attacker_model: str,
    reasoning_effort: str,
    builder_reasoning_effort: str,
    max_completion_tokens: int,
    timeout: int,
    retries: int,
    trajectory_context: dict | None = None,
    verifier_contract: dict | None = None,
    **_: Any,
) -> dict:
    del history_items
    os.makedirs(out_dir, exist_ok=True)
    usage: dict[str, int] = {}
    records: list[dict] = []
    pool = adapter.dataloader.get_split_items(source_split)
    rng = random.Random(seed + step * 1009)
    source_items = list(pool)
    rng.shuffle(source_items)
    source_items = source_items[: max(0, source_size)]
    materialize = getattr(adapter.dataloader, "_materialize_batch", None)
    if callable(materialize):
        # Match ordinary LiveMath rollouts: shuffle choices and remap the gold
        # label deterministically before either side of the paired audit runs.
        source_items = materialize(source_items, seed + step * 1009)
    if not source_items:
        return _write_result(out_dir, {
            "status": "no_attack", "generated_attacks": 0,
            "valid_attacks": 0, "patterns": 0, "attack_records": [],
            "usage": usage,
        })

    qualified: dict[str, dict] = {}
    qualification_failures = 0
    for pass_index in range(max(1, qualification_passes)):
        qdir = os.path.join(out_dir, "source_qualification", f"pass_{pass_index + 1:02d}")
        results = adapter.rollout(source_items, old_skill, qdir)
        by_id = {str(row.get("id", "")): row for row in results}
        for item in source_items:
            source_id = str(item.get("id", ""))
            row = by_id.get(source_id, {})
            if bool(row.get("hard")) and not _is_transport_failure(row):
                qualified[source_id] = item
            else:
                qualification_failures += 1
    if len(qualified) < max(1, min_qualified_sources):
        return _write_result(out_dir, {
            "status": "inconclusive", "generated_attacks": 0,
            "valid_attacks": 0, "patterns": 0,
            "qualified_sources": len(qualified),
            "source_qualification_failures": qualification_failures,
            "attack_records": records, "usage": usage,
            "error": "insufficient old-skill source passes",
        })

    diff = _semantic_diff(old_skill, candidate_skill)
    pattern_request = {
        "semantic_delta": diff,
        "source_tasks": [_compact_source(item) for item in qualified.values()],
        "recent_trajectory_context": trajectory_context or {},
        "max_patterns": max_patterns,
        "verifier_contract": verifier_contract or {},
    }
    pattern_response, pattern_usage = _call_attacker(
        system=PATTERN_SYSTEM,
        user=json.dumps(pattern_request, ensure_ascii=False, indent=2),
        model=attacker_model,
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        timeout=timeout,
    )
    _merge_usage(usage, pattern_usage)
    with open(os.path.join(out_dir, "pattern_response.txt"), "w", encoding="utf-8") as handle:
        handle.write(pattern_response)
    payload = extract_json(pattern_response)
    patterns = (
        payload.get("patterns", [])
        if isinstance(payload, dict) and payload.get("decision") == "patterns"
        else []
    )
    patterns = [
        row for row in patterns
        if isinstance(row, dict) and str(row.get("source_id", "")) in qualified
    ][:max_patterns]
    if not patterns:
        return _write_result(out_dir, {
            "status": "no_attack", "generated_attacks": 0,
            "valid_attacks": 0, "patterns": 0,
            "qualified_sources": len(qualified),
            "source_qualification_failures": qualification_failures,
            "attack_records": records, "usage": usage,
        })

    generated = validator_passed = old_qualified = paired = 0
    invalid = transport_failures = old_successes = candidate_successes = 0
    regressions = improvements = 0
    early_stopped = False
    for pattern in patterns:
        source_id = str(pattern["source_id"])
        source = qualified[source_id]
        labels = [str(choice.get("label", "")) for choice in source.get("choices", [])]
        for attempt in range(attacks_per_pattern):
            if generated >= max_attacks or early_stopped:
                break
            generated += 1
            request = {
                "semantic_delta": diff,
                "hypothesis": pattern,
                "source_task": _compact_source(source),
                "attempt": attempt + 1,
                "verifier_contract": verifier_contract or {},
            }
            try:
                response, attack_usage = _call_attacker(
                    system=ATTACK_SYSTEM,
                    user=json.dumps(request, ensure_ascii=False, indent=2),
                    model=attacker_model,
                    reasoning_effort=builder_reasoning_effort,
                    max_completion_tokens=max_completion_tokens,
                    retries=retries,
                    timeout=timeout,
                )
                _merge_usage(usage, attack_usage)
                with open(
                    os.path.join(out_dir, f"attack_{generated:02d}_response.txt"),
                    "w", encoding="utf-8",
                ) as handle:
                    handle.write(response)
                attack_payload = extract_json(response)
                if not isinstance(attack_payload, dict):
                    raise ValueError("attack response did not contain a JSON object")
                scratch_note = _validate_scratch_note(attack_payload, labels)
                validator_passed += 1
            except Exception as exc:  # noqa: BLE001
                invalid += 1
                records.append({
                    "status": "invalid_attack", "source_id": source_id,
                    "pattern": pattern, "attempt": attempt + 1,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue

            record = {
                "status": "materialized", "source_id": source_id,
                "attack_id": f"ccv-s{step:04d}-{generated:02d}",
                "pattern": pattern,
                "intervention": {
                    "kind": "bounded_prior_scratch_note",
                    "scratch_note": scratch_note,
                },
            }
            pair_root = os.path.join(out_dir, "paired_runs", record["attack_id"])
            rollout_kwargs = {
                "diagnostic_mode": True,
                "diagnostic_instruction": scratch_note,
            }
            old_result = adapter.rollout(
                [source], old_skill, os.path.join(pair_root, "old"), **rollout_kwargs
            )[0]
            record["old_result"] = _compact_result(old_result)
            if _is_transport_failure(old_result):
                transport_failures += 1
                record["status"] = "old_transport_failure"
                records.append(record)
                continue
            if not bool(old_result.get("hard")):
                record["status"] = "old_failed_skip_candidate"
                records.append(record)
                continue
            old_qualified += 1
            old_successes += 1

            candidate_result = adapter.rollout(
                [source], candidate_skill, os.path.join(pair_root, "candidate"), **rollout_kwargs
            )[0]
            record["candidate_result"] = _compact_result(candidate_result)
            if _is_transport_failure(candidate_result):
                transport_failures += 1
                record["status"] = "candidate_transport_failure"
                records.append(record)
                continue
            paired += 1
            if bool(candidate_result.get("hard")):
                candidate_successes += 1
                record["status"] = "both_pass"
            else:
                regressions += 1
                early_stopped = True
                record["status"] = "update_regression"
            records.append(record)

    valid = paired
    result = {
        "status": "error" if transport_failures else ("complete" if valid else "no_attack"),
        "generated_attacks": generated,
        "valid_attacks": valid,
        "materialized_attacks": validator_passed,
        "validator_passed_attacks": validator_passed,
        "old_qualified_attacks": old_qualified,
        "paired_evaluated_attacks": paired,
        "confirmed_regressions": regressions,
        "old_successes": old_successes,
        "candidate_successes": candidate_successes,
        "regressions": regressions,
        "improvements": improvements,
        "patterns": len(patterns),
        "invalid_attacks": invalid,
        "transport_failures": transport_failures,
        "qualified_sources": len(qualified),
        "source_qualification_failures": qualification_failures,
        "early_stopped": early_stopped,
        "attack_records": records,
        "usage": usage,
        "error": f"{transport_failures} transport failure(s)" if transport_failures else "",
    }
    return _write_result(out_dir, result)
