"""Delta-conditioned, verifier-backed CCV provider for SearchQA.

The provider may only insert a bounded natural-language distractor into a
released training context.  It cannot alter the question, gold answers, or
original context.  The ordinary SearchQA rollout and exact-match verifier are
then reused for the paired old/candidate decision.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import random
import re
from typing import Any

from skillopt.model import chat_optimizer
from skillopt.model.azure_openai import chat_with_deployment
from skillopt.utils import extract_json


PATTERN_SYSTEM = """\
You are CCV, auditing an evolving retrieval-QA skill for update-induced overfitting.

Find only assumptions introduced or strengthened by S_new that are not required
by SearchQA. Each hypothesis must be testable by inserting one short, ordinary
natural-language distractor passage into an existing context while leaving the
question, all original context, and gold answers unchanged. Do not use prompt
injection, instructions to the model, malformed input, answer changes, evaluator
exploits, ambiguity, or mere arbitrary difficulty. Return no_attack rather than
inventing a weakness.

Return JSON only:
{
  "decision": "patterns" | "no_attack",
  "patterns": [{
    "pattern_id": "p1",
    "source_id": "one supplied source id",
    "delta_evidence": "changed skill behavior",
    "incidental_assumption": "falsifiable update-induced assumption",
    "minimal_intervention": "bounded distractor insertion",
    "predicted_failure": "specific wrong-answer behavior",
    "confidence": 0.0
  }]
}
"""


ATTACK_SYSTEM = """\
Generate one minimal SearchQA counterexample for the supplied hypothesis and
source item. Return only JSON. The attack may add one self-contained, ordinary
reference-style distractor passage. It must not address the solver, mention
prompts/skills/instructions, contain XML answer tags, change the question or
gold answers, delete or rewrite original context, or claim the task is broken.
The distractor should target the update-induced assumption and keep the answer
recoverable from the original context.

Schema:
{
  "decision": "attack" | "no_attack",
  "insert_position": "prefix" | "suffix",
  "inserted_text": "the distractor passage",
  "preserved_invariant": "why the original answer remains valid",
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
        "stage": "searchqa_ccv_attacker",
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


def _compact_source(item: dict, context_limit: int = 6000) -> dict:
    context = str(item.get("context", ""))
    if len(context) > context_limit:
        half = context_limit // 2
        context = context[:half] + "\n...[middle omitted]...\n" + context[-half:]
    return {
        "id": str(item.get("id", "")),
        "question": str(item.get("question", "")),
        "context": context,
        "gold_answers": [str(value) for value in item.get("answers", [])],
    }


def _compact_result(row: dict) -> dict:
    return {
        "id": str(row.get("id", "")),
        "hard": row.get("hard"),
        "soft": row.get("soft"),
        "predicted_answer": str(row.get("predicted_answer", ""))[:500],
        "phase": str(row.get("phase", "")),
        "fail_reason": str(row.get("fail_reason", ""))[:500],
        "agent_ok": row.get("agent_ok"),
    }


def _is_transport_failure(row: dict) -> bool:
    return not bool(row.get("agent_ok", True)) or str(row.get("phase", "")) in {
        "error",
        "timeout",
    }


def _validate_insertion(payload: dict) -> tuple[str, str]:
    if payload.get("decision") != "attack":
        raise ValueError("generator returned no_attack")
    position = str(payload.get("insert_position", "")).strip().lower()
    if position not in {"prefix", "suffix"}:
        raise ValueError("insert_position must be prefix or suffix")
    text = str(payload.get("inserted_text", "")).strip()
    if not 40 <= len(text) <= 1600:
        raise ValueError("inserted_text must contain 40..1600 characters")
    lowered = text.casefold()
    forbidden = [token for token in _FORBIDDEN if token in lowered]
    if forbidden:
        raise ValueError(f"forbidden prompt-like content: {forbidden[0]}")
    if re.search(r"\b(prompt|instruction|assistant|chatbot)\b", lowered):
        raise ValueError("distractor contains prompt-oriented vocabulary")
    return position, text


def _materialize_attack(
    source: dict,
    payload: dict,
    *,
    step: int,
    index: int,
) -> dict:
    position, inserted = _validate_insertion(payload)
    original_context = str(source["context"])
    separator = "\n\n[DOC]\n"
    context = (
        inserted + separator + original_context
        if position == "prefix"
        else original_context + separator + inserted
    )
    digest = hashlib.sha256(
        f"{source['id']}\0{position}\0{inserted}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        "id": f"ccv-s{step:04d}-{index:02d}-{digest}",
        "question": source["question"],
        "context": context,
        "answers": list(source["answers"]),
        "task_type": "searchqa",
        "ccv_source_id": str(source["id"]),
        "ccv_intervention": {
            "kind": "bounded_context_insertion",
            "position": position,
            "inserted_text": inserted,
        },
    }


def _write_result(out_dir: str, result: dict) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ccv_result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def run_searchqa_ccv(
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

    generated = materialized = validator_passed = old_qualified = paired = 0
    invalid = transport_failures = old_successes = candidate_successes = 0
    regressions = improvements = 0
    early_stopped = False
    for pattern in patterns:
        source_id = str(pattern["source_id"])
        source = qualified[source_id]
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
                response_path = os.path.join(out_dir, f"attack_{generated:02d}_response.txt")
                with open(response_path, "w", encoding="utf-8") as handle:
                    handle.write(response)
                attack_payload = extract_json(response)
                if not isinstance(attack_payload, dict):
                    raise ValueError("attack response did not contain a JSON object")
                attack = _materialize_attack(source, attack_payload, step=step, index=generated)
                materialized += 1
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
                "attack_id": attack["id"], "pattern": pattern,
                "intervention": attack["ccv_intervention"],
            }
            old_dir = os.path.join(out_dir, "paired_runs", attack["id"], "old")
            old_result = adapter.rollout([attack], old_skill, old_dir)[0]
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

            candidate_dir = os.path.join(out_dir, "paired_runs", attack["id"], "candidate")
            candidate_result = adapter.rollout([attack], candidate_skill, candidate_dir)[0]
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
        "materialized_attacks": materialized,
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
