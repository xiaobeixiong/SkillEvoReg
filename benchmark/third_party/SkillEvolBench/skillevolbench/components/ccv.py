"""Candidate-conditioned CCV probe generation for SkillEvolBench.

The benchmark adapter deliberately mutates only an appendix to the public
instruction.  The original instruction, environment, fixtures, and verifier
remain byte-identical, which gives the adapter a small auditable
goal-preservation boundary while still allowing candidate-specific distractors
and context shifts.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CCVProbe:
    status: str
    appendix: str = ""
    attack_type: str = ""
    generation_mode: str = "llm"
    targeted_assumption: str = ""
    predicted_failure_region: str = ""
    responsible_skill_ids: tuple[str, ...] = ()
    goal_preservation: str = ""
    source_instruction_sha256: str = ""
    error: str = ""
    generation_attempts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ATTACK_TYPES = frozenset({
    "over_specificity",
    "ordering_dependency",
    "missing_branch",
    "tool_dependency",
    "wording_sensitivity",
})


_SYSTEM = """You generate one bounded SCAP probe for a coding-agent skill update.
The original task instruction and verifier are immutable. The old and candidate
skills differ, so you MUST generate a probe and choose exactly one attack_type
from: over_specificity, ordering_dependency, missing_branch, tool_dependency,
wording_sensitivity. Return JSON only:
{"decision":"probe","attack_type":"...","targeted_assumption":"...",
 "predicted_failure_region":"...","responsible_skill_ids":["..."],
 "goal_preservation":"...","appendix":"..."}.

The recent trajectory is PRIVATE hypothesis evidence. Never quote or paraphrase
its verifier output, test names, stack traces, file/line locations, scores, or
failure-specific repair hints into the appendix. The appendix may add only
neutral operational context, an answer-free counterfactual condition, or a
presentation variation targeted at an incidental assumption newly introduced
by the candidate skill. It MUST NOT change the
requested deliverable, acceptance criteria, repository contents, tests, or
environment. It MUST NOT reveal a solution, expected patch, hidden test, or
answer. Do not diagnose the observed source failure or tell the agent where to
edit. Do not ask the agent to ignore the original instruction. Target a
specific semantic addition, deletion, or strengthening in the candidate. Do
not return no_probe merely because the hypothesis is uncertain. Select
responsible_skill_ids only from candidate_changed_skill_ids and identify the
skill or skills whose candidate delta creates the attacked assumption."""


_FALLBACK_APPENDICES = {
    "over_specificity": (
        "Treat concrete filenames, literals, and examples in contextual "
        "guidance as illustrative unless the original requirements or "
        "repository evidence make them binding."
    ),
    "ordering_dependency": (
        "Equivalent subtasks may be encountered in a different discovery "
        "order. Derive dependencies from evidence rather than presentation "
        "order while preserving the original deliverable."
    ),
    "missing_branch": (
        "The supplied inputs may exercise an alternate branch that remains "
        "within the original requirements. Preserve the complete documented "
        "behavior rather than optimizing only for the most visible case."
    ),
    "tool_dependency": (
        "If a preferred helper or tool is unavailable, use an equivalent "
        "repository-compatible method without changing the requested output."
    ),
    "wording_sensitivity": (
        "Equivalent concepts may use different terminology. Resolve them by "
        "their semantics and the original requirements rather than exact "
        "surface wording."
    ),
}


def _fallback_probe(
    old_skill: str, candidate_skill: str, digest: str, reason: str,
    generation_attempts: int = 0,
    responsible_skill_ids: tuple[str, ...] = (),
) -> CCVProbe:
    old_lower = old_skill.lower()
    candidate_lower = candidate_skill.lower()
    added = "\n".join(
        line for line in candidate_lower.splitlines()
        if line.strip() and line.strip() not in old_lower
    )
    if re.search(r"(?:/|\\|\.py\b|\.json\b|filename|constant|literal)", added):
        attack_type = "over_specificity"
    elif re.search(r"\b(?:first|then|before|after|order|sequence)\b", added):
        attack_type = "ordering_dependency"
    elif re.search(r"\b(?:tool|command|library|package|cli|script)\b", added):
        attack_type = "tool_dependency"
    elif re.search(r"\b(?:only|unless|except|branch|case|fallback)\b", added):
        attack_type = "missing_branch"
    else:
        attack_type = "wording_sensitivity"
    return CCVProbe(
        status="ready",
        appendix=_FALLBACK_APPENDICES[attack_type],
        attack_type=attack_type,
        generation_mode="deterministic_fallback",
        targeted_assumption=(
            "Candidate-specific hypothesis selected from the semantic delta."
        ),
        predicted_failure_region=attack_type,
        responsible_skill_ids=responsible_skill_ids,
        goal_preservation=(
            "Original instruction, deliverable, repository, and verifier remain unchanged."
        ),
        source_instruction_sha256=digest,
        error=f"fallback_after:{reason}",
        generation_attempts=generation_attempts,
    )


def _json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("CCV response contains no JSON object")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("CCV response must be a JSON object")
    return value


class SkillEvolBenchCCVAdapter:
    """Generic CCV attacker plus SkillEvolBench goal-preservation checks."""

    def __init__(
        self, call_fn: Callable[..., str], *, max_appendix_chars: int = 1600,
        generation_retries: int = 2,
        allow_deterministic_fallback: bool = False,
    ):
        self._call = call_fn
        self.max_appendix_chars = int(max_appendix_chars)
        self.generation_retries = int(generation_retries)
        self.allow_deterministic_fallback = bool(allow_deterministic_fallback)

    def generate(
        self, *, original_instruction: str, old_skill: str,
        candidate_skill: str, task_id: str, trajectory_summary: str = "",
        candidate_changed_skill_ids: list[str] | None = None,
    ) -> CCVProbe:
        digest = hashlib.sha256(original_instruction.encode()).hexdigest()
        if old_skill.strip() == candidate_skill.strip():
            return CCVProbe(status="no_delta", source_instruction_sha256=digest)
        request = {
            "task_id": task_id,
            "original_instruction": original_instruction,
            "old_skill": old_skill,
            "candidate_skill": candidate_skill,
            "recent_trajectory": trajectory_summary[:12000],
            "max_appendix_chars": self.max_appendix_chars,
            "candidate_changed_skill_ids": list(
                candidate_changed_skill_ids or ()
            ),
        }
        last_error = "generation_failed"
        attempts = self.generation_retries + 1
        for attempt in range(1, attempts + 1):
            request["generation_attempt"] = attempt
            if attempt > 1:
                request["previous_invalid_response_reason"] = last_error
            try:
                try:
                    raw = self._call(
                        json.dumps(request, ensure_ascii=False, indent=2),
                        system_prompt=_SYSTEM,
                    )
                except TypeError:
                    raw = self._call(_SYSTEM + "\n\n" + json.dumps(
                        request, ensure_ascii=False, indent=2
                    ))
                payload = _json_object(raw)
                if payload.get("decision") != "probe":
                    raise ValueError("model_no_attack")
                attack_type = str(payload.get("attack_type", "")).strip()
                if attack_type not in _ATTACK_TYPES:
                    raise ValueError("invalid_attack_type")
                appendix = str(payload.get("appendix", "")).strip()
                error = self._validate_appendix(appendix, original_instruction)
                if error:
                    raise ValueError(error)
                allowed_ids = list(candidate_changed_skill_ids or ())
                raw_responsible = payload.get("responsible_skill_ids") or []
                if not isinstance(raw_responsible, list):
                    raise ValueError("invalid_responsible_skill_ids")
                responsible = tuple(dict.fromkeys(
                    str(skill_id) for skill_id in raw_responsible
                    if str(skill_id) in allowed_ids
                ))
                if allowed_ids and not responsible:
                    if len(allowed_ids) == 1:
                        responsible = (allowed_ids[0],)
                    else:
                        raise ValueError("missing_responsible_skill_ids")
                return CCVProbe(
                    status="ready", appendix=appendix, attack_type=attack_type,
                    targeted_assumption=str(payload.get("targeted_assumption", ""))[:1000],
                    predicted_failure_region=str(payload.get("predicted_failure_region", ""))[:1000],
                    responsible_skill_ids=responsible,
                    goal_preservation=str(payload.get("goal_preservation", ""))[:1000],
                    source_instruction_sha256=digest,
                    generation_attempts=attempt,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}:{exc}"

        if self.allow_deterministic_fallback:
            return _fallback_probe(
                old_skill, candidate_skill, digest, last_error,
                generation_attempts=attempts,
                responsible_skill_ids=tuple(candidate_changed_skill_ids or ()),
            )
        return CCVProbe(
            status="inconclusive_generation_failed",
            generation_mode="llm_failed",
            source_instruction_sha256=digest,
            error=last_error,
            generation_attempts=attempts,
        )

    def materialize(self, original_instruction: str, probe: CCVProbe) -> str:
        if probe.status != "ready":
            return original_instruction
        # Original requirements remain verbatim; only the bounded appendix is
        # attacker-controlled. This is the environment-specific contract.
        return (
            original_instruction.rstrip()
            + "\n\n## Robustness context (CCV probe)\n\n"
            + probe.appendix.strip() + "\n"
        )

    def _validate_appendix(self, appendix: str, original: str) -> str:
        if not appendix:
            return "empty_appendix"
        if len(appendix) > self.max_appendix_chars:
            return "appendix_too_long"
        lower = appendix.lower()
        forbidden = (
            "ignore the original", "ignore previous", "hidden test",
            "expected patch", "solution/solve", "change the requirement",
            "verifier", "traceback", "stack trace", "/root/task",
            "recent failure", "recent output", "test_",
        )
        if any(token in lower for token in forbidden):
            return "forbidden_goal_or_answer_language"
        if appendix.strip() in original:
            return "appendix_duplicates_original"
        return ""


__all__ = ["CCVProbe", "SkillEvolBenchCCVAdapter", "_ATTACK_TYPES"]
