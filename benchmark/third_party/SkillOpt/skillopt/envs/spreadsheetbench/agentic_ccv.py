"""Claude Agent SDK provider for verifier-backed SpreadsheetBench CCV.

The Claude agent owns hypothesis discovery, source inspection, attack
construction, and iterative paired testing. All filesystem and target-model
operations remain behind bounded in-process MCP tools: the agent receives no
shell, arbitrary file-read, network, or target-key access.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from functools import partial
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Annotated, Any, TypedDict

import anyio
from openpyxl.utils.cell import coordinate_to_tuple, range_boundaries

from skillopt.envs.spreadsheetbench.ccv import (
    GeneratedAttack,
    _answer_position,
    _compact_result,
    _is_transport_failure,
    _load_recent_trajectories,
    _materialize_attack,
    _semantic_diff,
    _source_manifest,
    _validate_operations_against_manifest,
)
from skillopt.utils import extract_json


AGENT_SYSTEM_PROMPT = """\
You are an autonomous Skill-Conditioned Adversarial Probing (SCAP) agent.
Your sole goal is to falsify the proposed update S_old -> S_candidate by
finding a valid, minimal task counterfactual on which S_old succeeds and
S_candidate fails.

Direction is mandatory: never design an attack to expose a weakness in S_old,
never test whether S_candidate improves on S_old, and never treat S_old failure
as success. Every proposal must explain why the old policy should remain
successful and why a newly introduced or strengthened candidate policy should
cause the candidate-specific failure.
Discard any hypothesis whose prediction is "S_old fails but S_candidate
succeeds." That is an improvement test in the wrong causal direction. The
expected_candidate_failure field itself must explicitly name S_candidate (or
the candidate/new skill) and explain why it fails.

You own the complete search loop:
1. Read the skill delta, recent execution trajectories, source catalog, and
   public verifier contract through read_context.
2. Prefer source_catalog entries marked passed_in_current_rollout=true. They
   already have one verifier-confirmed S_old pass; inspect them and call
   qualify_source once for the required confirmation. Qualify the minimum
   number of stable sources before attacking, and do not keep scanning after
   that unless an attack fails because S_old does not pass.
3. Form an update-specific, falsifiable hypothesis. The weakness may involve
   the whole delta or interactions among multiple rules; it need not target a
   single atom.
4. Inspect mutation_contract returned by inspect_source, then call
   validate_operations on the exact operations and modified_locations. This
   preflight is side-effect-free and does not consume the invalid-proposal
   repair budget. Only after status=valid may you submit the same operations
   through propose_attack. The controller then applies them to a private copy
   and validates declared changes, workbook integrity, unchanged verifier
   contract, and golden-vs-golden solvability.
5. Run each accepted attack through run_pair. Pair execution is asynchronous:
   keep calling run_pair with the same attack_id until it returns a terminal
   result rather than status=running. S_old is always evaluated first;
   S_candidate is evaluated only when S_old passes.
6. Use validator and execution feedback to revise the next attack.
7. Stop immediately when run_pair reports old_pass_candidate_fail=true, or
   when the controller reports that the paired-attack budget is exhausted.

An attack must preserve the latent objective and must not be prompt injection,
file corruption, ambiguity, an impossible task, a verifier exploit, or merely
an indiscriminate increase in difficulty. Prefer no attack over a fabricated
weakness.

The public verifier contract is authoritative. A proposed candidate failure
must remain a failure after every declared normalization and equivalence rule.
Do not spend attack budget on raw-output differences that the verifier ignores
or treats as equivalent. Never infer hidden labels or examples from the
contract; it contains only task-independent comparison semantics.

The intervention must causally expose the assigned candidate weakness. State
the exact bridge from the changed input/instruction property to the changed
skill rule and then to the predicted answer failure. Also explain why the clean
source does not already trigger the same failure. A neutral mutation that only
preserves the answer, while the hypothesized failure would occur equally on the
unmodified source, is not an adversarial probe and must be rejected.
Prefer input-only interventions with an unchanged golden workbook. Never
design an attack that requires turning a populated golden answer cell into a
blank; that changes the verifier's scored-cell contract and is invalid. If a
hypothesis fundamentally requires such a change, abandon it and choose another
weakness or source.

For propose_attack, source_task_id must exactly match an inspected source.
modified_locations must list exact individual Sheet!Cell coordinates, not
ranges or prose; declare a structural sheet creation as sheet:SheetName. If
adversarial_instruction differs from the original instruction, also include
the literal entry "instruction" in modified_locations. If the instruction is
unchanged, do not include that entry.
skill_delta_evidence and trajectory_evidence are concise evidence strings.
expected_failure_locations are chosen by you, not by the controller. Declare
the narrowest answer cell(s) or range(s) that should fail if your hypothesized
mechanism is correct. A candidate failure outside those locations is treated
as non-causal and cannot reject the update. The controller accepts at most
eight predicted answer cells in total, so do not declare the full answer range
unless it is genuinely small.

Do not write Python or manipulate files yourself. Supply a short operations
list and let the controller mutate its private workbook copy. Supported ops:
set_cell, clear_cell, and set_formula. Each operation has workbook="input" or
"golden" (normally input), one exact Sheet!Cell location, and for set_cell or
set_formula a JSON value. set_cell value_type may be auto, string, number, or
boolean. Use a JSON boolean plus value_type="boolean" for an Excel boolean.
Use set_formula with a formula string beginning with '='. The source-specific
mutation_contract is authoritative: do not target cells absent from its
editable_locations, do not create cells or sheets when disallowed, and never
target one workbook cell more than once. Always run validate_operations before
propose_attack; repair preflight errors before spending proposal budget. Modify
both workbook copies only when the same structural counterfactual must be
reflected in the oracle. Never copy answer values from golden into input. Do not alter evaluated
answer cells in input. If the intervention changes the correct answer, update
only necessary answer cells in golden and declare them.

If an intervention changes inputs but leaves expected answers unchanged, it is
input-only: do not include any operation with workbook="golden".

You have no shell and no arbitrary filesystem access. Use only the supplied
SCAP tools. At the end, return a concise account of hypotheses tested and the
verifier-confirmed decision.
"""


SCOUT_SYSTEM_PROMPT = """\
You are the Scout for parallel Skill-Conditioned Adversarial Probing. Analyze
one S_old -> S_candidate update and the bounded source pool supplied by the
controller. Discover the strongest diverse candidate-specific weaknesses and
return one independent attack brief per weakness. Each brief must target
exactly ONE weakness and exactly ONE source task. Do not construct workbook
mutations and do not run attacks; independent Attack Agents do that in
parallel after your response.

Return JSON only:
{"briefs":[{"brief_id":"w1","source_task_id":"...",
"single_weakness":"...","skill_delta_evidence":["..."],
"trajectory_evidence":["..."],"why_source":"...",
"expected_mechanism":"...","preserved_invariants":["..."]}]}

Weaknesses must be distinct, falsifiable, introduced or strengthened by the
candidate, and source IDs must come from the supplied pool. Prefer input-only
attacks with an unchanged golden workbook. Reject hypotheses that require
blanking a populated golden answer cell or changing the scoring contract.
Prefer distinct source IDs when the pool supports them, and prefer fewer
high-quality briefs over fabricated weaknesses. Return an empty briefs list
when no grounded, falsifiable candidate-specific weakness can be identified.
The input includes a public verifier_contract. Only return a brief when its
predicted failure is observable under that contract after normalization and
equivalence rules are applied.
"""


ATTACK_WORKER_SYSTEM_PROMPT = AGENT_SYSTEM_PROMPT + """

Parallel-worker specialization: a Scout has already completed weakness
discovery and supplied one authoritative single-weakness brief with exact
skill-delta evidence. Do not rediscover hypotheses and do not inspect any
other source. Inspect only the assigned source, construct the minimal causal
counterfactual, preflight its exact operations with validate_operations,
submit it only after status=valid, and run exactly one old/candidate pair.
After validator feedback, make only the smallest necessary repair. The source has
already been clean-qualified by the controller for both S_old and S_candidate.
"""


class OperationPreflightInput(TypedDict):
    source_task_id: Annotated[str, "Exact ID previously passed to inspect_source."]
    modified_locations: Annotated[
        list[str], "Exact Sheet!Cell declarations covering every operation."
    ]
    operations: Annotated[
        list[dict[str, Any]], "Exact operations intended for propose_attack."
    ]


class AttackProposalInput(TypedDict):
    source_task_id: Annotated[str, "Exact ID previously passed to inspect_source."]
    hypothesis: Annotated[str, "Falsifiable weakness introduced by the update."]
    skill_delta_evidence: Annotated[str, "Exact or precise changed-skill evidence."]
    trajectory_evidence: Annotated[str, "Specific observed evidence, or 'none'."]
    true_invariant: Annotated[str, "Latent task objective that remains unchanged."]
    intervention: Annotated[str, "Smallest task transformation."]
    modified_locations: Annotated[
        list[str],
        "Exact changed Sheet!Cell locations, plus 'instruction' iff revised.",
    ]
    adversarial_instruction: Annotated[
        str, "Complete minimally revised instruction, or the original instruction."
    ]
    preserved_invariants: Annotated[list[str], "Task properties kept unchanged."]
    expected_candidate_failure: Annotated[str, "Specific observable failure."]
    expected_failure_locations: Annotated[
        list[str],
        "Narrow answer cells/ranges expected to fail, e.g. Sheet1!H17.",
    ]
    why_old_should_still_pass: Annotated[
        str, "Why the unchanged old policy should remain successful."
    ]
    difficulty_control: Annotated[str, "Why the task is not merely harder."]
    causal_trigger: Annotated[
        str,
        "Exact mutation -> candidate-delta -> predicted-failure mechanism.",
    ]
    why_clean_source_does_not_trigger: Annotated[
        str,
        "Why the same candidate failure is not expected on the clean source.",
    ]
    operations: Annotated[
        list[dict[str, Any]],
        "Declarative set_cell, clear_cell, or set_formula operations on copied workbooks.",
    ]


def _safe_json(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _safe_json(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


_OBSERVED_FAILURE_LOCATION_RE = re.compile(
    r"@(?P<location>(?:'[^']+'|[^!:@]+)!\$?[A-Z]{1,3}\$?\d+)",
    flags=re.IGNORECASE,
)


def _split_excel_location(value: str) -> tuple[str, str] | None:
    text = str(value or "").strip()
    if "!" not in text:
        return None
    sheet, cells = text.rsplit("!", 1)
    sheet = sheet.strip().strip("'").casefold()
    cells = cells.strip().replace("$", "").upper()
    if not sheet or not cells:
        return None
    return sheet, cells


def _failure_location_matches(
    fail_reason: str,
    expected_locations: list[str],
) -> tuple[bool, list[str]]:
    """Check whether an observed verifier mismatch hits a declared region."""
    observed = [
        match.group("location")
        for match in _OBSERVED_FAILURE_LOCATION_RE.finditer(
            str(fail_reason or "")
        )
    ]
    for observed_location in observed:
        parsed_observed = _split_excel_location(observed_location)
        if parsed_observed is None:
            continue
        observed_sheet, observed_cell = parsed_observed
        try:
            observed_row, observed_col = coordinate_to_tuple(observed_cell)
        except ValueError:
            continue
        for expected in expected_locations:
            parsed_expected = _split_excel_location(expected)
            if parsed_expected is None:
                continue
            expected_sheet, expected_cells = parsed_expected
            if expected_sheet != observed_sheet:
                continue
            try:
                min_col, min_row, max_col, max_row = range_boundaries(
                    expected_cells
                )
            except ValueError:
                continue
            if (
                min_row <= observed_row <= max_row
                and min_col <= observed_col <= max_col
            ):
                return True, observed
    return False, observed


def _is_retryable_codegen_failure(result: dict) -> bool:
    """Identify malformed generated programs that cannot test CCV semantics."""
    phase = str(result.get("phase", "")).strip().lower()
    reason = str(result.get("fail_reason", "") or "")
    # The API call completed, but the model emitted no executable fenced code.
    # This is sampling/format noise rather than evidence that a skill fails the
    # attacked task, so allow one fresh outer rollout.
    if phase == "extract" and reason == "empty-code-block":
        return True
    if phase != "exec":
        return False
    if not reason.startswith("exec-error:"):
        return False
    return any(
        marker in reason
        for marker in (
            "SyntaxError:",
            "IndentationError:",
            "TabError:",
        )
    )


def _region_bounds(value: str) -> tuple[str, int, int, int, int] | None:
    parsed = _split_excel_location(value)
    if parsed is None:
        return None
    sheet, cells = parsed
    try:
        min_col, min_row, max_col, max_row = range_boundaries(cells)
    except ValueError:
        return None
    return sheet, min_col, min_row, max_col, max_row


def _validate_expected_failure_locations(
    expected_locations: list[str],
    answer_position: str,
    *,
    max_cells: int = 8,
) -> None:
    """Require a narrow, verifier-visible failure prediction."""
    answer = _region_bounds(answer_position)
    total_cells = 0
    for location in expected_locations:
        region = _region_bounds(location)
        if region is None:
            raise ValueError(
                f"invalid expected failure location: {location!r}"
            )
        sheet, min_col, min_row, max_col, max_row = region
        total_cells += (max_col - min_col + 1) * (max_row - min_row + 1)
        if answer is not None:
            (
                answer_sheet,
                answer_min_col,
                answer_min_row,
                answer_max_col,
                answer_max_row,
            ) = answer
            if not (
                sheet == answer_sheet
                and answer_min_col <= min_col <= max_col <= answer_max_col
                and answer_min_row <= min_row <= max_row <= answer_max_row
            ):
                raise ValueError(
                    "expected failure locations must lie inside the unchanged "
                    f"verifier answer region {answer_position!r}"
                )
    if total_cells > max_cells:
        raise ValueError(
            "expected failure locations are too broad for causal validation: "
            f"{total_cells} cells declared, maximum is {max_cells}"
        )


def _compact_history(
    adapter,
    trajectory_context: dict | None,
    history_items: int,
    *,
    allowed_source_ids: set[str] | None = None,
) -> list[dict]:
    context = trajectory_context or {}
    rows: list[dict] = []
    for key in ("current_step_rollouts", "selection_rollouts"):
        for row in context.get(key, []) or []:
            if isinstance(row, dict):
                rows.append(_compact_result(row))
    rows.extend(
        _load_recent_trajectories(
            str(getattr(adapter, "_cfg", {}).get("out_root", "") or ""),
            history_items,
        )
    )
    if allowed_source_ids is not None:
        rows = [
            row for row in rows
            if str(row.get("id", "")) in allowed_source_ids
        ]
    return rows[:history_items]


class AgenticCCVRuntime:
    """Stateful controller exposed to one Claude CCV session."""

    def __init__(
        self,
        adapter,
        *,
        old_skill: str,
        candidate_skill: str,
        out_dir: str,
        source_split: str,
        source_size: int,
        min_qualified_sources: int = 1,
        qualification_passes: int = 1,
        max_attacks: int,
        history_items: int,
        trajectory_context: dict | None,
        verifier_contract: dict | None = None,
        compact_output_ids: bool = False,
    ) -> None:
        self.adapter = adapter
        self.old_skill = old_skill
        self.candidate_skill = candidate_skill
        self.out_dir = os.path.abspath(out_dir)
        self.max_sources = max(1, int(source_size or 1))
        self.min_qualified_sources = min(
            self.max_sources, max(1, int(min_qualified_sources or 1))
        )
        self.qualification_passes = max(1, int(qualification_passes or 1))
        self.max_attacks = max(1, int(max_attacks or 1))
        self.compact_output_ids = bool(compact_output_ids)
        self.verifier_contract = dict(verifier_contract or {})
        # A one-attack smoke still needs enough room for the agent to repair a
        # semantic-direction or workbook-validation mistake. Formal runs keep
        # the original two repairs per attack.
        self.max_invalid_proposals = max(self.max_attacks * 2, 4)
        self.history = _compact_history(
            adapter, trajectory_context, int(history_items or 0)
        )
        context = trajectory_context or {}
        self.require_candidate_clean = bool(
            context.get("require_candidate_clean", False)
        )
        source_items = adapter.dataloader.get_split_items(source_split)
        self.source_by_id = {
            str(item.get("id", "")): item
            for item in source_items
            if str(item.get("id", ""))
        }
        current_passes: dict[str, dict] = {}
        for row in context.get("current_step_rollouts", []) or []:
            if not (
                isinstance(row, dict)
                and bool(row.get("hard"))
                and str(row.get("id", "")) in self.source_by_id
            ):
                continue
            compact = _compact_result(row)
            for key in (
                "stable_passes",
                "qualification_status",
                "qualification_provenance",
            ):
                if key in row:
                    compact[key] = row[key]
            current_passes[str(row.get("id", ""))] = compact
        self.candidate_clean_qualifications = {
            str(row.get("id", "")): _compact_result(row)
            for row in context.get("candidate_clean_rollouts", []) or []
            if isinstance(row, dict)
            and bool(row.get("hard"))
            and str(row.get("id", "")) in self.source_by_id
        }
        self.catalog = [
            {
                "id": source_id,
                "instruction": str(item.get("instruction", "")),
                "instruction_type": str(item.get("instruction_type", "")),
                "answer_position": _answer_position(item),
                "passed_in_current_rollout": source_id in current_passes,
            }
            for source_id, item in self.source_by_id.items()
        ]
        self.catalog.sort(
            key=lambda row: (not row["passed_in_current_rollout"], row["id"])
        )
        self.manifests: dict[str, dict] = {}
        self.attacks: dict[str, GeneratedAttack] = {}
        self.paired_attack_ids: set[str] = set()
        self.generated_proposals = 0
        self.invalid_proposals = 0
        self.operation_preflight_calls = 0
        self.materialized_attacks = 0
        self.validator_passed_attacks = 0
        self.old_qualified_attacks = 0
        self.paired_evaluated_attacks = 0
        self.valid_attacks = 0
        self.old_successes = 0
        self.candidate_successes = 0
        self.regressions = 0
        self.transport_failures = 0
        self.inconclusive_failures = 0
        self.early_stopped = False
        self.records: list[dict] = []
        self.usage: dict[str, Any] = {}
        self.source_qualifications: dict[str, dict] = {}
        for source_id, compact in current_passes.items():
            supplied_status = str(
                compact.get("qualification_status", "") or ""
            )
            supplied_stable_passes = max(
                1, int(compact.get("stable_passes", 1) or 1)
            )
            self.source_qualifications[source_id] = {
                **compact,
                "hard": True,
                "passes": [{**compact, "provenance": "current_step_rollout"}],
                "stable_passes": supplied_stable_passes,
                "qualification_status": (
                    "qualified"
                    if supplied_status == "qualified"
                    else "prequalified_needs_confirmation"
                ),
            }
        self._pair_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ccv-pair"
        )
        self._pair_jobs: dict[str, Future] = {}
        self._pair_output_ids: dict[str, str] = {}
        self._qualification_jobs: dict[str, Future] = {}
        self._pair_lock = threading.Lock()

    def _rollout_with_codegen_retry(
        self,
        item: dict,
        skill: str,
        out_dir: str,
        *,
        max_retries: int = 1,
    ) -> tuple[dict, list[dict]]:
        """Retry only malformed-code noise, preserving every attempt on disk."""
        attempts: list[dict] = []
        result: dict = {}
        for attempt in range(max_retries + 1):
            attempt_dir = (
                out_dir
                if attempt == 0
                else f"{out_dir}_codegen_retry_{attempt:02d}"
            )
            result = self.adapter.rollout([item], skill, attempt_dir)[0]
            compact = _compact_result(result)
            compact["attempt"] = attempt + 1
            compact["retryable_codegen_failure"] = (
                _is_retryable_codegen_failure(result)
            )
            attempts.append(compact)
            if not _is_retryable_codegen_failure(result):
                break
        return result, attempts

    def read_context(self, section: str) -> dict:
        normalized = str(section).strip().lower()
        if normalized == "skills":
            return {
                "S_old": self.old_skill,
                "S_candidate": self.candidate_skill,
                "semantic_delta": _semantic_diff(
                    self.old_skill, self.candidate_skill
                ),
            }
        if normalized == "history":
            return {"recent_execution_trajectories": self.history}
        if normalized == "sources":
            return {
                "source_catalog": self.catalog,
                "max_distinct_sources_to_inspect": self.max_sources,
            }
        if normalized == "budget":
            return self.budget()
        if normalized == "verifier":
            return {"verifier_contract": self.verifier_contract}
        raise ValueError(
            "section must be one of: skills, history, sources, budget, verifier"
        )

    def budget(self) -> dict:
        def confirmed(result: dict) -> bool:
            return bool(result.get("hard")) and int(
                result.get("stable_passes", 0) or 0
            ) >= self.qualification_passes

        return {
            "max_paired_attacks": self.max_attacks,
            "paired_attacks": len(self.paired_attack_ids),
            "remaining_paired_attacks": max(
                0, self.max_attacks - len(self.paired_attack_ids)
            ),
            "max_distinct_sources": self.max_sources,
            "min_qualified_sources_before_attack": self.min_qualified_sources,
            "qualification_passes_per_source": self.qualification_passes,
            "inspected_sources": sorted(self.manifests),
            "qualified_sources": sorted(
                source_id
                for source_id, result in self.source_qualifications.items()
                if confirmed(result)
            ),
            "prequalified_sources": sorted(
                source_id
                for source_id, result in self.source_qualifications.items()
                if bool(result.get("hard")) and not confirmed(result)
            ),
            "candidate_clean_qualified_sources": sorted(
                self.candidate_clean_qualifications
            ),
            "invalid_proposals": self.invalid_proposals,
            "max_invalid_proposals": self.max_invalid_proposals,
            "operation_preflight_calls": self.operation_preflight_calls,
            "early_stopped": self.early_stopped,
        }

    def inspect_source(self, source_task_id: str) -> dict:
        source_id = str(source_task_id)
        if source_id not in self.source_by_id:
            raise ValueError(f"source task is not allowed: {source_id}")
        if source_id not in self.manifests:
            if len(self.manifests) >= self.max_sources:
                raise ValueError(
                    "source inspection budget exhausted; use an already inspected "
                    f"source: {sorted(self.manifests)}"
                )
            manifest = _source_manifest(
                self.adapter, self.source_by_id[source_id]
            )
            if manifest is None:
                raise ValueError(f"source task has no runnable cases: {source_id}")
            self.manifests[source_id] = manifest
        manifest = self.manifests[source_id]
        return {
            key: value for key, value in manifest.items() if key != "task_dir"
        }

    def validate_operations(
        self,
        source_task_id: str,
        operations: list[dict],
        modified_locations: list[str],
    ) -> dict:
        """Preflight exact mutation structure without spending proposal budget."""
        source_id = str(source_task_id)
        if source_id not in self.manifests:
            raise ValueError("inspect_source must be called before validate_operations")
        self.operation_preflight_calls += 1
        result = _validate_operations_against_manifest(
            self.manifests[source_id], operations
        )
        declared = {
            str(location).replace("$", "").strip().casefold()
            for location in modified_locations
            if str(location).strip() and str(location).strip() != "instruction"
        }
        operation_locations = {
            str(operation.get("location", "")).replace("$", "").strip().casefold()
            for operation in result["operations"]
        }
        missing = sorted(operation_locations - declared)
        if missing:
            raise ValueError(
                "modified_locations does not declare operation targets: "
                + ", ".join(missing)
            )
        return {
            "status": "valid",
            "source_task_id": source_id,
            "normalized_operations": result["operations"],
            "mutation_contract_version": (
                self.manifests[source_id].get("mutation_contract", {}).get("version")
            ),
            "consumes_invalid_proposal_budget": False,
            "budget": self.budget(),
        }

    def _execute_source_qualification(self, source_id: str) -> dict:
        qualification_dir = os.path.join(
            self.out_dir, "source_qualification", source_id
        )
        existing = self.source_qualifications.get(source_id, {})
        pass_records = list(existing.get("passes", []) or [])
        result = {}
        status = "qualified"
        for pass_index in range(len(pass_records), self.qualification_passes):
            result, attempts = self._rollout_with_codegen_retry(
                self.source_by_id[source_id],
                self.old_skill,
                os.path.join(qualification_dir, f"pass_{pass_index + 1:02d}"),
            )
            pass_records.append({**_compact_result(result), "attempts": attempts})
            if _is_transport_failure(result):
                self.transport_failures += 1
                status = "transport_failure"
                break
            if not bool(result.get("hard")):
                status = "old_clean_failed"
                break
        compact = _compact_result(result)
        compact["passes"] = pass_records
        compact["stable_passes"] = sum(bool(row.get("hard")) for row in pass_records)
        compact["hard"] = status == "qualified"
        compact["qualification_status"] = status
        self.source_qualifications[source_id] = compact
        return {
            "status": status,
            "source_task_id": source_id,
            "old_clean": compact,
            "budget": self.budget(),
        }

    def qualify_source(self, source_task_id: str) -> dict:
        """Verify that S_old solves a clean source before attack design."""
        source_id = str(source_task_id)
        with self._pair_lock:
            if source_id not in self.manifests:
                raise ValueError(
                    "inspect_source must be called before qualify_source"
                )
            cached = self.source_qualifications.get(source_id)
            if cached is not None and (
                not bool(cached.get("hard"))
                or int(cached.get("stable_passes", 0) or 0)
                >= self.qualification_passes
            ):
                return {
                    "status": cached.get(
                        "qualification_status",
                        (
                            "qualified"
                            if bool(cached.get("hard"))
                            else "old_clean_failed"
                        ),
                    ),
                    "source_task_id": source_id,
                    "old_clean": cached,
                    "budget": self.budget(),
                }
            job = self._qualification_jobs.get(source_id)
            if job is None:
                job = self._pair_executor.submit(
                    self._execute_source_qualification, source_id
                )
                self._qualification_jobs[source_id] = job
        try:
            return job.result(timeout=45)
        except FutureTimeout:
            return {
                "status": "running",
                "source_task_id": source_id,
                "message": (
                    "Clean source qualification is still running. Call "
                    "qualify_source again with this source_task_id."
                ),
                "budget": self.budget(),
            }

    def propose_attack(self, attack_json: str, mutation_code: str = "") -> dict:
        if self.early_stopped:
            raise ValueError("a regression is already confirmed; stop the search")
        if len(self.paired_attack_ids) >= self.max_attacks:
            raise ValueError("paired-attack budget exhausted")
        if self.invalid_proposals >= self.max_invalid_proposals:
            raise ValueError("invalid-proposal repair budget exhausted")

        # Always perform the same cheap structural preflight server-side. Thus
        # a skipped explicit tool call or a malformed operation never burns the
        # semantic/materialization repair budget.
        preflight_payload = json.loads(str(attack_json))
        if not isinstance(preflight_payload, dict):
            raise ValueError("attack_json must decode to an object")
        preflight_operations = preflight_payload.get("operations")
        if preflight_operations is not None:
            self.validate_operations(
                str(preflight_payload.get("source_task_id", "")),
                preflight_operations,
                list(preflight_payload.get("modified_locations") or []),
            )

        self.generated_proposals += 1
        try:
            payload = preflight_payload
            if not isinstance(payload, dict):
                raise ValueError("attack_json must decode to an object")
            source_id = str(payload.get("source_task_id", ""))
            if source_id not in self.manifests:
                raise ValueError("inspect_source must be called before propose_attack")
            qualification = self.source_qualifications.get(source_id)
            if (
                not qualification
                or not bool(qualification.get("hard"))
                or int(qualification.get("stable_passes", 0) or 0)
                < self.qualification_passes
            ):
                raise ValueError(
                    "qualify_source must confirm S_old passes the clean source "
                    "before propose_attack"
                )
            if (
                self.require_candidate_clean
                and source_id not in self.candidate_clean_qualifications
            ):
                raise ValueError(
                    "controller must confirm S_candidate passes the clean "
                    "source before propose_attack"
                )
            qualified_count = sum(
                bool(row.get("hard"))
                and int(row.get("stable_passes", 0) or 0) >= self.qualification_passes
                for row in self.source_qualifications.values()
            )
            if (
                qualified_count < self.min_qualified_sources
                and len(self.manifests) < self.max_sources
            ):
                raise ValueError(
                    f"qualify at least {self.min_qualified_sources} stable sources "
                    "before proposing an attack, or exhaust the source scan budget"
                )
            modified_locations = payload.get("modified_locations")
            if not isinstance(modified_locations, list):
                raise ValueError("modified_locations must be a JSON list")
            expected_failure_locations = payload.get(
                "expected_failure_locations"
            )
            if (
                not isinstance(expected_failure_locations, list)
                or not expected_failure_locations
                or not all(
                    isinstance(location, str) and location.strip()
                    for location in expected_failure_locations
                )
            ):
                raise ValueError(
                    "expected_failure_locations must be a non-empty JSON list "
                    "of answer cells/ranges chosen by the attack agent"
                )
            _validate_expected_failure_locations(
                expected_failure_locations,
                _answer_position(self.source_by_id[source_id]),
            )
            pattern = {
                "pattern_id": f"agent-{self.generated_proposals:02d}",
                "source_id": source_id,
                "delta_evidence": payload.get("skill_delta_evidence", []),
                "trajectory_evidence": payload.get("trajectory_evidence", []),
                "true_invariant": payload.get("true_invariant", ""),
                "incidental_assumption": payload.get("hypothesis", ""),
                "minimal_intervention": payload.get("intervention", ""),
                "predicted_failure": payload.get(
                    "expected_candidate_failure", ""
                ),
                "why_old_should_still_pass": payload.get(
                    "why_old_should_still_pass", ""
                ),
                "difficulty_control": payload.get("difficulty_control", ""),
            }
            hypothesis = str(payload.get("hypothesis", ""))
            predicted = str(payload.get("expected_candidate_failure", ""))
            direction_text = " ".join([hypothesis, predicted])
            if not re.search(
                r"\b(candidate|s_candidate|new skill|new policy)\b",
                predicted,
                flags=re.IGNORECASE,
            ):
                raise ValueError(
                    "expected_candidate_failure must explicitly identify how "
                    "S_candidate (the candidate/new skill) fails"
                )
            if re.search(
                r"\b(s_old|old skill|old policy)\b",
                predicted,
                flags=re.IGNORECASE,
            ):
                raise ValueError(
                    "expected_candidate_failure targets S_old instead of S_candidate"
                )
            if (
                re.search(
                    r"\b(s_old|old skill|old policy)\b.{0,180}"
                    r"\b(fail|fails|failed|failure)\b",
                    direction_text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                and re.search(
                    r"\b(s_candidate|candidate|new skill|new policy)\b.{0,180}"
                    r"\b(pass|passes|succeed|succeeds|correct|enable|enables)\b",
                    direction_text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
            ):
                raise ValueError(
                    "attack direction is reversed: the proposal predicts S_old "
                    "fails while S_candidate succeeds; CCV requires S_old to "
                    "pass and S_candidate to fail"
                )
            why_old_passes = str(
                payload.get("why_old_should_still_pass", "")
            ).strip()
            if not why_old_passes:
                raise ValueError("why_old_should_still_pass is required")
            causal_trigger = str(payload.get("causal_trigger", "")).strip()
            clean_non_trigger = str(
                payload.get("why_clean_source_does_not_trigger", "")
            ).strip()
            if not causal_trigger:
                raise ValueError(
                    "causal_trigger is required: explain how the intervention "
                    "activates the candidate-specific weakness"
                )
            if not clean_non_trigger:
                raise ValueError(
                    "why_clean_source_does_not_trigger is required"
                )
            noncausal_text = " ".join(
                [str(payload.get("intervention", "")), causal_trigger]
            )
            if re.search(
                r"\b(regardless of|independent of|unrelated to|even without)\b"
                r"|\bdoes not (?:cause|trigger|expose|affect)\b",
                noncausal_text,
                flags=re.IGNORECASE,
            ):
                raise ValueError(
                    "intervention is explicitly non-causal for the predicted "
                    "candidate failure"
                )
            generation = {
                "decision": "attack",
                "intervention": payload.get("intervention", ""),
                "modified_locations": modified_locations,
                "adversarial_instruction": payload.get(
                    "adversarial_instruction",
                    self.source_by_id[source_id].get("instruction", ""),
                ),
                "preserved_invariants": payload.get(
                    "preserved_invariants", []
                ),
                "expected_target_failure": payload.get(
                    "expected_candidate_failure", ""
                ),
                "expected_failure_locations": expected_failure_locations,
                "verifier_plan": payload.get(
                    "verifier_plan", "unchanged benchmark verifier"
                ),
                "operations": payload.get("operations"),
            }
            operations = generation.get("operations")
            if operations is not None and (
                not isinstance(operations, list) or not operations
            ):
                raise ValueError("operations must be a non-empty JSON list")
            intervention_text = str(generation.get("intervention", ""))
            instruction_text = str(generation.get("adversarial_instruction", ""))
            if re.search(
                r"\b(instruction|request|user)\b.{0,100}"
                r"\b(preserve|keep|leave|do not overwrite|without changing)\b",
                intervention_text,
                flags=re.IGNORECASE | re.DOTALL,
            ) and not re.search(
                r"\b(preserve|keep|leave|do not overwrite|without changing)\b",
                instruction_text,
                flags=re.IGNORECASE,
            ):
                raise ValueError(
                    "attack intervention claims an explicit preservation constraint "
                    "that is absent from adversarial_instruction"
                )
            response = "```json\n" + json.dumps(
                generation, ensure_ascii=False
            ) + "\n```"
            if operations is None:
                response += "\n```python\n" + str(mutation_code).strip() + "\n```"
            attack = _materialize_attack(
                adapter=self.adapter,
                source_item=self.source_by_id[source_id],
                source_manifest=self.manifests[source_id],
                pattern=pattern,
                attack_index=self.generated_proposals,
                response=response,
                out_dir=self.out_dir,
            )
            if attack is None:
                raise ValueError("attack generator abstained")
            self.materialized_attacks += 1
            declared = {
                str(location).replace("$", "").lower()
                for location in modified_locations
            }
            observed = {
                str(change.get("location", "")).replace("$", "").lower()
                for case in attack.record.get("cases", [])
                for change in case.get("input_cell_changes", [])
            }
            undeclared = sorted(location for location in observed if location not in declared)
            if undeclared:
                raise ValueError(
                    "materialized attack changed undeclared input locations: "
                    + ", ".join(undeclared[:8])
                )
            attack_id = str(attack.item["id"])
            self.attacks[attack_id] = attack
            self.validator_passed_attacks += 1
            validation = {
                "status": "valid",
                "attack_id": attack_id,
                "source_task_id": source_id,
                "answer_position": attack.record["answer_position"],
                "instruction_type": attack.record["instruction_type"],
                "case_changes": [
                    {
                        "case": row.get("case"),
                        "input_changed": row.get("input_changed"),
                        "golden_changed": row.get("golden_changed"),
                        "input_cell_changes": row.get(
                            "input_cell_changes", []
                        ),
                    }
                    for row in attack.record.get("cases", [])
                ],
                "budget": self.budget(),
            }
            return validation
        except Exception as exc:
            self.invalid_proposals += 1
            self.records.append(
                {
                    "status": "invalid_attack",
                    "proposal": self.generated_proposals,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            raise

    def _execute_pair(self, normalized: str) -> dict:
        """Execute one controller-owned pair in a worker thread."""
        attack = self.attacks[normalized]
        record = attack.record
        output_id = self._pair_output_ids.get(normalized, normalized)
        evaluation_item = dict(attack.item)
        evaluation_item["id"] = output_id
        record["paired_output_id"] = output_id
        pair_root_name = "r" if self.compact_output_ids else "paired_runs"
        old_stage_name = "o" if self.compact_output_ids else "old"
        candidate_stage_name = "c" if self.compact_output_ids else "candidate"

        old_dir = os.path.join(
            self.out_dir, pair_root_name, output_id, old_stage_name
        )
        old_result, old_attempts = self._rollout_with_codegen_retry(
            evaluation_item, self.old_skill, old_dir
        )
        old_compact = _compact_result(old_result)
        record["old_result"] = old_compact
        record["old_attempts"] = old_attempts
        if _is_transport_failure(old_result):
            self.transport_failures += 1
            record["status"] = "old_transport_failure"
            self.records.append(record)
            return {
                "status": "transport_failure",
                "attack_id": normalized,
                "old": old_compact,
                "candidate": None,
                "old_pass_candidate_fail": False,
                "budget": self.budget(),
            }

        if _is_retryable_codegen_failure(old_result):
            self.inconclusive_failures += 1
            record["status"] = "old_execution_inconclusive"
            self.records.append(record)
            return {
                "status": "old_execution_inconclusive",
                "attack_id": normalized,
                "old": old_compact,
                "candidate": None,
                "old_pass_candidate_fail": False,
                "verdict": (
                    "INCONCLUSIVE: S_old repeatedly produced malformed code, "
                    "so the attack did not test the semantic update."
                ),
                "budget": self.budget(),
            }

        if not bool(old_result.get("hard")):
            record["status"] = "old_failed_skip_candidate"
            self.records.append(record)
            return {
                "status": "old_failed_skip_candidate",
                "attack_id": normalized,
                "old": old_compact,
                "candidate": None,
                "old_pass_candidate_fail": False,
                "verdict": (
                    "NOT A REGRESSION: S_old failed, so this attack cannot "
                    "falsify the update. Do not claim success."
                ),
                "budget": self.budget(),
            }

        self.old_qualified_attacks += 1
        self.old_successes += 1
        candidate_dir = os.path.join(
            self.out_dir, pair_root_name, output_id, candidate_stage_name
        )
        candidate_result, candidate_attempts = (
            self._rollout_with_codegen_retry(
                evaluation_item, self.candidate_skill, candidate_dir
            )
        )
        candidate_compact = _compact_result(candidate_result)
        record["candidate_result"] = candidate_compact
        record["candidate_attempts"] = candidate_attempts
        if _is_transport_failure(candidate_result):
            self.transport_failures += 1
            record["status"] = "candidate_transport_failure"
            self.records.append(record)
            return {
                "status": "transport_failure",
                "attack_id": normalized,
                "old": old_compact,
                "candidate": candidate_compact,
                "old_pass_candidate_fail": False,
                "budget": self.budget(),
            }

        if _is_retryable_codegen_failure(candidate_result):
            self.inconclusive_failures += 1
            record["status"] = "candidate_execution_inconclusive"
            self.records.append(record)
            return {
                "status": "candidate_execution_inconclusive",
                "attack_id": normalized,
                "old": old_compact,
                "candidate": candidate_compact,
                "old_pass_candidate_fail": False,
                "verdict": (
                    "INCONCLUSIVE: S_candidate repeatedly produced malformed "
                    "code, so no causal answer-region regression was confirmed."
                ),
                "budget": self.budget(),
            }

        self.valid_attacks += 1
        self.paired_evaluated_attacks += 1
        if bool(candidate_result.get("hard")):
            self.candidate_successes += 1
            record["status"] = "both_pass"
        else:
            clean_dir = os.path.join(
                self.out_dir,
                pair_root_name,
                output_id,
                "x" if self.compact_output_ids else "candidate_clean_confirmation",
            )
            source_id = str(
                record.get("source_id")
                or (record.get("pattern") or {}).get("source_id")
                or ""
            )
            cached_candidate_clean = self.candidate_clean_qualifications.get(
                source_id
            )
            if cached_candidate_clean is not None:
                candidate_clean_result = cached_candidate_clean
                candidate_clean_attempts = [{
                    **cached_candidate_clean,
                    "provenance": "controller_clean_precheck",
                }]
            else:
                clean_item = dict(self.source_by_id[source_id])
                if self.compact_output_ids:
                    clean_item["id"] = f"{output_id}c"
                candidate_clean_result, candidate_clean_attempts = (
                    self._rollout_with_codegen_retry(
                        clean_item,
                        self.candidate_skill,
                        clean_dir,
                    )
                )
            candidate_clean_compact = _compact_result(
                candidate_clean_result
            )
            record["candidate_clean_result"] = candidate_clean_compact
            record["candidate_clean_attempts"] = candidate_clean_attempts
            if _is_transport_failure(candidate_clean_result):
                self.valid_attacks -= 1
                self.paired_evaluated_attacks -= 1
                self.transport_failures += 1
                record["status"] = "candidate_clean_transport_failure"
                self.records.append(record)
                return {
                    "status": "transport_failure",
                    "attack_id": normalized,
                    "old": old_compact,
                    "candidate": candidate_compact,
                    "candidate_clean": candidate_clean_compact,
                    "old_pass_candidate_fail": False,
                    "budget": self.budget(),
                }

            expected_locations = list(
                (record.get("generation") or {}).get(
                    "expected_failure_locations", []
                )
            )
            location_match, observed_locations = _failure_location_matches(
                str(candidate_result.get("fail_reason", "")),
                expected_locations,
            )
            causal = bool(candidate_clean_result.get("hard")) and location_match
            record["causal_validation"] = {
                "candidate_clean_pass": bool(
                    candidate_clean_result.get("hard")
                ),
                "expected_failure_locations": expected_locations,
                "observed_failure_locations": observed_locations,
                "failure_location_match": location_match,
                "confirmed": causal,
            }
            if causal:
                self.regressions += 1
                self.early_stopped = True
                record["status"] = "update_regression"
            else:
                self.inconclusive_failures += 1
                record["status"] = "non_causal_candidate_failure"
        self.records.append(record)
        return {
            "status": record["status"],
            "attack_id": normalized,
            "old": old_compact,
            "candidate": candidate_compact,
            "candidate_clean": record.get("candidate_clean_result"),
            "causal_validation": record.get("causal_validation"),
            "old_pass_candidate_fail": self.early_stopped,
            "budget": self.budget(),
        }

    def run_pair(self, attack_id: str) -> dict:
        """Start or briefly poll a pair without holding the MCP call open."""
        normalized = str(attack_id)
        with self._pair_lock:
            job = self._pair_jobs.get(normalized)
            if job is None:
                if self.early_stopped:
                    raise ValueError(
                        "a regression is already confirmed; stop the search"
                    )
                if normalized not in self.attacks:
                    raise ValueError(
                        f"unknown validated attack: {normalized}"
                    )
                if len(self.paired_attack_ids) >= self.max_attacks:
                    raise ValueError("paired-attack budget exhausted")
                self.paired_attack_ids.add(normalized)
                self._pair_output_ids[normalized] = (
                    f"p{len(self._pair_output_ids) + 1:02d}"
                    if self.compact_output_ids
                    else normalized
                )
                job = self._pair_executor.submit(
                    self._execute_pair, normalized
                )
                self._pair_jobs[normalized] = job
        try:
            # Claude Code's MCP bridge can terminate long-running tool calls.
            # A short long-poll keeps the bridge responsive while target-model
            # evaluation continues in the controller-owned worker.
            return job.result(timeout=45)
        except FutureTimeout:
            return {
                "status": "running",
                "attack_id": normalized,
                "message": (
                    "Paired evaluation is still running. Call run_pair again "
                    "with this same attack_id until a terminal status is returned."
                ),
                "budget": self.budget(),
            }

    def result(self, *, agent_error: str = "") -> dict:
        agent_inconclusive = bool(agent_error) and bool(re.search(
            r"timeout|timed out|max.?turn|cancel",
            agent_error,
            flags=re.IGNORECASE,
        ))
        # Invalid-only runs did not establish that no attack exists. They may
        # reflect a tool/materialization failure, so keep them out of the
        # conclusive ``no_attack`` bucket.
        invalid_only = bool(self.invalid_proposals) and not (
            self.valid_attacks or self.inconclusive_failures
        )
        status = (
            "inconclusive"
            if agent_inconclusive or invalid_only
            else "error"
            if agent_error or self.transport_failures
            else (
                "complete"
                if self.valid_attacks or self.inconclusive_failures
                else "no_attack"
            )
        )
        return {
            "status": status,
            "generated_attacks": self.generated_proposals,
            "valid_attacks": self.valid_attacks,
            "materialized_attacks": self.materialized_attacks,
            "validator_passed_attacks": self.validator_passed_attacks,
            "old_qualified_attacks": self.old_qualified_attacks,
            "paired_evaluated_attacks": self.paired_evaluated_attacks,
            "confirmed_regressions": self.regressions,
            "old_successes": self.old_successes,
            "candidate_successes": self.candidate_successes,
            "regressions": self.regressions,
            "improvements": 0,
            "patterns": self.generated_proposals,
            "invalid_attacks": self.invalid_proposals,
            "transport_failures": self.transport_failures,
            "inconclusive_failures": self.inconclusive_failures,
            "qualified_sources": sum(
                1
                for result in self.source_qualifications.values()
                if bool(result.get("hard"))
                and int(result.get("stable_passes", 0) or 0)
                >= self.qualification_passes
            ),
            "source_qualification_failures": sum(
                1
                for result in self.source_qualifications.values()
                if not bool(result.get("hard"))
            ),
            "early_stopped": self.early_stopped,
            "attack_records": self.records,
            "usage": self.usage,
            "error": agent_error or (
                f"{self.transport_failures} paired rollout transport failure(s)"
                if self.transport_failures
                else ""
            ),
        }

    def finish_pairs(self) -> list[str]:
        """Wait for every controller-started pair before final telemetry."""
        errors: list[str] = []
        with self._pair_lock:
            jobs = list(self._pair_jobs.items())
            qualification_jobs = list(self._qualification_jobs.items())
        for source_id, job in qualification_jobs:
            try:
                job.result()
            except Exception as exc:
                errors.append(
                    f"qualification {source_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
        for attack_id, job in jobs:
            try:
                job.result()
            except Exception as exc:
                errors.append(f"{attack_id}: {type(exc).__name__}: {exc}")
        self._pair_executor.shutdown(wait=True)
        return errors


async def _run_agent_session(
    runtime: AgenticCCVRuntime,
    *,
    model: str,
    effort: str,
    max_turns: int,
    timeout: int,
    api_key: str,
    base_url: str,
    task_prompt: str = "",
    system_prompt: str = "",
    max_thinking_tokens: int = 32768,
) -> tuple[dict, str]:
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
            create_sdk_mcp_server,
            query,
            tool,
        )
    except ImportError as exc:
        return {}, f"Claude Agent SDK unavailable: {exc}"

    def tool_result(payload: dict) -> dict:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }
            ]
        }

    def tool_error(exc: Exception) -> dict:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                    ),
                }
            ],
            "is_error": True,
        }

    @tool(
        "read_context",
        "Read one bounded CCV context section: skills, history, sources, or budget.",
        {"section": str},
    )
    async def read_context(args: dict[str, Any]) -> dict:
        try:
            return tool_result(runtime.read_context(str(args["section"])))
        except Exception as exc:
            return tool_error(exc)

    @tool(
        "inspect_source",
        "Inspect one allowed source task's native workbook preview and manifest.",
        {"source_task_id": str},
    )
    async def inspect_source(args: dict[str, Any]) -> dict:
        try:
            result = await anyio.to_thread.run_sync(
                runtime.inspect_source, str(args["source_task_id"])
            )
            return tool_result(result)
        except Exception as exc:
            return tool_error(exc)

    @tool(
        "validate_operations",
        "Side-effect-free structural preflight for exact operations and "
        "modified_locations. This does not consume invalid-proposal budget.",
        OperationPreflightInput,
    )
    async def validate_operations(args: dict[str, Any]) -> dict:
        try:
            result = await anyio.to_thread.run_sync(
                runtime.validate_operations,
                str(args["source_task_id"]),
                list(args["operations"]),
                list(args["modified_locations"]),
            )
            return tool_result(result)
        except Exception as exc:
            return tool_error(exc)

    @tool(
        "qualify_source",
        "Evaluate S_old on one inspected clean source. Poll with the same "
        "source_task_id until the result is terminal.",
        {"source_task_id": str},
    )
    async def qualify_source(args: dict[str, Any]) -> dict:
        try:
            result = await anyio.to_thread.run_sync(
                runtime.qualify_source, str(args["source_task_id"])
            )
            return tool_result(result)
        except Exception as exc:
            return tool_error(exc)

    @tool(
        "propose_attack",
        "Submit one minimal counterfactual transformation for independent "
        "materialization and verifier validation. Supply only declarative "
        "cell operations; the controller applies them to copied workbooks.",
        AttackProposalInput,
    )
    async def propose_attack(args: dict[str, Any]) -> dict:
        try:
            payload = {
                "source_task_id": str(args["source_task_id"]),
                "hypothesis": str(args["hypothesis"]),
                "skill_delta_evidence": [
                    str(args["skill_delta_evidence"])
                ],
                "trajectory_evidence": [
                    str(args["trajectory_evidence"])
                ],
                "true_invariant": str(args["true_invariant"]),
                "intervention": str(args["intervention"]),
                "modified_locations": list(args["modified_locations"]),
                "adversarial_instruction": str(
                    args["adversarial_instruction"]
                ),
                "preserved_invariants": list(args["preserved_invariants"]),
                "expected_candidate_failure": str(
                    args["expected_candidate_failure"]
                ),
                "expected_failure_locations": list(
                    args["expected_failure_locations"]
                ),
                "why_old_should_still_pass": str(
                    args["why_old_should_still_pass"]
                ),
                "difficulty_control": str(args["difficulty_control"]),
                "causal_trigger": str(args["causal_trigger"]),
                "why_clean_source_does_not_trigger": str(
                    args["why_clean_source_does_not_trigger"]
                ),
                "operations": list(args["operations"]),
            }
            result = await anyio.to_thread.run_sync(
                runtime.propose_attack,
                json.dumps(payload, ensure_ascii=False),
                "",
            )
            return tool_result(result)
        except Exception as exc:
            return tool_error(exc)

    @tool(
        "run_pair",
        "Evaluate S_old first and, only if it passes, evaluate S_candidate on "
        "the exact same validated attack.",
        {"attack_id": str},
    )
    async def run_pair(args: dict[str, Any]) -> dict:
        try:
            result = await anyio.to_thread.run_sync(
                runtime.run_pair, str(args["attack_id"])
            )
            return tool_result(result)
        except Exception as exc:
            return tool_error(exc)

    server = create_sdk_mcp_server(
        name="ccv",
        version="1.0.0",
        tools=[
            read_context,
            inspect_source,
            validate_operations,
            qualify_source,
            propose_attack,
            run_pair,
        ],
    )
    env = {
        "ANTHROPIC_API_KEY": api_key,
        "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
        "ANTHROPIC_MODEL": model,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "PYTHONUTF8": "1",
    }
    options = ClaudeAgentOptions(
        cwd=runtime.out_dir,
        model=model,
        fallback_model=None,
        max_turns=max(1, int(max_turns)),
        tools=[],
        allowed_tools=[
            "mcp__ccv__read_context",
            "mcp__ccv__inspect_source",
            "mcp__ccv__validate_operations",
            "mcp__ccv__qualify_source",
            "mcp__ccv__propose_attack",
            "mcp__ccv__run_pair",
        ],
        disallowed_tools=[
            "Bash",
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "WebSearch",
            "WebFetch",
        ],
        permission_mode="bypassPermissions",
        mcp_servers={"ccv": server},
        strict_mcp_config=True,
        env=env,
        # CC Switch owns the user-level Claude endpoint/auth/model mapping.
        # Loading only "user" makes that routing intentional while still
        # excluding project and local settings from the attack workspace.
        setting_sources=["user"],
        effort=effort if effort in {"low", "medium", "high", "xhigh", "max"} else "high",
        max_thinking_tokens=max(1024, int(max_thinking_tokens)),
        system_prompt=system_prompt or AGENT_SYSTEM_PROMPT,
    )
    events_path = Path(runtime.out_dir) / "agent_events.jsonl"
    stderr_path = Path(runtime.out_dir) / "agent_stderr.log"
    summary: dict[str, Any] = {
        "requested_model": model,
        "status": "unknown",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    def capture_stderr(line: str) -> None:
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")

    options.stderr = capture_stderr
    started = time.perf_counter()
    error = ""
    try:
        with anyio.fail_after(max(1, int(timeout))):
            async for message in query(
                prompt=task_prompt or (
                    "Run the complete bounded CCV search now. Start with "
                    "read_context for skills, history, sources, and budget. "
                    "Use at most the controller-provided paired-attack budget."
                ),
                options=options,
            ):
                row = {
                    "timestamp": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "type": type(message).__name__,
                    "payload": _safe_json(message),
                }
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            print(f"    [CCV agent] tool={block.name}")
                        elif isinstance(block, TextBlock):
                            print(f"    [CCV agent] {block.text[:240]}")
                        elif isinstance(block, ThinkingBlock):
                            print("    [CCV agent] thinking")
                elif isinstance(message, ResultMessage):
                    summary.update(
                        {
                            "status": "error" if message.is_error else "success",
                            "subtype": message.subtype,
                            "duration_ms": message.duration_ms,
                            "duration_api_ms": message.duration_api_ms,
                            "num_turns": message.num_turns,
                            "session_id": message.session_id,
                            "total_cost_usd": message.total_cost_usd,
                            "usage": _safe_json(message.usage),
                            "model_usage": _safe_json(message.model_usage),
                            "result": message.result,
                            "errors": message.errors,
                        }
                    )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        summary["status"] = "error"
        summary["controller_error"] = error
    finally:
        summary["wall_time_s"] = round(time.perf_counter() - started, 3)
        summary["completed_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        (Path(runtime.out_dir) / "agent_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if summary.get("status") == "error" and not error:
        error = (
            "; ".join(map(str, summary.get("errors") or []))
            or str(summary.get("result") or "")
            or str(summary.get("subtype") or "agent session failed")
        )
    return summary, error


def _retryable_agent_transport_error(error: str) -> bool:
    return bool(re.search(
        r"stream idle timeout|no chunks received|connection reset|"
        r"connection closed|broken pipe",
        str(error or ""),
        flags=re.IGNORECASE,
    ))


async def _run_agent_session_with_retry(
    runtime: AgenticCCVRuntime,
    *,
    model: str,
    effort: str,
    max_turns: int,
    timeout: int,
    api_key: str,
    base_url: str,
    task_prompt: str = "",
    system_prompt: str = "",
    transport_retries: int = 1,
    max_thinking_tokens: int = 32768,
) -> tuple[dict, str]:
    """Run an agent within one wall-clock budget, resuming transport faults."""
    deadline = time.perf_counter() + max(1, int(timeout))
    attempts: list[dict] = []
    summary: dict[str, Any] = {}
    error = ""
    prompt = task_prompt
    for attempt in range(max(0, int(transport_retries)) + 1):
        remaining = int(deadline - time.perf_counter())
        if remaining <= 0:
            error = "TimeoutError: agent wall-clock budget exhausted"
            break
        summary, error = await _run_agent_session(
            runtime,
            model=model,
            effort=effort,
            max_turns=max_turns,
            timeout=remaining,
            api_key=api_key,
            base_url=base_url,
            task_prompt=prompt,
            system_prompt=system_prompt,
            max_thinking_tokens=max_thinking_tokens,
        )
        attempts.append({
            "attempt": attempt + 1,
            "status": summary.get("status"),
            "subtype": summary.get("subtype"),
            "error": error,
            "wall_time_s": summary.get("wall_time_s"),
            "total_cost_usd": summary.get("total_cost_usd"),
        })
        # Some Claude SDK builds emit the useful provider error in the result
        # payload, then raise a misleading wrapper exception such as
        # ``Claude Code returned an error result: success``. Inspect both so a
        # stream-idle failure still resumes transparently.
        retry_signal = " ; ".join([
            str(error or ""),
            str(summary.get("result") or ""),
            "; ".join(map(str, summary.get("errors") or [])),
        ])
        if not _retryable_agent_transport_error(retry_signal):
            break
        if attempt >= max(0, int(transport_retries)):
            break
        prompt = (
            "Resume the same bounded SCAP arm after a transport interruption. "
            "The controller runtime retained inspected sources, qualification, "
            "validated attacks, running pairs, and budgets. Read budget and only "
            "the context needed to continue; do not restart completed work."
        )
        await anyio.sleep(min(10, max(1, remaining // 20)))
    summary = dict(summary or {})
    summary["transport_attempts"] = attempts
    summary["transport_retry_count"] = max(0, len(attempts) - 1)
    summary["total_cost_usd"] = sum(
        float(row.get("total_cost_usd") or 0.0) for row in attempts
    )
    (Path(runtime.out_dir) / "agent_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary, error


class _RestrictedAdapter:
    """Expose exactly one controller-selected source to an Attack Agent."""

    def __init__(self, adapter, source_item: dict) -> None:
        self._adapter = adapter
        self._cfg = adapter._cfg
        self.data_root = adapter.data_root
        self.dataloader = self
        self.source_item = source_item

    def __getattr__(self, name: str):
        return getattr(self._adapter, name)

    def get_split_items(self, split: str) -> list[dict]:
        del split
        return [self.source_item]

    def rollout(self, *args, **kwargs):
        return self._adapter.rollout(*args, **kwargs)


def _scout_source_pool(
    adapter,
    *,
    source_split: str,
    source_size: int,
    trajectory_context: dict | None,
) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """Select a bounded, type-diverse pool already passed once by S_old."""
    items = {
        str(item.get("id", "")): item
        for item in adapter.dataloader.get_split_items(source_split)
        if str(item.get("id", ""))
    }
    passed_rows = []
    for row in (trajectory_context or {}).get("current_step_rollouts", []) or []:
        source_id = str(row.get("id", "")) if isinstance(row, dict) else ""
        if source_id in items and bool(row.get("hard")):
            passed_rows.append(row)

    limit = max(1, int(source_size or 1))
    selected: list[dict] = []
    selected_ids: set[str] = set()
    seen_types: set[str] = set()
    for row in passed_rows:
        source_id = str(row["id"])
        task_type = str(
            items[source_id].get("instruction_type")
            or row.get("instruction_type")
            or ""
        )
        if task_type in seen_types:
            continue
        selected.append(row)
        selected_ids.add(source_id)
        seen_types.add(task_type)
        if len(selected) >= limit:
            break
    for row in passed_rows:
        if len(selected) >= limit:
            break
        source_id = str(row["id"])
        if source_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(source_id)
        if len(selected) >= limit:
            break

    summaries: list[dict] = []
    passed_by_id = {str(row["id"]): row for row in selected}
    for row in selected:
        source_id = str(row["id"])
        manifest = _source_manifest(adapter, items[source_id])
        if manifest is None:
            continue
        summaries.append({
            "id": source_id,
            "instruction": str(items[source_id].get("instruction", "")),
            "instruction_type": str(
                items[source_id].get("instruction_type", "")
            ),
            "answer_position": _answer_position(items[source_id]),
            "manifest": {
                key: value for key, value in manifest.items()
                if key != "task_dir"
            },
            "old_rollout": _compact_result(row),
        })
    allowed = {str(row["id"]) for row in summaries}
    return (
        summaries,
        {source_id: item for source_id, item in items.items() if source_id in allowed},
        {source_id: row for source_id, row in passed_by_id.items() if source_id in allowed},
    )


async def _run_scout_once(
    *,
    old_skill: str,
    candidate_skill: str,
    history: list[dict],
    source_summaries: list[dict],
    verifier_contract: dict,
    max_briefs: int,
    out_dir: str,
    model: str,
    effort: str,
    timeout: int,
    max_thinking_tokens: int,
    api_key: str,
    base_url: str,
) -> tuple[list[dict], dict, str]:
    """Run one discovery-only agent and validate its bounded brief list."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
    except ImportError as exc:
        return [], {}, f"Claude Agent SDK unavailable: {exc}"

    scout_dir = Path(out_dir) / "scout"
    scout_dir.mkdir(parents=True, exist_ok=True)
    prompt = json.dumps({
        "old_skill": old_skill,
        "candidate_skill": candidate_skill,
        "semantic_delta": _semantic_diff(old_skill, candidate_skill),
        "recent_execution_trajectories": history,
        "source_pool": source_summaries,
        "verifier_contract": verifier_contract,
        "max_attack_briefs": max(1, int(max_briefs)),
    }, ensure_ascii=False)
    options = ClaudeAgentOptions(
        cwd=str(scout_dir),
        model=model,
        fallback_model=None,
        max_turns=8,
        tools=[],
        allowed_tools=[],
        disallowed_tools=[
            "Bash", "Read", "Write", "Edit", "Glob", "Grep",
            "WebSearch", "WebFetch",
        ],
        permission_mode="bypassPermissions",
        strict_mcp_config=True,
        env={
            "ANTHROPIC_API_KEY": api_key,
            "ANTHROPIC_BASE_URL": base_url.rstrip("/"),
            "ANTHROPIC_MODEL": model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "PYTHONUTF8": "1",
        },
        setting_sources=["user"],
        effort=(effort if effort in {"low", "medium", "high", "xhigh", "max"}
                else "medium"),
        max_thinking_tokens=max(1024, int(max_thinking_tokens)),
        system_prompt=SCOUT_SYSTEM_PROMPT,
    )
    summary: dict[str, Any] = {
        "requested_model": model,
        "status": "unknown",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    started = time.perf_counter()
    error = ""
    try:
        with anyio.fail_after(max(1, int(timeout))):
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    summary.update({
                        "status": "error" if message.is_error else "success",
                        "subtype": message.subtype,
                        "num_turns": message.num_turns,
                        "total_cost_usd": message.total_cost_usd,
                        "result": message.result,
                        "usage": _safe_json(message.usage),
                        "model_usage": _safe_json(message.model_usage),
                    })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        summary["status"] = "error"
        summary["controller_error"] = error
    finally:
        summary["wall_time_s"] = round(time.perf_counter() - started, 3)
        summary["completed_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

    parsed = extract_json(str(summary.get("result", ""))) or {}
    raw_briefs = parsed.get("briefs", []) if isinstance(parsed, dict) else []
    allowed_ids = {str(row["id"]) for row in source_summaries}
    briefs: list[dict] = []
    seen_weaknesses: set[str] = set()
    for brief in raw_briefs:
        if not isinstance(brief, dict):
            continue
        source_id = str(brief.get("source_task_id", ""))
        weakness = " ".join(
            str(brief.get("single_weakness", "")).casefold().split()
        )
        if (
            source_id not in allowed_ids
            or not weakness
            or weakness in seen_weaknesses
        ):
            continue
        briefs.append(brief)
        seen_weaknesses.add(weakness)
        if len(briefs) >= max(1, int(max_briefs)):
            break
    summary["validated_briefs"] = briefs
    (scout_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if summary.get("status") == "error" and not error:
        error = str(summary.get("result") or summary.get("subtype") or "scout failed")
    return briefs, summary, error


def _run_harbor_json_agent(
    *,
    out_dir: str,
    name: str,
    system_prompt: str,
    payload: dict,
    output_name: str,
    model: str,
    timeout: int,
    provider: str = "harbor_codex",
    effort: str = "medium",
) -> tuple[dict, dict, str]:
    """Run one bounded structured CCV role through Harbor or native Codex."""
    from skillopt.model.codex_harness import (
        prepare_workspace,
        run_regularization_codex_exec,
        run_target_exec,
    )

    normalized_provider = str(provider).strip().lower()
    work_dir = Path(out_dir) / name
    task_text = (
        system_prompt.strip()
        + "\n\n## Controller input\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + f"\n\nWrite JSON only to `{output_name}`."
    )
    prepare_workspace(
        work_dir=str(work_dir),
        skill_md="",
        task_text=task_text,
    )
    summary: dict[str, Any] = {
        "requested_model": model,
        "provider": normalized_provider,
        "status": "unknown",
    }
    started = time.perf_counter()
    try:
        prompt = (
            f"Read task.md and write the required valid JSON object to {output_name}. "
            "Do not access anything outside this workspace."
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
            raise ValueError(f"unknown structured CCV provider: {provider}")
        output_path = work_dir / output_name
        parsed = (
            json.loads(output_path.read_text(encoding="utf-8"))
            if output_path.is_file()
            else extract_json(final)
        )
        if not isinstance(parsed, dict):
            raise RuntimeError("structured CCV output is not a JSON object")
        summary.update({
            "status": "success",
            "result": final,
            "raw_chars": len(raw),
            "wall_time_s": round(time.perf_counter() - started, 3),
        })
        return parsed, summary, ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        summary.update({
            "status": "error",
            "controller_error": error,
            "wall_time_s": round(time.perf_counter() - started, 3),
        })
        return {}, summary, error


async def _run_scout_once_harbor(
    *,
    old_skill: str,
    candidate_skill: str,
    history: list[dict],
    source_summaries: list[dict],
    verifier_contract: dict,
    max_briefs: int,
    out_dir: str,
    model: str,
    timeout: int,
    provider: str = "harbor_codex",
    effort: str = "medium",
) -> tuple[list[dict], dict, str]:
    parsed, summary, error = await anyio.to_thread.run_sync(partial(
        _run_harbor_json_agent,
        out_dir=out_dir,
        name="scout_harbor_codex",
        system_prompt=SCOUT_SYSTEM_PROMPT,
        payload={
            "old_skill": old_skill,
            "candidate_skill": candidate_skill,
            "semantic_delta": _semantic_diff(old_skill, candidate_skill),
            "recent_execution_trajectories": history,
            "source_pool": source_summaries,
            "verifier_contract": verifier_contract,
            "max_attack_briefs": max(1, int(max_briefs)),
        },
        output_name="briefs.json",
        model=model,
        timeout=timeout,
        provider=provider,
        effort=effort,
    ))
    raw_briefs = parsed.get("briefs", []) if isinstance(parsed, dict) else []
    allowed_ids = {str(row["id"]) for row in source_summaries}
    briefs: list[dict] = []
    seen_weaknesses: set[str] = set()
    for brief in raw_briefs:
        if not isinstance(brief, dict):
            continue
        source_id = str(brief.get("source_task_id", ""))
        weakness = " ".join(str(brief.get("single_weakness", "")).casefold().split())
        if source_id not in allowed_ids or not weakness or weakness in seen_weaknesses:
            continue
        briefs.append(brief)
        seen_weaknesses.add(weakness)
        if len(briefs) >= max(1, int(max_briefs)):
            break
    summary["validated_briefs"] = briefs
    return briefs, summary, error


async def _run_parallel_attack_worker_harbor(
    *,
    index: int,
    brief: dict,
    source_item: dict,
    old_qualification: dict,
    candidate_qualification: dict,
    adapter,
    old_skill: str,
    candidate_skill: str,
    out_dir: str,
    model: str,
    timeout: int,
    verifier_contract: dict,
    provider: str = "harbor_codex",
    effort: str = "medium",
) -> dict:
    """Codex replacement for one interactive Attack Worker arm.

    The model proposes only declarative operations. Existing controller code
    still performs structural preflight, materialization, clean qualification,
    paired evaluation, and causal validation.
    """
    source_id = str(source_item["id"])
    worker_dir = os.path.join(out_dir, f"w{index:02d}")
    os.makedirs(worker_dir, exist_ok=True)
    restricted = _RestrictedAdapter(adapter, source_item)
    runtime = AgenticCCVRuntime(
        restricted,
        old_skill=old_skill,
        candidate_skill=candidate_skill,
        out_dir=worker_dir,
        source_split="controller_selected",
        source_size=1,
        min_qualified_sources=1,
        qualification_passes=1,
        max_attacks=1,
        history_items=1,
        trajectory_context={
            "current_step_rollouts": [old_qualification],
            "candidate_clean_rollouts": [candidate_qualification],
            "require_candidate_clean": True,
        },
        verifier_contract=dict(verifier_contract or {}),
        compact_output_ids=True,
    )
    runtime.read_context("skills")
    runtime.read_context("history")
    runtime.read_context("sources")
    runtime.read_context("budget")
    source_detail = await anyio.to_thread.run_sync(runtime.inspect_source, source_id)
    payload = {
        "scout_brief": brief,
        "source": source_detail,
        "old_skill": old_skill,
        "candidate_skill": candidate_skill,
        "old_clean_qualification": old_qualification,
        "candidate_clean_qualification": candidate_qualification,
        "verifier_contract": verifier_contract,
        "required_schema": list(AttackProposalInput.__annotations__),
        "supported_operations": ["set_cell", "clear_cell", "set_formula"],
    }
    proposal, summary, error = await anyio.to_thread.run_sync(partial(
        _run_harbor_json_agent,
        out_dir=worker_dir,
        name="attack_harbor_codex",
        system_prompt=ATTACK_WORKER_SYSTEM_PROMPT,
        payload=payload,
        output_name="attack_proposal.json",
        model=model,
        timeout=timeout,
        provider=provider,
        effort=effort,
    ))
    try:
        if error:
            raise RuntimeError(error)
        # Normalize the two scalar evidence fields exactly as the MCP tool did.
        proposal["source_task_id"] = source_id
        for key in ("skill_delta_evidence", "trajectory_evidence"):
            value = proposal.get(key, "none")
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value)
            proposal[key] = str(value)
        preflight = await anyio.to_thread.run_sync(
            runtime.validate_operations,
            source_id,
            list(proposal.get("operations") or []),
            list(proposal.get("modified_locations") or []),
        )
        proposal["operations"] = preflight["normalized_operations"]
        validation = await anyio.to_thread.run_sync(
            runtime.propose_attack,
            json.dumps(proposal, ensure_ascii=False),
            "",
        )
        attack_id = str(validation["attack_id"])
        deadline = time.monotonic() + max(60, int(timeout))
        pair = await anyio.to_thread.run_sync(runtime.run_pair, attack_id)
        while pair.get("status") == "running" and time.monotonic() < deadline:
            await anyio.sleep(2)
            pair = await anyio.to_thread.run_sync(runtime.run_pair, attack_id)
        if pair.get("status") == "running":
            raise TimeoutError("paired evaluation exceeded CCV worker timeout")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    pair_errors = await anyio.to_thread.run_sync(runtime.finish_pairs)
    if pair_errors:
        pair_text = "; ".join(pair_errors)
        error = f"{error}; {pair_text}" if error else pair_text
    effective_error = "" if runtime.valid_attacks or runtime.regressions else error
    result = runtime.result(agent_error=effective_error)
    result.update({
        "brief": brief,
        "worker_index": index,
        "agent_summary": summary,
        "agent_error": error,
    })
    Path(worker_dir, "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


async def _controller_clean_qualification(
    adapter,
    *,
    source_items: dict[str, dict],
    selected_ids: list[str],
    old_passes: dict[str, dict],
    old_skill: str,
    candidate_skill: str,
    qualification_passes: int,
    out_dir: str,
    force_old_clean_evaluation: bool = False,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Confirm all selected sources in batched target-model calls."""
    active_ids = [source_id for source_id in selected_ids if source_id in old_passes]
    stable_counts = {
        source_id: (0 if force_old_clean_evaluation else 1)
        for source_id in active_ids
    }
    latest_old = {source_id: old_passes[source_id] for source_id in active_ids}
    first_pass = 1 if force_old_clean_evaluation else 2
    for pass_index in range(
        first_pass, max(1, int(qualification_passes)) + 1
    ):
        if not active_ids:
            break
        rows = await anyio.to_thread.run_sync(
            adapter.rollout,
            [source_items[source_id] for source_id in active_ids],
            old_skill,
            os.path.join(out_dir, "controller_qualification", "old", f"pass_{pass_index:02d}"),
        )
        row_map = {str(row.get("id", "")): row for row in rows}
        survivors: list[str] = []
        for source_id in active_ids:
            row = row_map.get(source_id, {})
            latest_old[source_id] = row
            if bool(row.get("hard")):
                stable_counts[source_id] += 1
                survivors.append(source_id)
        active_ids = survivors

    old_qualified = {
        source_id: {
            **_compact_result(latest_old[source_id]),
            "hard": True,
            "stable_passes": stable_counts[source_id],
            "qualification_status": "qualified",
            "qualification_provenance": "parallel_controller",
        }
        for source_id in active_ids
        if stable_counts[source_id] >= max(1, int(qualification_passes))
    }
    candidate_rows = []
    if old_qualified:
        candidate_rows = await anyio.to_thread.run_sync(
            adapter.rollout,
            [source_items[source_id] for source_id in old_qualified],
            candidate_skill,
            os.path.join(out_dir, "controller_qualification", "candidate"),
        )
    candidate_qualified = {
        str(row.get("id", "")): _compact_result(row)
        for row in candidate_rows
        if bool(row.get("hard"))
    }
    return old_qualified, candidate_qualified


async def _run_parallel_attack_worker(
    *,
    index: int,
    brief: dict,
    source_item: dict,
    old_qualification: dict,
    candidate_qualification: dict,
    adapter,
    old_skill: str,
    candidate_skill: str,
    out_dir: str,
    model: str,
    effort: str,
    max_turns: int,
    timeout: int,
    max_thinking_tokens: int,
    api_key: str,
    base_url: str,
    verifier_contract: dict | None = None,
) -> dict:
    source_id = str(source_item["id"])
    # Keep the physical path short on Windows. Source and attack provenance are
    # retained in result.json/telemetry; repeating them in nested directory
    # names pushed formal runs beyond the legacy MAX_PATH boundary.
    worker_dir = os.path.join(out_dir, f"w{index:02d}")
    os.makedirs(worker_dir, exist_ok=True)
    restricted = _RestrictedAdapter(adapter, source_item)
    runtime = AgenticCCVRuntime(
        restricted,
        old_skill=old_skill,
        candidate_skill=candidate_skill,
        out_dir=worker_dir,
        source_split="controller_selected",
        source_size=1,
        min_qualified_sources=1,
        qualification_passes=1,
        max_attacks=1,
        history_items=1,
        trajectory_context={
            "current_step_rollouts": [old_qualification],
            "candidate_clean_rollouts": [candidate_qualification],
            "require_candidate_clean": True,
        },
        verifier_contract=dict(verifier_contract or {}),
        compact_output_ids=True,
    )
    task_prompt = (
        "Execute exactly one independent SCAP attack arm from this Scout brief:\n"
        + json.dumps(brief, ensure_ascii=False, indent=2)
        + "\nUse only this source and only single_weakness. The controller has "
        "already clean-qualified both skills, so do not run qualification. "
        "Inspect the source, materialize the minimal causal attack, and run "
        "the pair. Prefer input-only edits and never change the scoring contract.\n"
        "The following public verifier contract is authoritative. Predicted "
        "failures must remain observable after its normalization and equivalence "
        "rules; do not target ignored differences:\n"
        + json.dumps(verifier_contract, ensure_ascii=False, indent=2)
    )
    summary, error = await _run_agent_session_with_retry(
        runtime,
        model=model,
        effort=effort,
        max_turns=min(max(1, int(max_turns)), 24),
        timeout=timeout,
        api_key=api_key,
        base_url=base_url,
        task_prompt=task_prompt,
        system_prompt=ATTACK_WORKER_SYSTEM_PROMPT,
        transport_retries=1,
        max_thinking_tokens=max_thinking_tokens,
    )
    pair_errors = await anyio.to_thread.run_sync(runtime.finish_pairs)
    if pair_errors:
        pair_text = "; ".join(pair_errors)
        error = f"{error}; {pair_text}" if error else pair_text
    # Once a pair has reached a verifier terminal state, an agent timeout while
    # narrating/polling must not erase the conclusive controller-owned result.
    effective_error = "" if runtime.valid_attacks or runtime.regressions else error
    result = runtime.result(agent_error=effective_error)
    result["brief"] = brief
    result["worker_index"] = index
    result["agent_summary"] = summary
    result["agent_error"] = error
    Path(worker_dir, "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


async def _run_parallel_agentic_ccv(
    adapter,
    *,
    old_skill: str,
    candidate_skill: str,
    out_dir: str,
    source_split: str,
    source_size: int,
    min_qualified_sources: int,
    qualification_passes: int,
    max_attacks: int,
    history_items: int,
    trajectory_context: dict | None,
    model: str,
    effort: str,
    max_turns: int,
    timeout: int,
    max_thinking_tokens: int,
    api_key: str,
    base_url: str,
    agent_runtime: str = "claude_agent_sdk",
    verifier_contract: dict | None = None,
) -> dict:
    source_summaries, source_items, old_passes = _scout_source_pool(
        adapter,
        source_split=source_split,
        source_size=source_size,
        trajectory_context=trajectory_context,
    )
    if not source_summaries:
        return {
            "status": "no_attack", "generated_attacks": 0,
            "valid_attacks": 0, "old_successes": 0,
            "candidate_successes": 0, "regressions": 0,
            "improvements": 0, "patterns": 0, "invalid_attacks": 0,
            "transport_failures": 0, "inconclusive_failures": 0,
            "qualified_sources": 0, "source_qualification_failures": 0,
            "early_stopped": False, "attack_records": [],
            "usage": {"orchestration": "scout_parallel"},
            "error": "No verifier-passed current-step source was available",
        }
    allowed_source_ids = {str(row["id"]) for row in source_summaries}
    history = _compact_history(
        adapter,
        trajectory_context,
        history_items,
        allowed_source_ids=allowed_source_ids,
    )
    if agent_runtime in {"harbor_codex", "codex_exec"}:
        briefs, scout_summary, scout_error = await _run_scout_once_harbor(
            old_skill=old_skill,
            candidate_skill=candidate_skill,
            history=history,
            source_summaries=source_summaries,
            verifier_contract=dict(verifier_contract or {}),
            max_briefs=max_attacks,
            out_dir=out_dir,
            model=model,
            timeout=timeout,
            provider=agent_runtime,
            effort=effort,
        )
    else:
        briefs, scout_summary, scout_error = await _run_scout_once(
            old_skill=old_skill,
            candidate_skill=candidate_skill,
            history=history,
            source_summaries=source_summaries,
            verifier_contract=dict(verifier_contract or {}),
            max_briefs=max_attacks,
            out_dir=out_dir,
            model=model,
            effort=effort,
            timeout=timeout,
            max_thinking_tokens=max_thinking_tokens,
            api_key=api_key,
            base_url=base_url,
        )
    if not briefs:
        return {
            "status": "inconclusive" if scout_error else "no_attack",
            "generated_attacks": 0, "valid_attacks": 0,
            "old_successes": 0, "candidate_successes": 0,
            "regressions": 0, "improvements": 0, "patterns": 0,
            "invalid_attacks": 0, "transport_failures": 0,
            "inconclusive_failures": int(bool(scout_error)),
            "qualified_sources": 0, "source_qualification_failures": 0,
            "early_stopped": False, "attack_records": [],
            "usage": {"orchestration": "scout_parallel", "scout": scout_summary},
            "error": scout_error,
        }

    selected_ids = list(dict.fromkeys(
        str(brief["source_task_id"]) for brief in briefs
    ))
    old_qualified, candidate_qualified = await _controller_clean_qualification(
        adapter,
        source_items=source_items,
        selected_ids=selected_ids,
        old_passes=old_passes,
        old_skill=old_skill,
        candidate_skill=candidate_skill,
        qualification_passes=qualification_passes,
        out_dir=out_dir,
        force_old_clean_evaluation=bool(
            (trajectory_context or {}).get("force_old_clean_evaluation", False)
        ),
    )
    runnable = [
        (index, brief)
        for index, brief in enumerate(briefs, 1)
        if str(brief["source_task_id"]) in old_qualified
        and str(brief["source_task_id"]) in candidate_qualified
    ]
    required_sources = min(
        max(1, int(min_qualified_sources)), len(selected_ids)
    )
    runnable_source_count = len(set(
        str(brief["source_task_id"]) for _, brief in runnable
    ))
    qualification_shortfall = runnable_source_count < required_sources
    if qualification_shortfall:
        print(
            "    [CCV parallel] clean qualification below required minimum: "
            f"{runnable_source_count}/{required_sources}; no attack workers launched"
        )
        runnable = []

    slots: list[dict | None] = [None] * len(runnable)

    async def launch(slot: int, index: int, brief: dict) -> None:
        source_id = str(brief["source_task_id"])
        if agent_runtime in {"harbor_codex", "codex_exec"}:
            slots[slot] = await _run_parallel_attack_worker_harbor(
                index=index,
                brief=brief,
                source_item=source_items[source_id],
                old_qualification=old_qualified[source_id],
                candidate_qualification=candidate_qualified[source_id],
                adapter=adapter,
                old_skill=old_skill,
                candidate_skill=candidate_skill,
                out_dir=out_dir,
                model=model,
                timeout=timeout,
                verifier_contract=dict(verifier_contract or {}),
                provider=agent_runtime,
                effort=effort,
            )
        else:
            slots[slot] = await _run_parallel_attack_worker(
                index=index,
                brief=brief,
                source_item=source_items[source_id],
                old_qualification=old_qualified[source_id],
                candidate_qualification=candidate_qualified[source_id],
                adapter=adapter,
                old_skill=old_skill,
                candidate_skill=candidate_skill,
                out_dir=out_dir,
                model=model,
                effort=effort,
                max_turns=max_turns,
                timeout=timeout,
                max_thinking_tokens=max_thinking_tokens,
                api_key=api_key,
                base_url=base_url,
                verifier_contract=dict(verifier_contract or {}),
            )

    if runnable:
        print(f"    [CCV parallel] launching {len(runnable)} independent attack agents")
        async with anyio.create_task_group() as group:
            for slot, (index, brief) in enumerate(runnable):
                group.start_soon(launch, slot, index, brief)
    workers = [result for result in slots if isinstance(result, dict)]

    def total(key: str) -> int:
        return sum(int(worker.get(key, 0) or 0) for worker in workers)

    valid_attacks = total("valid_attacks")
    regressions = total("regressions")
    inconclusive = total("inconclusive_failures") + sum(
        worker.get("status") in {"error", "inconclusive"}
        for worker in workers
    )
    status = (
        "complete" if valid_attacks or regressions
        else "inconclusive" if inconclusive or (briefs and not workers)
        else "no_attack"
    )
    records = []
    for worker in workers:
        for record in worker.get("attack_records", []) or []:
            records.append({
                **record,
                "worker_index": worker.get("worker_index"),
                "brief": worker.get("brief"),
            })
    result = {
        "status": status,
        "generated_attacks": total("generated_attacks"),
        "valid_attacks": valid_attacks,
        "materialized_attacks": total("materialized_attacks"),
        "validator_passed_attacks": total("validator_passed_attacks"),
        "old_qualified_attacks": total("old_qualified_attacks"),
        "paired_evaluated_attacks": total("paired_evaluated_attacks"),
        "confirmed_regressions": regressions,
        "old_successes": total("old_successes"),
        "candidate_successes": total("candidate_successes"),
        "regressions": regressions,
        "improvements": 0,
        "patterns": len(briefs),
        "invalid_attacks": total("invalid_attacks"),
        "transport_failures": total("transport_failures"),
        "inconclusive_failures": inconclusive,
        "qualified_sources": len(set(old_qualified) & set(candidate_qualified)),
        "source_qualification_failures": len(selected_ids) - len(
            set(old_qualified) & set(candidate_qualified)
        ),
        "early_stopped": bool(regressions),
        "attack_records": records,
        "usage": {
            "orchestration": (
                "harbor_codex_parallel" if agent_runtime == "harbor_codex"
                else "codex_exec_parallel" if agent_runtime == "codex_exec"
                else "scout_parallel"
            ),
            "scout": scout_summary,
            "selected_source_ids": selected_ids,
            "old_clean_qualified_source_ids": sorted(old_qualified),
            "candidate_clean_qualified_source_ids": sorted(candidate_qualified),
            "workers": [{
                "worker_index": worker.get("worker_index"),
                "source_task_id": str(
                    (worker.get("brief") or {}).get("source_task_id", "")
                ),
                "status": worker.get("status"),
                "agent_summary": worker.get("agent_summary"),
                "agent_error": worker.get("agent_error"),
            } for worker in workers],
        },
        "error": (
            f"Only {runnable_source_count}/{required_sources} distinct Scout "
            "sources passed clean qualification"
            if qualification_shortfall
            else scout_error if not workers else ""
        ),
    }
    Path(out_dir, "parallel_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_agentic_spreadsheet_ccv(
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
    agent_max_turns: int = 96,
    agent_timeout: int = 3600,
    agent_max_thinking_tokens: int = 8192,
    agent_base_url: str = "https://yibuapi.com",
    agent_api_key_env: str = "ANTHROPIC_API_KEY",
    orchestration: str = "sequential",
    verifier_contract: dict | None = None,
) -> dict:
    """Run one bounded Agentic CCV session and return generic regularization telemetry."""
    del step, seed, max_patterns, attacks_per_pattern, max_completion_tokens, timeout, retries
    del builder_reasoning_effort
    os.makedirs(out_dir, exist_ok=True)
    normalized_orchestration = str(orchestration).casefold()
    is_harbor_codex = normalized_orchestration == "harbor_codex_parallel"
    is_native_codex = normalized_orchestration == "codex_exec_parallel"
    effective_key_env = (
        "YIBU_API_KEY"
        if is_harbor_codex or is_native_codex
        else str(agent_api_key_env)
    )
    api_key = os.environ.get(effective_key_env, "")
    if not api_key:
        result = {
            "status": "error",
            "generated_attacks": 0,
            "valid_attacks": 0,
            "old_successes": 0,
            "candidate_successes": 0,
            "regressions": 0,
            "improvements": 0,
            "patterns": 0,
            "invalid_attacks": 0,
            "transport_failures": 0,
            "inconclusive_failures": 0,
            "qualified_sources": 0,
            "source_qualification_failures": 0,
            "early_stopped": False,
            "attack_records": [],
            "usage": {},
            "error": f"missing required environment variable {effective_key_env}",
        }
        with open(
            os.path.join(out_dir, "ccv_result.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        return result

    if normalized_orchestration in {
        "scout_parallel", "harbor_codex_parallel", "codex_exec_parallel"
    }:
        result = anyio.run(partial(
            _run_parallel_agentic_ccv,
            adapter,
            old_skill=old_skill,
            candidate_skill=candidate_skill,
            out_dir=out_dir,
            source_split=source_split,
            source_size=source_size,
            min_qualified_sources=min_qualified_sources,
            qualification_passes=qualification_passes,
            max_attacks=max_attacks,
            history_items=history_items,
            trajectory_context=trajectory_context,
            model=attacker_model,
            effort=reasoning_effort,
            max_turns=agent_max_turns,
            timeout=agent_timeout,
            max_thinking_tokens=agent_max_thinking_tokens,
            api_key=api_key,
            base_url=agent_base_url,
            agent_runtime=(
                "harbor_codex" if is_harbor_codex
                else "codex_exec" if is_native_codex
                else "claude_agent_sdk"
            ),
            verifier_contract=dict(verifier_contract or {}),
        ))
        with open(
            os.path.join(out_dir, "ccv_result.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        return result

    runtime = AgenticCCVRuntime(
        adapter,
        old_skill=old_skill,
        candidate_skill=candidate_skill,
        out_dir=out_dir,
        source_split=source_split,
        source_size=source_size,
        min_qualified_sources=min_qualified_sources,
        qualification_passes=qualification_passes,
        max_attacks=max_attacks,
        history_items=history_items,
        trajectory_context=trajectory_context,
        verifier_contract=dict(verifier_contract or {}),
    )
    summary, error = anyio.run(
        partial(
            _run_agent_session_with_retry,
            runtime,
            model=attacker_model,
            effort=reasoning_effort,
            max_turns=agent_max_turns,
            timeout=agent_timeout,
            max_thinking_tokens=agent_max_thinking_tokens,
            api_key=api_key,
            base_url=agent_base_url,
            transport_retries=1,
        )
    )
    pair_errors = runtime.finish_pairs()
    if pair_errors:
        pair_error = "paired evaluation failure(s): " + "; ".join(pair_errors)
        error = f"{error}; {pair_error}" if error else pair_error
    runtime.usage["agent"] = {
        key: summary.get(key)
        for key in (
            "requested_model",
            "model_usage",
            "usage",
            "total_cost_usd",
            "duration_ms",
            "duration_api_ms",
            "num_turns",
            "wall_time_s",
        )
        if summary.get(key) is not None
    }
    result = runtime.result(agent_error=error)
    with open(
        os.path.join(out_dir, "ccv_result.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result
