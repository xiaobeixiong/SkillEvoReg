"""Auditable shared post-T1 anchors for paired evolution experiments."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillevolbench.schemas import OperationType, ReplayRecord, SkillPatch


class SharedT1AnchorError(ValueError):
    """Raised when a configured anchor set is incomplete or corrupted."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SharedT1AnchorError(f"unsafe anchor file path: {raw!r}")
    return path


@dataclass(frozen=True)
class SharedT1Anchor:
    patch: SkillPatch
    replay_record: ReplayRecord | None
    audit: dict[str, Any]


class SharedT1AnchorStore:
    """Load and verify one immutable set of post-T1 skill snapshots."""

    def __init__(self, manifest_path: Path | str) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise SharedT1AnchorError(
                f"shared T1 anchor manifest not found: {self.manifest_path}"
            )
        try:
            self.payload = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SharedT1AnchorError(
                f"invalid shared T1 anchor manifest: {self.manifest_path}"
            ) from exc
        self.schema_version = int(self.payload.get("schema_version", -1))
        if self.schema_version not in {1, 2}:
            raise SharedT1AnchorError(
                "shared T1 anchor schema_version must be 1 or 2"
            )
        self.anchor_set_id = str(self.payload.get("anchor_set_id", "")).strip()
        if not self.anchor_set_id:
            raise SharedT1AnchorError("shared T1 anchor_set_id is required")
        self.families = self.payload.get("families")
        if not isinstance(self.families, dict) or not self.families:
            raise SharedT1AnchorError("shared T1 manifest has no families")
        declared = int(self.payload.get("family_count", -1))
        if declared != len(self.families):
            raise SharedT1AnchorError(
                f"shared T1 family_count={declared} but found {len(self.families)}"
            )
        self._verified: dict[str, dict[str, Any]] = {}
        for family_id, record in self.families.items():
            self._verified[str(family_id)] = self._verify_family(
                str(family_id), record
            )

    def _verify_family(self, family_id: str, record: Any) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise SharedT1AnchorError(f"anchor {family_id} must be an object")
        task_id = str(record.get("task_id", ""))
        skill_id = str(record.get("skill_id", ""))
        if task_id != f"{family_id}-T1":
            raise SharedT1AnchorError(
                f"anchor {family_id} has non-T1 task_id {task_id!r}"
            )
        if not skill_id.startswith(f"{family_id}."):
            raise SharedT1AnchorError(
                f"anchor {family_id} has cross-family skill_id {skill_id!r}"
            )
        files = record.get("files")
        if not isinstance(files, dict) or not files:
            raise SharedT1AnchorError(f"anchor {family_id} has no files")
        loaded: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for rel_raw, expected_raw in sorted(files.items()):
            rel = _safe_relative_path(str(rel_raw))
            expected = str(expected_raw)
            source = self.manifest_path.parent / "families" / family_id / rel
            if not source.is_file():
                raise SharedT1AnchorError(f"anchor file missing: {source}")
            data = source.read_bytes()
            actual = _sha256(data)
            if actual != expected:
                raise SharedT1AnchorError(
                    f"anchor hash mismatch for {family_id}/{rel}: "
                    f"expected {expected}, got {actual}"
                )
            try:
                loaded[rel.as_posix()] = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SharedT1AnchorError(
                    f"anchor file must be UTF-8 text: {source}"
                ) from exc
            hashes[rel.as_posix()] = actual
        if not any(path.endswith("/SKILL.md") for path in loaded):
            raise SharedT1AnchorError(f"anchor {family_id} lacks SKILL.md")
        replay_record = None
        state_hash = ""
        state_file = str(record.get("t1_state_file", "")).strip()
        if state_file:
            state_rel = _safe_relative_path(state_file)
            state_path = self.manifest_path.parent / "families" / family_id / state_rel
            if not state_path.is_file():
                raise SharedT1AnchorError(f"shared T1 state missing: {state_path}")
            state_data = state_path.read_bytes()
            state_hash = _sha256(state_data)
            expected_state_hash = str(record.get("t1_state_sha256", ""))
            if state_hash != expected_state_hash:
                raise SharedT1AnchorError(
                    f"shared T1 state hash mismatch for {family_id}: expected "
                    f"{expected_state_hash}, got {state_hash}"
                )
            try:
                replay_record = ReplayRecord.model_validate_json(state_data)
            except ValueError as exc:
                raise SharedT1AnchorError(
                    f"invalid shared T1 replay state for {family_id}"
                ) from exc
            if (
                replay_record.task_id != task_id
                or replay_record.family_id != family_id
                or replay_record.task_role != "canonical"
                or replay_record.replay_mode != "shared_t1_anchor"
            ):
                raise SharedT1AnchorError(
                    f"shared T1 replay state identity mismatch for {family_id}"
                )
        elif self.schema_version >= 2:
            raise SharedT1AnchorError(
                f"schema v2 anchor {family_id} lacks t1_state_file"
            )
        canonical_payload = {
            "family_id": family_id,
            "task_id": task_id,
            "skill_id": skill_id,
            "files": hashes,
        }
        if self.schema_version >= 2:
            canonical_payload["t1_state_sha256"] = state_hash
        canonical = json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        actual_anchor_hash = _sha256(canonical)
        expected_anchor_hash = str(record.get("anchor_sha256", ""))
        if actual_anchor_hash != expected_anchor_hash:
            raise SharedT1AnchorError(
                f"anchor record hash mismatch for {family_id}: expected "
                f"{expected_anchor_hash}, got {actual_anchor_hash}"
            )
        return {
            **record,
            "family_id": family_id,
            "task_id": task_id,
            "skill_id": skill_id,
            "loaded_files": loaded,
            "verified_file_hashes": hashes,
            "anchor_sha256": actual_anchor_hash,
            "replay_record": replay_record,
            "t1_state_sha256": state_hash,
        }

    def anchor_for(
        self, *, family_id: str, task_id: str, latent_skill_id: str,
    ) -> SharedT1Anchor:
        record = self._verified.get(family_id)
        if record is None:
            raise SharedT1AnchorError(
                f"anchor set {self.anchor_set_id!r} has no family {family_id}"
            )
        if task_id != record["task_id"]:
            raise SharedT1AnchorError(
                f"anchor {family_id} expected task {record['task_id']}, got {task_id}"
            )
        if latent_skill_id != record["skill_id"]:
            raise SharedT1AnchorError(
                f"anchor {family_id} expected latent skill "
                f"{record['skill_id']}, got {latent_skill_id}"
            )
        patch_id = str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"skillevolbench:shared-t1:{self.anchor_set_id}:{family_id}",
        ))
        patch = SkillPatch(
            patch_id=patch_id,
            summary=(
                f"Load shared post-T1 anchor {self.anchor_set_id} for {family_id}"
            ),
            upsert_files=dict(record["loaded_files"]),
            target_skill_ids=[record["skill_id"]],
            operation_type=OperationType.CREATE,
            triggered_by_task=task_id,
        )
        audit = {
            "anchor_set_id": self.anchor_set_id,
            "manifest_path": str(self.manifest_path),
            "source_run_id": self.payload.get("source_run_id", ""),
            "source_benchmark_hash": self.payload.get(
                "source_benchmark_hash", ""
            ),
            "family_id": family_id,
            "task_id": task_id,
            "skill_id": record["skill_id"],
            "source_commit": record.get("source_commit", ""),
            "source_patch_id": record.get("source_patch_id", ""),
            "anchor_sha256": record["anchor_sha256"],
            "t1_state_sha256": record["t1_state_sha256"],
            "file_hashes": record["verified_file_hashes"],
            "t1_agent_saw_anchor": False,
            "t1_agent_executed_in_current_run": False,
        }
        return SharedT1Anchor(
            patch=patch,
            replay_record=record["replay_record"],
            audit=audit,
        )


__all__ = [
    "SharedT1Anchor",
    "SharedT1AnchorError",
    "SharedT1AnchorStore",
]
