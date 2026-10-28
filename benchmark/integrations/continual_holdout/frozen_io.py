"""Container-side frozen-library provisioning, supplied to the sandbox as an exec script.

Runs inside the task container.  The roots are injected from the host through
``CONTINUAL_FROZEN_ROOTS`` so that the canonical list in
``continual_regularization.container_io`` stays the single source of truth.

Behaviour
---------
1. Prune every non-helper entry under each root, so a condition is exactly its
   declared library and never inherits leftovers from the image.
2. Unpack the frozen archive into every root.
3. Re-read the canonical root from disk and report a content digest, so the host
   can prove that what the agent sees is what was declared.

The two native helpers (``create-skill`` / ``modify-skill``) ship inside the image
and are never part of a library; they are preserved in place.
"""
import hashlib
import json
import os
import pathlib
import shutil
import tarfile

HELPERS = {"create-skill", "modify-skill"}
CANONICAL = "/root/.agents/skills"
STAGE = "/tmp/continual-frozen-stage"


def _entries(root):
    base = pathlib.Path(root)
    if not base.is_dir():
        return []
    return [child for child in sorted(base.iterdir()) if child.name not in HELPERS]


def _digest(root):
    """Order-independent digest of the whole library, helpers excluded."""
    base = pathlib.Path(root)
    accumulator = hashlib.sha256()
    files = sorted(
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path.relative_to(base).parts[0] not in HELPERS
    )
    for relative in files:
        accumulator.update(relative.encode("utf-8"))
        accumulator.update(b"\0")
        accumulator.update((base / relative).read_bytes())
        accumulator.update(b"\0")
    return accumulator.hexdigest(), files


def main():
    roots = json.loads(os.environ["CONTINUAL_FROZEN_ROOTS"])
    archive_path = os.environ["CONTINUAL_FROZEN_ARCHIVE"]

    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(STAGE)
    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            archive.extractall(STAGE, filter="data")
        except TypeError:  # Python < 3.12 has no extraction filter argument
            archive.extractall(STAGE)

    payload = sorted(path for path in pathlib.Path(STAGE).rglob("*") if path.is_file())
    suppressed = sum(1 for path in payload if path.relative_to(STAGE).parts[0] in HELPERS)
    payload = [path for path in payload if path.relative_to(STAGE).parts[0] not in HELPERS]

    report = {"pruned": {}, "written": {}, "helpers_suppressed": suppressed}
    for root in roots:
        pruned = 0
        for entry in _entries(root):
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
            pruned += 1
        report["pruned"][root] = pruned

        written = 0
        for source in payload:
            relative = source.relative_to(STAGE)
            target = pathlib.Path(root) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            written += 1
        report["written"][root] = written

    digest, files = _digest(CANONICAL)
    report["canonical_digest"] = digest
    report["files"] = files
    report["n_files"] = len(files)
    report["skills"] = sorted({name.split("/")[0] for name in files})
    report["n_skills"] = len(report["skills"])

    present = sorted(child.name for child in pathlib.Path(CANONICAL).iterdir())
    report["helpers_present"] = [name for name in present if name in HELPERS]
    print(json.dumps(report))


if __name__ == "__main__":
    main()
