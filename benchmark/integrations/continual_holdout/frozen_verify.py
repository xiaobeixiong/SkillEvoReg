"""Container-side, read-only integrity check of the frozen library.

Runs inside the task container after the agent phase.  It recomputes exactly the same
digest the provisioning step reported, so the host can prove that the library the
agent was graded against is still the library that was declared -- no skill was
created, modified or deleted mid-run.
"""
import hashlib
import json
import os
import pathlib

HELPERS = {"create-skill", "modify-skill"}
CANONICAL = "/root/.agents/skills"


def digest(root):
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
    canonical_digest, files = digest(CANONICAL)
    report = {
        "canonical_digest": canonical_digest,
        "n_files": len(files),
        "skills": sorted({name.split("/")[0] for name in files}),
        "roots": {},
    }
    for root in roots:
        root_digest, root_files = digest(root)
        report["roots"][root] = {"digest": root_digest, "n_files": len(root_files)}
    report["helpers_present"] = sorted(
        child.name for child in pathlib.Path(CANONICAL).iterdir() if child.name in HELPERS
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
