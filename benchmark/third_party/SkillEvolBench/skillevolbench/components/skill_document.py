"""Structural validation shared by every SkillEvolBench skill write path."""
from __future__ import annotations

import re

from skillevolbench.discovery import parse_skill_md_frontmatter_text


_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")


def _unclosed_fence(text: str) -> tuple[str, int] | None:
    opened: tuple[str, int] | None = None
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        if match is None:
            continue
        run = match.group(1)
        char = run[0]
        if opened is None:
            opened = (char, len(run))
            continue
        if char == opened[0] and len(run) >= opened[1] and not match.group(2).strip():
            opened = None
    return opened


def _fences_balanced(text: str) -> bool:
    return _unclosed_fence(text) is None


def close_dangling_markdown_fence(text: str) -> tuple[str, bool]:
    """Close one mechanically identifiable trailing Markdown fence.

    LLM skill revisions occasionally omit only the final closing delimiter.
    Rejecting the whole semantic update for that formatting omission changes
    the evolution trajectory, while persisting the raw document leaves the
    remainder parsed as code.  Appending the matching delimiter is the unique
    content-preserving repair: it changes no existing byte and only terminates
    the already-open fenced block at end of document.

    The write boundary remains strict.  Callers must validate the returned
    text, and documents with any remaining structural error are still refused.
    """
    opened = _unclosed_fence(text)
    if opened is None:
        return text, False
    char, width = opened
    suffix = "" if text.endswith("\n") else "\n"
    return f"{text}{suffix}{char * width}\n", True


def validate_skill_document(
    text: str,
    *,
    source: str = "SKILL.md",
    expected_slug: str | None = None,
    allow_dropout_placeholders: bool = False,
) -> list[str]:
    """Return persistence-blocking errors for a complete ``SKILL.md``."""
    errors: list[str] = []
    if not text.strip():
        return [f"{source}: skill document is empty"]
    try:
        frontmatter = parse_skill_md_frontmatter_text(text, source=source)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    else:
        name = str(frontmatter.get("name", "")).strip()
        description = str(frontmatter.get("description", "")).strip()
        if not name:
            errors.append(f"{source}: frontmatter is missing name")
        if not description:
            errors.append(f"{source}: frontmatter is missing description")
    if not _fences_balanced(text):
        errors.append(f"{source}: unbalanced Markdown fenced code block")
    if not allow_dropout_placeholders and "MASKED_SKILL_ATOM" in text:
        errors.append(f"{source}: contains a dropout placeholder")
    return errors


def validate_skill_upserts(
    upserts: dict[str, str],
    *,
    allow_dropout_placeholders: bool = False,
) -> list[str]:
    """Validate every complete SKILL.md carried by a patch."""
    errors: list[str] = []
    for path, content in upserts.items():
        if path.endswith("/SKILL.md"):
            errors.extend(validate_skill_document(
                content,
                source=path,
                expected_slug=path.split("/", 1)[0],
                allow_dropout_placeholders=allow_dropout_placeholders,
            ))
    return errors


__all__ = [
    "close_dangling_markdown_fence",
    "validate_skill_document",
    "validate_skill_upserts",
]
