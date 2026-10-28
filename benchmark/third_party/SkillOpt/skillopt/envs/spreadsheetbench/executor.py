"""Execute LLM-generated Python code against an input xlsx to produce an output xlsx."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap


RUNNER_TEMPLATE = textwrap.dedent(
    """
    import os, sys, traceback
    INPUT_PATH = {input_path!r}
    OUTPUT_PATH = {output_path!r}
    try:
    {user_code_indented}
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    """
)

# Regex to strip user-defined INPUT_PATH / OUTPUT_PATH assignments,
# since the runner template injects the correct values.
_PATH_ASSIGN_RE = re.compile(
    r'^\s*(INPUT_PATH|OUTPUT_PATH)\s*=\s*.+$', re.MULTILINE
)


def _strip_path_assignments(code: str) -> str:
    """Remove INPUT_PATH/OUTPUT_PATH assignments from user code."""
    return _PATH_ASSIGN_RE.sub("", code)


def run_generated_code(code: str, input_path: str, output_path: str, timeout: int | None = 120) -> tuple[bool, str]:
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    # Generated programs are untrusted.  Give each execution a private copy so
    # code that edits, removes, or moves INPUT_PATH cannot mutate the prepared
    # benchmark dataset used by later paired evaluations.
    suffix = os.path.splitext(input_path)[1] or ".xlsx"
    fd, isolated_input = tempfile.mkstemp(
        prefix=".skillopt_input_",
        suffix=suffix,
        dir=output_dir,
    )
    os.close(fd)
    shutil.copy2(input_path, isolated_input)
    cleaned = _strip_path_assignments(code)
    indented = textwrap.indent(cleaned, "    ")
    script = RUNNER_TEMPLATE.format(
        input_path=isolated_input,
        output_path=output_path,
        user_code_indented=indented,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout if timeout and timeout > 0 else None,
        )
        if proc.returncode != 0:
            return False, (proc.stdout + "\n" + proc.stderr).strip()
        if not os.path.exists(output_path):
            return False, "output file was not created"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        try:
            os.unlink(isolated_input)
        except OSError:
            pass
