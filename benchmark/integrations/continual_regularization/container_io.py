"""Small container-side library transaction, supplied as an exec script."""
import json
import os
from pathlib import Path
import base64

ROOTS = ['/root/.agents/skills', '/root/.claude/skills', '/root/.terminus/skills',
         '/root/.codex/skills', '/root/.opencode/skill', '/root/.goose/skills',
         '/root/.factory/skills', '/root/.gemini/skills']


def read(root):
    result = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in regularized library')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes().decode('utf-8', 'surrogateescape')
    return result


def main():
    if os.environ.get('CONTINUAL_IO_PATH'):
        payload = Path(os.environ['CONTINUAL_IO_PATH'])
        request = json.loads(payload.read_text())
        payload.unlink()
    else:
        request = json.loads(base64.b64decode(os.environ['CONTINUAL_IO']))
    before = {root: read(root) for root in ROOTS}
    if request['op'] == 'read':
        print(json.dumps(before))
        return
    changes = request['changes']
    for p in changes:
        parts = Path(p).parts
        if len(parts) < 2 or Path(p).is_absolute() or '..' in parts or parts[0] in {'create-skill', 'modify-skill'}:
            raise ValueError('Out-of-scope library write')
    if before != request['expected']:
        raise ValueError('Library changed concurrently; transaction refused')
    # No agent command runs during this boundary. Journal is retained on host.
    for root in ROOTS:
        for p, text in changes.items():
            target = Path(root) / p
            if text is None:
                target.unlink(missing_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + '.regularization-tmp')
                temp.write_bytes(text.encode('utf-8', 'surrogateescape'))
                temp.replace(target)
    print(json.dumps({'status': 'committed'}))


if __name__ == '__main__':
    main()
