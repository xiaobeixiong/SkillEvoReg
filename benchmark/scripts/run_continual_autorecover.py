"""Bounded whole-task recovery; preserve failed segments, never reuse dirty state."""
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml
from continual_regularization.checkpoint import load_checkpoint


def retryable(text):
    text = text.lower()
    return any(s in text for s in (
        'invalid_encrypted_content', 'encrypted content could not be decrypted',
        '502 bad gateway', '503 service unavailable', '504 gateway timeout',
        'status code 502', 'status code 503', 'status code 504',
        'stream disconnected', 'stream closed before response.completed',
        'exceeded rate limit', 'too many requests', 'connection reset',
    ))


def latest_checkpoint(config, output):
    candidates = [Path(config['agents'][0]['kwargs']['resume_checkpoint'])]
    candidates.extend(output.glob('*/agent/checkpoints/step_*'))
    # Fail closed on invalid published checkpoints; never silently skip corruption.
    valid = [load_checkpoint(p) for p in candidates if not p.name.endswith('.pending')]
    return max(valid, key=lambda pair: pair[1]['completed_step'])


def failure_reports(output):
    """Inspect terminal trial results, not historical errors in successful logs.

    SDK exceptions and preflight failures never reach CheckedEnvironment's
    infrastructure_failure.json. Unknown exceptions must still fail closed.
    """
    reports = []
    for path in sorted(output.glob('*/result.json')):
        result = json.loads(path.read_text())
        error = result.get('exception_info')
        if not error:
            continue
        message = error.get('exception_message', '')
        evidence = None
        if message == 'Container API preflight failed; no training started':
            evidence = path.parent / 'agent/preflight_error.txt'
        elif message == 'Continual command failed; stopping before further skill updates':
            evidence = path.parent / 'agent/infrastructure_failure.json'
        text = evidence.read_text() if evidence and evidence.exists() else message
        reports.append(dict(trial=str(path.parent), exception_type=error.get('exception_type'),
                            evidence=str(evidence or path), retryable=retryable(text)))
    return reports


def main():
    path = Path(sys.argv[1]).resolve()
    base = yaml.safe_load(path.read_text())['job_name']
    for attempt in range(4):
        config = yaml.safe_load(path.read_text())
        output = Path(config['jobs_dir']) / config['job_name']
        if output.exists():
            raise RuntimeError(f'Refusing to overwrite {output}')
        result = subprocess.run([sys.executable, '-u', '-m', 'harbor.cli.main', 'run', '-c', str(path)])
        if result.returncode < 0:
            raise SystemExit(result.returncode)
        failures = failure_reports(output)
        if not failures:
            summary = output / 'result.json'
            if not summary.exists():
                raise SystemExit('Stopped: missing job result; cannot confirm completion')
            data = json.loads(summary.read_text())
            if not data.get('finished_at') or data.get('stats', {}).get('n_errors', 0):
                raise SystemExit('Stopped: incomplete job or unclassified error; artifacts preserved')
            raise SystemExit(result.returncode)
        if attempt == 3 or not all(f['retryable'] for f in failures):
            raise SystemExit('Stopped: retry budget exhausted or non-transport failure; artifacts preserved')
        checkpoint, data = latest_checkpoint(config, output)
        config['job_name'] = f'{base}_auto{attempt + 1:02d}'
        config['agents'][0]['kwargs']['resume_checkpoint'] = str(checkpoint)
        dest = path.parent / f'auto{attempt + 1:02d}.yaml'
        if dest.exists():
            raise RuntimeError(f'Refusing to overwrite {dest}')
        dest.write_text(yaml.safe_dump(config, sort_keys=False))
        (output / 'auto_recovery.json').write_text(json.dumps(dict(
            checkpoint=str(checkpoint), completed_step=data['completed_step'],
            next_config=str(dest), failed_segment=str(output), attempt=attempt + 1,
            failures=failures)))
        delay = (30, 90, 180)[attempt]
        print(f'[Recovery] task {data["completed_step"] + 1}: new container from complete checkpoint; wait {delay}s', flush=True)
        time.sleep(delay)
        path = dest


if __name__ == '__main__':
    main()
