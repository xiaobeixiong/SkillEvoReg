"""Persist successful solve usage separately from init/judge/retry overhead."""
import json
import re


def record_usage(wrapper, result, phase, number, command):
    rows = []
    for line in (result.stdout or '').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    threads = [r['thread_id'] for r in rows if r.get('type') == 'thread.started']
    usages = [r['usage'] for r in rows if r.get('type') == 'turn.completed']
    if not hasattr(wrapper, '_usage_prefix'):
        wrapper._usage_prefix = {}
    if not threads or not usages:
        if phase == '做题':
            with (wrapper.logs_dir / 'solve_token_usage.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(task_position=number, usage_available=False))+'\n')
        return
    thread, usage = threads[-1], usages[-1]
    prefix = wrapper._usage_prefix.get(thread, {})
    wrapper._usage_prefix[thread] = usage
    if phase != '做题':
        return
    delta = {k: usage.get(k, 0)-prefix.get(k, 0)
             for k in ('input_tokens', 'output_tokens', 'cached_input_tokens')}
    task = re.search(r'/app/task_(\d+)_output\.json', command)
    valid = all(v >= 0 for v in delta.values())
    record = dict(task_position=number, task_id=int(task.group(1)) if task else None,
                  thread_id=thread, usage_available=valid, initialization_subtracted=bool(prefix),
                  **delta, total_tokens=delta['input_tokens']+delta['output_tokens'])
    with (wrapper.logs_dir / 'solve_token_usage.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
