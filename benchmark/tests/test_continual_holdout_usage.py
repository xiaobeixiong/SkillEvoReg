import json
from types import SimpleNamespace
from continual_holdout.usage import record_usage


def output(thread, inp, out):
    return SimpleNamespace(stdout='\n'.join(map(json.dumps, [
        {'type':'thread.started', 'thread_id':thread},
        {'type':'turn.completed','usage':{'input_tokens':inp,'output_tokens':out,'cached_input_tokens':0}}
    ])))


def test_initialization_subtracted_and_fresh_retry_independent(tmp_path):
    wrapper = SimpleNamespace(logs_dir=tmp_path)
    record_usage(wrapper, output('a', 20, 2), '初始化会话', 0, '')
    record_usage(wrapper, output('a', 120, 12), '做题', 1, '/app/task_051_output.json')
    record_usage(wrapper, output('fresh', 200, 20), '做题', 2, '/app/task_052_output.json')
    rows = [json.loads(s) for s in (tmp_path/'solve_token_usage.jsonl').read_text().splitlines()]
    assert [r['total_tokens'] for r in rows] == [110,220]
    assert [r['task_id'] for r in rows] == [51,52]
    assert rows[0]['initialization_subtracted']
    assert not rows[1]['initialization_subtracted']


def test_missing_usage_not_zero(tmp_path):
    record_usage(SimpleNamespace(logs_dir=tmp_path), SimpleNamespace(stdout=''), '做题', 1, '')
    row=json.loads((tmp_path/'solve_token_usage.jsonl').read_text())
    assert not row['usage_available']
    assert 'total_tokens' not in row
