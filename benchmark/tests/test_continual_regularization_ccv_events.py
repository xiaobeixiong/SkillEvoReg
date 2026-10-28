import json

from continual_regularization.ccv import parse_codex_event_stream


def event(kind):
    return json.dumps({'type': kind})


def test_malformed_intermediate_event_does_not_hide_valid_completion():
    stdout = '\n'.join([
        event('item.started'),
        '{"type":"item.completed","item":{"aggregated_output":"unterminated}',
        event('turn.completed'),
    ])
    events, malformed, completed = parse_codex_event_stream(stdout)
    assert [row['type'] for row in events] == ['item.started', 'turn.completed']
    assert len(malformed) == 1
    assert malformed[0]['lifecycle'] is False
    assert completed is True


def test_valid_turn_failed_remains_fail_closed():
    stdout = '\n'.join([event('item.started'), event('turn.failed')])
    _, _, completed = parse_codex_event_stream(stdout)
    assert completed is False


def test_malformed_lifecycle_event_remains_fail_closed():
    stdout = '\n'.join([
        '{"type":"turn.failed","error":"unterminated}',
        event('turn.completed'),
    ])
    _, malformed, completed = parse_codex_event_stream(stdout)
    assert malformed[0]['lifecycle'] is True
    assert completed is False
