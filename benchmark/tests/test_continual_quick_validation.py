import copy

import pytest

from continual_regularization.ports import quick
from continual_regularization.transactions import Window


def request():
    return dict(mode='modify', source_path='s/SKILL.md',
                candidate={'s/SKILL.md': 'old new'}, old={'s/SKILL.md': 'old'},
                matches={'history': [], 'windows': {
                    'source_0': Window('s/SKILL.md', 4, 7, 'new')}})


def test_retry_unknown_id_without_partial_application():
    req = request()
    before = copy.deepcopy(req)
    calls = []
    good = {'operation': 'REWRITE', 'changes': {'source_0': 'short'}}
    def model(payload):
        calls.append(copy.deepcopy(payload))
        return ({'operation': 'REWRITE', 'changes': {'source_0': 'partial', 'source_99': 'bad'}}
                if len(calls) == 1 else good)
    assert quick(model, req) == good
    assert req == before
    assert calls[1]['candidate'] == before['candidate']
    assert calls[0]['allowed_window_ids'] == ['source_0']
    assert 'source_99' in calls[1]['validation_feedback']['error']


def test_exhaustion_stops_without_noop_or_mutation():
    req = request()
    before = copy.deepcopy(req)
    calls = []
    def model(payload):
        calls.append(payload)
        return {'operation': 'REWRITE', 'changes': {'s/SKILL.md': 'bad'}}
    with pytest.raises(ValueError, match='invalid after 3 attempts'):
        quick(model, req)
    assert len(calls) == 3
    assert req == before


@pytest.mark.parametrize('response', [
    {'operation': 'NOOP', 'changes': {}},
    {'operation': 'REWRITE', 'changes': {'source_0': 'short'}},
])
def test_valid_response_single_call(response):
    calls = []
    def model(payload):
        calls.append(payload)
        return response
    assert quick(model, request()) is response
    assert len(calls) == 1


def test_transport_error_not_reinterpreted_as_format_error():
    def model(payload):
        raise RuntimeError('API unavailable')
    with pytest.raises(RuntimeError, match='API unavailable'):
        quick(model, request())
