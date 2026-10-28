from continual_regularization import ccv


def test_oversized_attack_is_explicit_no_attack(monkeypatch):
    monkeypatch.setattr(ccv, 'ATTACK_INPUT_TOKEN_BUDGET', 10)
    result = ccv.attack_budget_check({'source_trajectory': 'large output ' * 100})
    assert result['status'] == 'no_attack'
    assert result['reason'] == 'input_budget_exceeded'
    assert result['input_tokens'] > result['input_token_budget']


def test_small_request_unchanged():
    request = {'source': 'Question', 'delta': {}, 'source_trajectory': 'short'}
    assert ccv.attack_budget_check(request) is None
    assert request['source_trajectory'] == 'short'
