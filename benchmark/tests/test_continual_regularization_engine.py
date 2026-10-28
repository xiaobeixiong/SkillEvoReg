from continual_regularization.engine import Settings, StepRegularization


def test_soft_budget_keeps_repair_and_no_second_compression():
    calls = []
    def quick(request):
        calls.append(request)
        return {'operation': 'REWRITE', 'changes': {'new/SKILL.md': 'still long'}}
    regularization = StepRegularization(Settings(1, .15, 2, 4, .1),
        measure=lambda s: dict(total=len(s), atom_count=1 if s else 0),
        retrieve=lambda **kw: {'history': []}, local_edit=quick,
        ccv=lambda **kw: {'status': 'no_attack'}, repair_regression=lambda **kw: None)
    result, events = regularization.finish({}, {'new/SKILL.md': 'long source'}, masked_paths=set(), source={})
    assert len(calls) == 1
    assert calls[0]['modules'] == ('complexity',)
    assert result['new/SKILL.md'] == 'still long'
    assert events[0]['still_over_budget']


def test_dropout_only_feedback_no_compression_and_ccv_uses_regularized():
    seen = []
    def quick(request):
        assert request['modules'] == ('dropout',)
        assert 'Complexity feedback:' not in request['instruction']
        return {'operation': 'NOOP'}
    def ccv(**kwargs):
        seen.append(kwargs)
        return {'status': 'inconclusive'}
    regularization = StepRegularization(Settings(100, .15, 100, 4, .1),
        measure=lambda s: dict(total=len(s), atom_count=1),
        retrieve=lambda **kw: {'history': []}, local_edit=quick,
        ccv=ccv, repair_regression=lambda **kw: (_ for _ in ()).throw(AssertionError()))
    old = {'old/SKILL.md': 'original'}
    raw = dict(old, **{'new/SKILL.md': 'new'})
    result, events = regularization.finish(old, raw, masked_paths={'old/SKILL.md'}, source={'id': 1})
    assert result == raw
    assert seen[0]['old'] == old and seen[0]['candidate'] == raw
    assert events[-1]['status'] == 'inconclusive'
