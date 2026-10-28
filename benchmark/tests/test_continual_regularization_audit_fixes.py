from continual_regularization.engine import StepRegularization, Settings
from continual_regularization.dropout import restore_library


def build(**overrides):
    ports = dict(measure=lambda s: dict(total=len(s), atom_count=bool(s)),
        retrieve=lambda **kw: {'history': ['old/SKILL.md']},
        local_edit=lambda req: {'operation': 'NOOP'},
        ccv=lambda **kw: {'status': 'no_attack'}, repair_regression=lambda **kw: None)
    ports.update(overrides)
    return StepRegularization(Settings(1, .15, 40, 4, .1), **ports)


def test_receiving_budget_soft_and_logged():
    def update(req):
        assert req['receiving_skill_budgets']['old/SKILL.md']['budget']['total'] == 3 * 1.15
        return {'operation': 'MERGE', 'changes': {'old/SKILL.md': 'much longer', 'new/SKILL.md': None}}
    result, events = build(local_edit=update).finish({'old/SKILL.md': 'old'},
        {'old/SKILL.md': 'old', 'new/SKILL.md': 'new'}, masked_paths=set(), source=None)
    assert result == {'old/SKILL.md': 'much longer'}
    assert events[0]['receiving_skills']['old/SKILL.md']['still_over_budget']


def test_deleted_skill_can_be_repaired_after_confirmed_probe():
    pair = dict(status='ok', score=1, passed=True, probe_hash='p', grading_hash='g')
    probe = dict(status='paired', old=pair, new=dict(pair, score=0), responsible_paths=['old/SKILL.md'])
    def repair(candidate, probe):
        assert probe['deleted_skill_context'] == {'old/SKILL.md': 'useful'}
        return {'old/SKILL.md': 'necessary invariant'}
    result, events = build(ccv=lambda **kw: probe, repair_regression=repair).finish(
        {'old/SKILL.md': 'useful'}, {}, masked_paths=set(), source={'task_id': 1})
    assert result['old/SKILL.md'] == 'necessary invariant'
    assert events[0]['mode'] == 'delete'
    assert events[-1]['source_task_id'] == 1


def test_explicit_native_delete_not_restored_by_dropout():
    assert restore_library({}, {'s/SKILL.md': {'marker': 'original'}}, allow_deleted=True) == {}


def test_clean_cache_requires_exact_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from continual_regularization.ccv import MathCCV
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://example.invalid/v1')
    monkeypatch.setattr('continual_regularization.ccv.subprocess.run', lambda *a, **k: SimpleNamespace(stdout='sha256:fixed'))
    tests = tmp_path / 'tests'
    tests.mkdir()
    (tests / 'grader.py').write_text('v1')
    source = dict(tests=str(tests), task_id=1, grading={'answer': 'A'})
    calls = []
    def probe(*args):
        calls.append(args)
        return dict(status='ok', passed=True, score=1)
    for i, text in enumerate(['same', 'same', 'different']):
        ccv = MathCCV(None, tmp_path / 'regularization' / str(i) / 'ccv', 600)
        ccv.slot = probe
        ccv.clean_slot({'s/SKILL.md': text}, 'task', source, 'clean_old')
    assert len(calls) == 2
    (tests / 'grader.py').write_text('v2')
    ccv = MathCCV(None, tmp_path / 'regularization' / '3' / 'ccv', 600)
    ccv.slot = probe
    ccv.clean_slot({'s/SKILL.md': 'same'}, 'task', source, 'clean_old')
    assert len(calls) == 3
