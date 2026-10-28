from continual_regularization.ccv import MathCCV
import pytest


def test_no_training_success_does_not_probe(tmp_path):
    ccv = MathCCV(lambda _: (_ for _ in ()).throw(AssertionError()), tmp_path, 10)
    ccv.slot = lambda *args: (_ for _ in ()).throw(AssertionError())
    assert ccv(old={}, candidate={'s/SKILL.md': 'x'}, source=None)['status'] == 'no_attack'


def test_no_attack_not_forced(tmp_path):
    ccv = MathCCV(lambda _: {'attack': False}, tmp_path, 10)
    seen = []
    def slot(*args):
        seen.append(args)
        return {'status': 'ok', 'passed': True, 'score': 1}
    ccv.slot = slot
    ccv.clean_slot = slot
    out = ccv(old={'s/SKILL.md': 'old'}, candidate={'s/SKILL.md': 'new'}, source={'instruction': 'task'})
    assert out['status'] == 'no_attack'
    assert len(seen) == 2


def test_attack_slots_preserve_original_task(tmp_path):
    from continual_regularization.attacks import CHECKS
    replies = iter([{'attack': True, 'kind': 'equivalent_rephrasing',
        'attacked_instruction': 'Add two and two. Output JSON.', 'responsible_paths': ['s/SKILL.md'],
        'rule_quote': 'new', 'assumption': 'digits required', 'mechanism': 'worded numbers',
        'equivalence_argument': 'Same addition and output.'}, dict(valid=True, **{k: True for k in CHECKS})])
    ccv = MathCCV(lambda _: next(replies), tmp_path, 10)
    seen = []
    def slot(files, instruction, source, name):
        seen.append((files, instruction, source, name))
        return {'status': 'ok', 'passed': True, 'score': 1}
    ccv.slot = slot
    ccv.clean_slot = slot
    source = {'instruction': 'Compute 2+2. Output JSON.'}
    old, new = {'s/SKILL.md': 'old'}, {'s/SKILL.md': 'new'}
    out = ccv(old=old, candidate=new, source=source)
    assert out['status'] == 'paired'
    assert len(seen) == 4
    assert seen[2][1] == seen[3][1]
    assert seen[2][1] == 'Add two and two. Output JSON.'
    assert 'Irrelevant background' not in seen[2][1]
    assert seen[2][0] == old and seen[3][0] == new


def test_api_error_does_not_trigger_attack(tmp_path):
    ccv = MathCCV(lambda _: (_ for _ in ()).throw(AssertionError()), tmp_path, 10)
    ccv.slot = lambda *args: {'status': 'error'}
    ccv.clean_slot = ccv.slot
    assert ccv(old={'a': 'x'}, candidate={'a': 'y'}, source={'instruction': 'task'})['status'] == 'inconclusive'


@pytest.mark.parametrize('wrapped', [True, False])
def test_repair_accepts_both_shapes_without_extra_call(tmp_path, wrapped):
    mapping = {'s/SKILL.md': 'repaired invariant'}
    calls = []
    def model(request):
        calls.append(request)
        return {'skills': mapping} if wrapped else mapping
    ccv = MathCCV(model, tmp_path, 10)
    candidate = {'s/SKILL.md': 'old', 'other/SKILL.md': 'untouched'}
    result = ccv.repair(candidate=candidate, probe={'responsible_paths': ['s/SKILL.md']})
    assert result == dict(candidate, **mapping)
    assert candidate['s/SKILL.md'] == 'old'
    assert len(calls) == 1


@pytest.mark.parametrize('reply', [{}, [], {'skills': []}, {'skills': {}},
    {'s/SKILL.md': ''}, {'s/SKILL.md': None}, {'s/SKILL.md': 2},
    {'other/SKILL.md': 'bad'}, {'skills': {'s/SKILL.md': 'x'}, 'other/SKILL.md': 'bad'}])
def test_repair_invalid_retries_bounded_without_mutation(tmp_path, reply):
    calls=[]
    def model(request): calls.append(request); return reply
    ccv=MathCCV(model,tmp_path,10)
    candidate={'s/SKILL.md':'original'}
    with pytest.raises(ValueError,match='after 3 attempts'):
        ccv.repair(candidate=candidate,probe={'responsible_paths':['s/SKILL.md']})
    assert len(calls)==3 and candidate=={'s/SKILL.md':'original'}


def test_repair_recovers_on_second_reply(tmp_path):
    replies=iter([{}, {'skills': {'s/SKILL.md':'valid'}}])
    ccv=MathCCV(lambda _:next(replies),tmp_path,10)
    assert ccv.repair(candidate={},probe={'responsible_paths':['s/SKILL.md']})=={'s/SKILL.md':'valid'}
