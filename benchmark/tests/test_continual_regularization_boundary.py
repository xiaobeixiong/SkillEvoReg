import json
import pytest
from continual_regularization.attacks import proposal_error, numeric_equal, BOUNDARY_CHECKS
from continual_regularization.ccv import MathCCV


CASES = [
    ('sin u=0 has 2N-1 roots on any open interval of N periods',
     'How many roots of sin(u)=0 lie in (pi/2,5*pi/2)?', 2,
     'Roots are pi and 2*pi, both strictly inside.'),
    ('f-m is a monic quadratic square',
     'For f(x)=2*(x*x-1)**2, find a in f(x)=a*(x-1)**2*(x+1)**2.', 2,
     'Factoring x*x-1=(x-1)*(x+1) leaves coefficient 2.'),
    ('root mapping suffices for polynomial divisibility',
     'P(x)=x*x*(x-1), Q(x)=1-x. Does P(x) divide P(Q(x))? Return 1 for yes, 0 for no.', 0,
     'P(Q(x))=-x*(1-x)**2 has multiplicity one at zero, but P has two.')]


def proposal(case):
    rule, question, answer, derivation = case
    return dict(attack=True, kind='boundary_counterexample', responsible_paths=['s/SKILL.md'],
        rule_quote=rule, assumption='missing precondition', mechanism='boundary violates shortcut',
        equivalence_argument='Conditions changed; use a new oracle.', expected_answer=answer,
        oracle_derivation=derivation,
        attacked_instruction=question + ' Write JSON {"answer":number} to /app/scap_answer.json.')


@pytest.mark.parametrize('case', CASES)
def test_boundary_new_oracle_is_shared_without_leaking_to_solver(tmp_path, case):
    p = proposal(case)
    requests, slots = [], []
    replies = iter([p, dict(solvable=True, answer=case[2], derivation=case[3]),
                    dict(valid=True, **{k: True for k in BOUNDARY_CHECKS})])
    def model(request):
        requests.append(request)
        return next(replies)
    ccv = MathCCV(model, tmp_path, 10)
    ccv.clean_slot = lambda *a: dict(status='ok', passed=True, score=1.0)
    def slot(files, instruction, source, name):
        slots.append((files, instruction, source, name))
        return dict(status='ok', passed=True, score=1)
    ccv.slot = slot
    result = ccv(old={'s/SKILL.md': 'careful general rule'}, candidate={'s/SKILL.md': case[0]},
        source=dict(instruction='Original question /app/task.json', grading={'old_answer': 999}))
    assert result['status'] == 'paired'
    assert len(slots) == 2 and slots[0][2] == slots[1][2]
    assert slots[0][2]['grading']['expected_answer'] == case[2]
    assert 'old_answer' not in slots[0][2]['grading']
    assert set(requests[1]) == {'instruction', 'question'}
    assert 'oracle_derivation' not in slots[0][1]


def test_disagreement_never_scores_or_repairs(tmp_path):
    p = proposal(CASES[0])
    replies = iter([p, dict(solvable=True, answer=99, derivation='wrong')])
    ccv = MathCCV(lambda _: next(replies), tmp_path, 10)
    ccv.clean_slot = lambda *a: dict(status='ok', passed=True, score=1.0)
    ccv.slot = lambda *a: pytest.fail('invalid oracle must not be evaluated')
    result = ccv(old={}, candidate={'s/SKILL.md': CASES[0][0]}, source={'instruction': 'original'})
    assert result['reason'] == 'oracle_disagreement'


def test_numeric_grader_strict_types_and_finiteness():
    assert numeric_equal(2, 2.0)
    for value in (True, '2', None, float('nan'), float('inf')):
        assert not numeric_equal(value, 2)


def test_new_grading_requires_derivation():
    p = proposal(CASES[0])
    p.pop('oracle_derivation')
    assert proposal_error(p, {'s/SKILL.md': {'old': '', 'new': CASES[0][0]}},
                          {'instruction': 'original'}) == 'missing oracle_derivation'


def test_missing_output_contract_is_attached_but_foreign_path_not_rewritten():
    from continual_regularization.attacks import normalize_output_contract
    p = dict(kind='boundary_counterexample', attacked_instruction='Compute 2+2.')
    assert normalize_output_contract(p)['attacked_instruction'].startswith('Compute 2+2.\n')
    assert '/app/scap_answer.json' in normalize_output_contract(p)['attacked_instruction']
    p['attacked_instruction'] = 'Compute 2+2. Write /app/wrong.json'
    assert normalize_output_contract(p) == p
