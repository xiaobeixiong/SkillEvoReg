import json
from continual_regularization.attacks import delta_context, select_source, proposal_error, validated, CHECKS
from continual_regularization.ccv import MathCCV


def test_source_relevance_not_recency_or_failed_score():
    pool = [dict(task_id=1, training_score=1, instruction='relevant'),
            dict(task_id=2, training_score=1, instruction='recent'),
            dict(task_id=3, training_score=0, instruction='failed')]
    delta = delta_context({'s/SKILL.md': 'old'}, {'s/SKILL.md': 'new'})
    selected, ranking = select_source(pool, delta, similarity=lambda a,b: [[.9,.2] for _ in a])
    assert selected['task_id'] == 1
    assert len(ranking) == 2


def test_exact_step60_invalid_path_does_not_crash(tmp_path):
    # Same malformed responsibility shape as the recorded step60 failure.
    malformed = dict(attack=True, cue='bounded scan', responsible_paths=['source_trajectory:item_7'])
    calls = []
    def model(request):
        calls.append(request)
        return malformed
    ccv = MathCCV(model, tmp_path, 600)
    ccv.clean_slot = lambda *a: dict(status='ok', passed=True, score=1.0)
    ccv.slot = lambda *a: (_ for _ in ()).throw(AssertionError('must not evaluate invalid attack'))
    result = ccv(old={'s/SKILL.md': 'old'}, candidate={'s/SKILL.md': 'new'}, source={'instruction': 'task'})
    assert result['reason'] == 'invalid_proposal'
    assert len(calls) == 2
    assert 'validation_error' in calls[1]


def test_validation_requires_all_equivalence_checks():
    assert not validated({'valid': True})
    result = dict(valid=True, **{k: True for k in CHECKS})
    assert validated(result)
    result['same_answer'] = False
    assert not validated(result)


def test_schema_rejects_unchanged_and_output_path_changes():
    delta = delta_context({}, {'s/SKILL.md': 'Use digits'})
    p = dict(attack=True, kind='symbol_renaming', responsible_paths=['s/SKILL.md'],
        rule_quote='Use digits', assumption='digit shortcut', mechanism='rename', equivalence_argument='same',
        attacked_instruction='new question /app/task_002_output.json')
    source = {'instruction': 'question /app/task_001_output.json'}
    assert proposal_error(p, delta, source) == 'output paths changed'
    p['attacked_instruction'] = source['instruction']
    assert proposal_error(p, delta, source) == 'transformation is unchanged'


def test_malformed_json_bounded_retry(tmp_path):
    calls = []
    def model(request):
        calls.append(request)
        raise json.JSONDecodeError('bad', '', 0)
    ccv = MathCCV(model, tmp_path, 600)
    ccv.clean_slot = lambda *a: dict(status='ok', passed=True, score=1.0)
    result = ccv(old={'s/SKILL.md': 'old'}, candidate={'s/SKILL.md': 'new'}, source={'instruction': 'task'})
    assert result['reason'] == 'invalid_proposal' and len(calls) == 2
