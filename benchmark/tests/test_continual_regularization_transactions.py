import pytest
from continual_regularization.transactions import Window, apply_repair, paired_regression
from continual_regularization.dropout import mask_library, restore_library
from types import SimpleNamespace


def test_noop_preserves_delta():
    original = {'s/SKILL.md': 'new delta'}
    assert apply_repair(original, {'operation': 'NOOP'}, source='s/SKILL.md',
                        history=set(), mode='create') == original


def test_rewrite_cannot_change_history():
    with pytest.raises(ValueError, match='scope'):
        apply_repair({'s/SKILL.md': 'new', 'h/SKILL.md': 'old'},
                     {'operation': 'REWRITE', 'changes': {'h/SKILL.md': 'bad'}},
                     source='s/SKILL.md', history={'h/SKILL.md'}, mode='create')


def test_merge_cannot_orphan_resources():
    with pytest.raises(ValueError, match='auxiliary'):
        apply_repair({'s/SKILL.md': 'new', 's/tool.py': 'code'},
                     {'operation': 'MERGE', 'changes': {'s/SKILL.md': None}},
                     source='s/SKILL.md', history=set(), mode='create')


def test_modify_preserves_outside_windows():
    files = {'s/SKILL.md': 'header\nlong delta\nfooter'}
    windows = {'d': Window('s/SKILL.md', 7, 17, 'long delta')}
    result = apply_repair(files, {'operation': 'REWRITE', 'changes': {'d': 'short'}},
                          source='s/SKILL.md', history=set(), mode='modify', windows=windows)
    assert result['s/SKILL.md'] == 'header\nshort\nfooter'
    assert files['s/SKILL.md'] == 'header\nlong delta\nfooter'


def test_inconclusive_not_regression():
    assert paired_regression({'status': 'error'}, {'status': 'ok'}, minimum_drop=.1) == 'inconclusive'


def test_paired_identity_and_continuous_reward():
    old = dict(status='ok', probe_hash='p', grading_hash='g', passed=True, score=1.)
    new = dict(old, score=.6)
    assert paired_regression(old, new, minimum_drop=.1) == 'regression'
    with pytest.raises(ValueError, match='Unpaired'):
        paired_regression(old, dict(new, probe_hash='q'), minimum_drop=.1)


def test_dropout_restores_and_preserves_other_edits():
    def segment(text):
        offset = 0
        result = []
        for line in text.splitlines(keepends=True):
            result.append(SimpleNamespace(start=offset, end=offset+len(line), span=line))
            offset += len(line)
        return result
    old = {'s/SKILL.md': '# Title\n- first\n- second\n', 's/tool.py': 'code'}
    masked, records = mask_library(old, rate=.05, max_atoms=4,
                                   ensure_at_least_one=True, seed=101, segment=segment)
    assert len(records['s/SKILL.md']) == 1
    assert restore_library(masked, records) == old
    masked['s/SKILL.md'] += '- third\n'
    assert restore_library(masked, records)['s/SKILL.md'] == old['s/SKILL.md'] + '- third\n'
    assert restore_library(old, records) == old
    missing = dict(old, **{'s/SKILL.md': '# Title\n'})
    with pytest.raises(ValueError, match='Missing'):
        restore_library(missing, records)
