"""Frozen evaluation source and task isolation checks."""
from scripts.evaluate_frozen import _source_library_path, _target
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import pytest
from scripts import evaluate_frozen as runner


def test_explicit_source_is_used_for_every_environment(tmp_path):
    for index in range(1, 7):
        assert _source_library_path(tmp_path, f'E{index}') == tmp_path / 'library' / f'E{index}'


def test_existing_sources_and_outputs_remain_separate(tmp_path):
    for method in ('baseline', 'noreg', 'regularization'):
        root = tmp_path / method
        assert _source_library_path(root, 'E6') == root / 'library' / 'E6'
    paths = {_target(method, block, tmp_path / method).run_root
             for method in ('baseline', 'noreg', 'regularization') for block in ('t1-t3', 't4-t6')}
    assert len(paths) == 6


def test_single_task_repair_is_isolated_and_rejects_wrong_block(monkeypatch):
    task = SimpleNamespace(spec=SimpleNamespace(task_id='E6-LS3-T2'))
    monkeypatch.setattr(runner, '_schedule', lambda config, block: ([task], None))
    normal = _target('regularization', 't1-t3', Path('explicit_source'))
    repair = replace(normal, only_task='E6-LS3-T2')
    assert runner._target_schedule(None, repair)[0] == [task]
    assert repair.run_root != normal.run_root
    with pytest.raises(ValueError):
        runner._target_schedule(None, replace(normal, only_task='E6-LS3-T4'))


def test_single_task_summary_does_not_divide_by_empty_roles(tmp_path):
    target = replace(_target('regularization', 't1-t3', tmp_path), only_task='E6-LS3-T2')
    record = SimpleNamespace(task_role='enriched', task_id='E6-LS3-T2__replay',
        family_id='E6-LS3', env_id='E6', outcome=SimpleNamespace(
            verifier_passed=False, normalized_score=0.5,
            outcome_passed=False, process_passed=False))
    runtime = SimpleNamespace(run_root=tmp_path,
        replay_store=SimpleNamespace(all_records=lambda: [record]))
    import json
    result = json.loads(runner._write_frozen_summary(target, runtime).read_text())
    assert result['overall']['n_tasks'] == 1
    assert set(result['by_role']) == {'enriched'}
