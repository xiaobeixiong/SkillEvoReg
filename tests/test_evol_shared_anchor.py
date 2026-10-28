"""Three-way comparison must share one immutable first learning pass."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run import EVOL, plan, shared_anchor_source


@pytest.fixture
def anchor(tmp_path):
    source = tmp_path / 'first_pass'
    (source / 'reports').mkdir(parents=True)
    (source / 'config.json').write_text(json.dumps(dict(learning_passes=1,
        shared_learning_anchor_source=None, anchor_source_only=True)))
    (source / 'reports/full_report.json').write_text('{}')
    export = tmp_path / 'shared_anchor'
    export.mkdir()
    for index in range(1, 7):
        library = source / 'library' / f'E{index}'
        (library / '.git').mkdir(parents=True)
        (library / 'active').mkdir()
        (library / 'manifest.yaml').write_text('skills: {}\n')
        (library / 'active/SKILL.md').write_text(f'First-pass skill E{index}')
        shutil.copytree(library, export / 'library' / f'E{index}')
    (export / 'anchor_manifest.json').write_text(json.dumps(dict(
        kind='complete_t1_t3_learning_anchor', learning_records=90, source_run=str(source))))
    return export, source


def args(method, anchor=None, **changes):
    values = dict(framework='evol', method=method, anchor=str(anchor) if anchor else None,
                  run_id=None if method == 'baseline' else method, repeat=1)
    values.update(changes)
    return argparse.Namespace(**values)


def test_one_pass_is_only_frozen_evaluation_of_the_shared_source(anchor):
    export, source = anchor
    command, cwd, _ = plan(args('baseline', export))
    assert command[command.index('--source') + 1] == str(source)
    assert command[command.index('--block') + 1] == 'all'
    assert command[command.index('--method') + 1] == 'baseline'
    assert '--learning-passes' not in command and '-m' not in command
    assert cwd == ROOT


def test_second_passes_load_the_same_export(anchor):
    export, source = anchor
    assert shared_anchor_source(export) == source
    for method in ('twopass', 'skillevoreg'):
        command, cwd, _ = plan(args(method, export))
        assert command[command.index('--shared-learning-anchor-source') + 1] == str(export)
        assert command[command.index('--learning-passes') + 1] == '2'
        assert cwd == EVOL


def test_baseline_cannot_silently_start_another_learning_run():
    with pytest.raises(ValueError, match='complete shared'):
        plan(args('baseline'))


def test_baseline_repeats_have_distinct_observation_indices(anchor):
    export, _ = anchor
    a, _, _ = plan(args('baseline', export, repeat=1))
    b, _, _ = plan(args('baseline', export, repeat=2))
    assert a[a.index('--repeat') + 1] == '1'
    assert b[b.index('--repeat') + 1] == '2'


@pytest.mark.parametrize('method', ['baseline', 'twopass', 'skillevoreg'])
def test_diverged_source_cannot_be_mixed_with_an_export(anchor, method):
    export, source = anchor
    (source / 'library/E3/active/SKILL.md').write_text('Changed after export')
    with pytest.raises(ValueError, match='differs'):
        plan(args(method, export))


def test_git_metadata_and_freeze_markers_are_not_skill_changes(anchor):
    export, source = anchor
    (source / 'library/E1/.git/index').write_text('index metadata')
    (source / 'library/E1/.frozen').write_text('frozen')
    assert shared_anchor_source(export) == source


def test_an_already_second_pass_source_is_rejected(anchor):
    export, source = anchor
    (source / 'config.json').write_text(json.dumps(dict(learning_passes=2)))
    with pytest.raises(ValueError, match='one learning pass'):
        shared_anchor_source(export)


def test_baseline_dispatches_both_frozen_blocks_without_training(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('public_eval_wrapper', ROOT / 'scripts/evaluate_evol.py')
    wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
    fake = ModuleType('fake_frozen_runner')
    seen = []
    fake.EvalTarget = lambda **kwargs: SimpleNamespace(**kwargs)
    async def run(target):
        seen.append((target.block, target.source_root, target.repeat_index))
    fake._run_target = run
    loader = SimpleNamespace(exec_module=lambda module: None)
    monkeypatch.setattr(importlib.util, 'spec_from_file_location',
        lambda *args: SimpleNamespace(name='fake_frozen_runner', loader=loader))
    monkeypatch.setattr(importlib.util, 'module_from_spec', lambda spec: fake)
    monkeypatch.setenv('_RELEASE_EVOL_EVAL_CHILD', '1')
    monkeypatch.setattr(wrapper, 'environment', lambda framework:
        dict(OPENAI_API_KEY='dummy-test-key', OPENAI_BASE_URL='https://review.invalid/v1'))
    monkeypatch.setattr(sys, 'argv', ['evaluate_evol.py', '--source', str(tmp_path),
        '--method', 'baseline', '--block', 'all', '--repeat', '1', '--execute'])
    wrapper.main()
    assert seen == [('t4-t6', tmp_path, 1), ('t1-t3', tmp_path, 1)]
