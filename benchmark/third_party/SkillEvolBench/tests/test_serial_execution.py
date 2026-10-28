"""The release uses the native serial schedule without environment sharding."""
import inspect
import json

import pytest

from skillevolbench.orchestration.lifelong_runner import LifelongRunner
from skillevolbench.schemas import RunConfig


def test_run_config_has_no_environment_shard_selector():
    assert 'environment_ids' not in RunConfig.model_fields


def test_runner_keeps_the_unfiltered_native_schedule():
    assert not hasattr(LifelongRunner, '_select_environments')
    source = inspect.getsource(LifelongRunner)
    assert 'resume_regularization_schedule_prefix' not in source
    assert '_filter_schedule' not in source
    assert 'return runtime, ordered_tasks, registry' in source


def test_export_rejects_a_second_pass_source(tmp_path):
    from scripts.export_learning_anchor import export
    source = tmp_path / 'second_pass'; source.mkdir()
    (source / 'config.json').write_text(json.dumps(dict(learning_passes=2)))
    with pytest.raises(RuntimeError, match='first-pass run'):
        export(source, tmp_path / 'anchor')
    assert not (tmp_path / 'anchor').exists()
