import asyncio
import threading
from pathlib import Path
import yaml
from continual_regularization.ccv import MathCCV


def test_pair_runs_concurrently_and_retains_old_new_identity():
    barrier = threading.Barrier(2, timeout=2)
    def slot(files, instruction, source, name):
        barrier.wait()  # Sequential execution would time out.
        return (files, instruction, source, name)
    old, new = MathCCV.paired_slots(slot, {'old': 1}, {'new': 2}, 'same', {'oracle': 3}, 'clean')
    assert old == ({'old': 1}, 'same', {'oracle': 3}, 'clean_old')
    assert new == ({'new': 2}, 'same', {'oracle': 3}, 'clean_new')


def test_unlimited_config_and_asyncio_timeout_compatibility():
    from harbor.models.job.config import JobConfig
    root = Path(__file__).resolve().parents[2]
    config = JobConfig.model_validate(yaml.safe_load((root /
        'benchmark/configs/continual_math_regularization_gpt54mini_medium.yaml').read_text()))
    assert config.agents[0].override_timeout_sec == float('inf')
    async def run():
        return await asyncio.wait_for(asyncio.sleep(0, result=42), timeout=float('inf'))
    assert asyncio.run(run()) == 42
