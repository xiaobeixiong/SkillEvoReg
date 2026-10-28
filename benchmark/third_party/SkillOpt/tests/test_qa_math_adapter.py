"""Offline checks for the native-agent CCV adapter boundary."""
import anyio
import pytest

from skillopt.envs.searchqa.adapter import SearchQAAdapter
from skillopt.envs.searchqa.agentic_ccv import _RestrictedAdapter as QARestricted
from skillopt.envs.livemathematicianbench.adapter import LiveMathematicianBenchAdapter
from skillopt.envs.livemathematicianbench import agentic_ccv as math_ccv


@pytest.mark.parametrize('cls,wrapper', [
    (SearchQAAdapter, QARestricted),
    (LiveMathematicianBenchAdapter, math_ccv._RestrictedAdapter),
])
def test_worker_wrapper_needs_no_spreadsheet_data_root(cls, wrapper):
    adapter = cls()
    adapter._cfg = {}
    source = {'id': 'm1'}
    restricted = wrapper(adapter, source)
    assert restricted.get_split_items('controller_selected') == [source]
    assert restricted.workers == adapter.workers


def test_math_rollout_and_scout_keep_actual_choice_order(monkeypatch, tmp_path):
    adapter = LiveMathematicianBenchAdapter()
    source = {
        'id': 'm1', 'question': 'x+1=3; x?',
        'choices': [{'label': 'A', 'text': '1'}, {'label': 'B', 'text': '2'},
                    {'label': 'C', 'text': '3'}],
        'correct_choice': {'label': 'B', 'text': '2'},
    }
    shown = adapter.dataloader._materialize_batch([source], 102)[0]
    assert shown['choices'] != adapter.dataloader._materialize_batch([source], 0)[0]['choices']
    monkeypatch.setattr('skillopt.envs.livemathematicianbench.adapter.run_batch',
                        lambda **kwargs: [{'id': 'm1', 'hard': 1}])
    rows = adapter.rollout([shown], 'old', str(tmp_path))
    monkeypatch.setattr(adapter.dataloader, 'get_split_items', lambda split: [source])
    _, items, passed = math_ccv._scout_source_pool(
        adapter, source_split='train', source_size=1,
        trajectory_context={'current_step_rollouts': rows})
    assert items['m1']['choices'] == shown['choices']
    assert items['m1']['correct_choice'] == shown['correct_choice']
    assert passed['m1']['hard'] == 1


@pytest.mark.parametrize('force,expected', [(False, ['new']), (True, ['old', 'new'])])
def test_clean_qualification_reuses_only_when_authorized(tmp_path, force, expected):
    calls = []
    class Adapter:
        def rollout(self, items, skill, out_dir):
            calls.append(skill)
            return [{'id': x['id'], 'hard': 1, 'agent_ok': True} for x in items]
    async def run():
        return await math_ccv._controller_clean_qualification(
            Adapter(), source_items={'m1': {'id': 'm1'}}, selected_ids=['m1'],
            old_passes={'m1': {'id': 'm1', 'hard': 1}}, old_skill='old',
            candidate_skill='new', qualification_passes=1, out_dir=str(tmp_path),
            force_old_clean_evaluation=force)
    old, new = anyio.run(run)
    assert set(old) == set(new) == {'m1'}
    assert calls == expected
