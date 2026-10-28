from harbor.agents.installed import sequential_subtask_judge as judge


def test_runtime_import_renders_full_agent_output_without_legacy_cap():
    rendered = judge.format_sequential_subtask_judge_script(
        judge_model="openai/gpt-5.4-mini", pass_threshold=0.6)
    assert "[:2000]" not in rendered
    assert 'encoding="utf-8", errors="replace"' in rendered
