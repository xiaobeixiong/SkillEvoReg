import importlib.util
from pathlib import Path
import sys

import pytest

path = Path(__file__).parents[1] / "integrations/continual_regularization/policy.py"
spec = importlib.util.spec_from_file_location("continual_regularization_policy", path)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


def test_directional_coverage_not_library_union():
    matrices = {"a": [[.9], [.2]], "b": [[.2], [.9]]}
    assert policy.rank_matches(matrices, atom_count=2, tau=.8, rho=.75) == []
    matches = policy.rank_matches(matrices, atom_count=2, tau=.8, rho=.5)
    assert [m.coverage for m in matches] == [.5, .5]


def test_threshold_inclusive_top3_deterministic():
    matches = policy.rank_matches({k: [[.8]] for k in "dcba"},
                                  atom_count=1, tau=.8, rho=1)
    assert [m.path for m in matches] == list("abc")


@pytest.mark.parametrize("matrix", [[[float("nan")]], [[1.2]], [[.8], []]])
def test_invalid_matrix(matrix):
    with pytest.raises(ValueError):
        policy.rank_matches({"a": matrix}, atom_count=len(matrix), tau=.8, rho=.5)


def test_empty_atoms_never_match():
    assert policy.rank_matches({"a": []}, atom_count=0, tau=.8, rho=0) == []


def test_retrieval_audit_keeps_rejected_and_topk_excluded():
    matrices = {'a': [[.7], [.6]], 'b': [[.71], [.6]], 'c': [[.69], [.6]]}
    audit = []
    selected = policy.rank_matches(matrices, atom_count=2, tau=.7, rho=.5,
                                   top_k=1, audit=audit)
    assert [m.path for m in selected] == ['a']
    assert [r['coverage'] for r in audit] == [.5, .5, 0]
    assert [r['selected'] for r in audit] == [True, False, False]
    assert audit[2]['atom_maxima'] == [.69, .6]


def test_actual_change_classification():
    assert policy.classify_changes({"a": "1", "b": "2", "c": "3"},
                                   {"a": "1", "b": "4", "d": "5"}) == {
        "create": ["d"], "modify": ["b"], "delete": ["c"]}


def test_dropout_does_not_request_compression():
    text = policy.render_instruction("modify", ("dropout",))
    assert "Complexity feedback:" not in text
    assert "not a compression request" in text


def test_modify_merge_receiving_budget_guidance():
    text = policy.render_instruction('modify', ('dropout',))
    assert 'For MERGE, use receiving_skill_budgets' in text
    assert 'soft targets, not rejection gates' in text
    assert 'Complexity feedback:' not in text


def test_unmatched_create_can_compress_in_same_update():
    text = policy.render_instruction("create", ("complexity",))
    assert "With no historical match" in text
    assert "In THIS update" in text
    assert "do not run a second compression phase" in text


def test_meta_skills_excluded_and_unchanged(tmp_path):
    for slug in ("create-skill", "modify-skill", "learned"):
        directory = tmp_path / slug
        directory.mkdir()
        (directory / "SKILL.md").write_text(slug)
    assert policy.snapshot(tmp_path) == {"learned/SKILL.md": "learned"}


def test_symlink_escape_rejected(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("private")
    (library / "SKILL.md").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        policy.snapshot(library)
