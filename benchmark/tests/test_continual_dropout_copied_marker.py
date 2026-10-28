import pytest
from continual_regularization.dropout import restore_library

MARKER = '<!-- CONTINUAL_DROPOUT_abc_0 -->\n'
RECORDS = {'a/SKILL.md': {MARKER: 'original rule\n'}}


def test_copied_marker_removed_after_owner_restored():
    files = {'a/SKILL.md': MARKER, 'b/SKILL.md': 'before\n' + MARKER + 'after\n'}
    assert restore_library(files, RECORDS) == {
        'a/SKILL.md': 'original rule\n', 'b/SKILL.md': 'before\nafter\n'}
    assert files['a/SKILL.md'] == MARKER


def test_already_restored_owner_and_copy_at_eof():
    assert restore_library({'a/SKILL.md': 'original rule\n',
                            'b/SKILL.md': MARKER.rstrip('\n')}, RECORDS)['b/SKILL.md'] == ''


@pytest.mark.parametrize('owner', ['', MARKER + MARKER])
def test_missing_or_duplicate_owner_not_ignored(owner):
    with pytest.raises(ValueError):
        restore_library({'a/SKILL.md': owner, 'b/SKILL.md': MARKER}, RECORDS)


def test_deleted_owner_does_not_authorize_removal():
    with pytest.raises(ValueError, match='Residual'):
        restore_library({'b/SKILL.md': MARKER}, RECORDS, allow_deleted=True)


@pytest.mark.parametrize('residue', [MARKER.replace('abc', 'unknown'), 'inline ' + MARKER])
def test_unknown_or_inline_marker_not_ignored(residue):
    with pytest.raises(ValueError, match='Residual'):
        restore_library({'a/SKILL.md': MARKER, 'b/SKILL.md': residue}, RECORDS)
