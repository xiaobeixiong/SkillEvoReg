import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1]/'scripts/analyze_continual_frozen_preparation.py'
spec = importlib.util.spec_from_file_location('frozen_preparation', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reconstruction_patch_does_not_execute_or_touch_disk():
    files = {}
    module.patch_files(files, '*** Begin Patch\n*** Add File: /root/.agents/skills/a/SKILL.md\n+x\n+y\n*** End Patch')
    module.patch_files(files, '*** Begin Patch\n*** Update File: /root/.agents/skills/a/SKILL.md\n@@\n x\n-y\n+z\n*** End Patch')
    assert files == {'a/SKILL.md': 'x\nz\n'}


def test_reconstruction_rejects_ambiguous_context():
    with pytest.raises(ValueError):
        module.patch_files({'a/SKILL.md':'x\nx\n'},
            '*** Begin Patch\n*** Update File: /root/.agents/skills/a/SKILL.md\n@@\n-x\n+y\n*** End Patch')
