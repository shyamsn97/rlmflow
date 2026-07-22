import pytest

from rlmflow.tools.filesystem import edit_file


def _write(tmp_path, text):
    p = tmp_path / "sample.py"
    p.write_text(text)
    return p


def test_edit_file_unique_anchor_applies(tmp_path):
    p = _write(tmp_path, "a = 1\nb = 2\n")
    edit_file(str(p), ("b = 2", "b = 3"))
    assert p.read_text() == "a = 1\nb = 3\n"


def test_edit_file_missing_anchor_raises_and_does_not_write(tmp_path):
    p = _write(tmp_path, "a = 1\n")
    with pytest.raises(ValueError, match="anchor not found"):
        edit_file(str(p), ("nope", "x"))
    assert p.read_text() == "a = 1\n"


def test_edit_file_ambiguous_anchor_raises(tmp_path):
    # "import torch" is a prefix of the .nn/.functional lines: the exact bug.
    text = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    p = _write(tmp_path, text)
    with pytest.raises(ValueError, match="appears 3x"):
        edit_file(str(p), ("import torch", "import torch\nimport math"))
    assert p.read_text() == text


def test_edit_file_replace_all_opt_in(tmp_path):
    p = _write(tmp_path, "x x x\n")
    edit_file(str(p), ("x", "y"), replace_all=True)
    assert p.read_text() == "y y y\n"


def test_edit_file_is_atomic_on_partial_failure(tmp_path):
    p = _write(tmp_path, "a = 1\nb = 2\n")
    with pytest.raises(ValueError, match="anchor not found"):
        edit_file(str(p), ("a = 1", "a = 9"), ("missing", "x"))
    # First edit must not persist because a later edit failed.
    assert p.read_text() == "a = 1\nb = 2\n"
