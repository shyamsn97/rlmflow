from __future__ import annotations

from rlmflow import DoneOutput, Graph, LLMOutput
from rlmflow.cli import main


def test_cli_view_prints_step_frames(tmp_path, capsys):
    graph = Graph(query="root query")
    graph.append(LLMOutput(content="plan", code="pass"))
    graph.append(DoneOutput(result="ok"))
    path = graph.save(tmp_path / "run")

    assert main(["view", str(path)]) == 0
    out = capsys.readouterr().out
    assert "=== step 1/" in out
    assert "done ok" in out


def test_cli_view_step_selects_one_frame(tmp_path, capsys):
    graph = Graph(query="root query")
    graph.append(DoneOutput(result="ok"))
    path = graph.save(tmp_path / "run")

    assert main(["view", str(path), "--step", "1"]) == 0
    out = capsys.readouterr().out
    assert "=== step" not in out
    assert "root" in out


def test_cli_view_missing_path_errors(capsys):
    assert main(["view", "/no/such/graph"]) == 1
    err = capsys.readouterr().err
    assert "not found" in err
