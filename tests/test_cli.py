"""The command line: option precedence, exit codes, and a headless run end to end."""

from __future__ import annotations

import pytest

from rlmflow import Flow, LocalRuntime
from rlmflow.cli import main
from rlmflow.cli.options import CliError, RunOptions, resolve
from rlmflow.cli.run import build_flow, load_factory, load_root, require_api_key, run_agent


class ScriptedLLM:
    """Answers once, with the working directory it can see."""

    def __init__(self, reply="```python\nfinish('done')\n```"):
        self.reply = reply
        self.calls = 0

    def chat(self, _messages):
        self.calls += 1
        return self.reply


def flow_factory():
    return Flow(ScriptedLLM(), runtime=LocalRuntime())


NOT_A_FLOW = "not a flow at all"


def not_a_flow_factory():
    return NOT_A_FLOW


@pytest.fixture
def ran_graph(tmp_path):
    """A one-answer run, saved, for the commands that read a checkpoint."""
    flow = Flow(ScriptedLLM(), runtime=LocalRuntime(working_directory=tmp_path))
    root = flow.start("say done")
    flow.run(root, close_repls=True)
    return root.save(tmp_path / "graph")


def test_resolve_prefers_a_flag_over_env_over_project_over_user(tmp_path):
    home = tmp_path / "home"
    (home / ".config" / "rlmflow").mkdir(parents=True)
    (home / ".config" / "rlmflow" / "config.toml").write_text(
        '[run]\nmodel = "user"\nfast_model = "user-fast"\nmax_iters = 11\nworkdir = "user-dir"\n'
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / "rlmflow.toml").write_text('[run]\nmodel = "project"\nfast_model = "project-fast"\n')
    environ = {"RLMFLOW_MODEL": "env"}

    options, origins = resolve({"model": "flag"}, cwd=project, home=home, environ=environ)

    assert (options.model, origins["model"]) == ("flag", "flag")
    assert (options.fast_model, origins["fast_model"]) == ("project-fast", "project config")
    assert (options.workdir, origins["workdir"]) == ("user-dir", "user config")
    assert (options.max_depth, origins["max_depth"]) == (3, "default")


def test_resolve_reads_env_when_no_flag_and_coerces_numbers(tmp_path):
    environ = {"RLMFLOW_MODEL": "env-model", "RLMFLOW_MAX_ITERS": "7"}

    options, origins = resolve({"model": None}, cwd=tmp_path, home=tmp_path, environ=environ)

    assert options.model == "env-model"
    assert options.max_iters == 7 and origins["max_iters"] == "env"


def test_resolve_rejects_an_unknown_setting_and_a_bad_number(tmp_path):
    (tmp_path / "rlmflow.toml").write_text('[run]\nmodle = "typo"\n')

    with pytest.raises(CliError, match="unknown project config setting 'modle'"):
        resolve({}, cwd=tmp_path, home=tmp_path, environ={})

    with pytest.raises(CliError, match="max_iters must be a whole number"):
        resolve({}, cwd=tmp_path / "none", home=tmp_path, environ={"RLMFLOW_MAX_ITERS": "many"})


def test_run_options_rejects_unknown_tools():
    with pytest.raises(CliError, match="tools must be"):
        RunOptions(tools="everything")


def test_build_flow_wires_the_options_into_the_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    options = RunOptions(workdir=str(tmp_path), max_depth=1, max_iters=5, workers=2)

    flow = build_flow(options, workdir=tmp_path)

    assert (flow.root_config.max_depth, flow.root_config.max_iters) == (1, 5)
    assert flow.runtime.working_directory == tmp_path
    assert "fast" in flow._llm_clients
    assert "read_file" in flow.tools


def test_build_flow_without_file_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    flow = build_flow(RunOptions(tools="none"), workdir=tmp_path)

    assert flow.tools == {}


def test_require_api_key_names_the_one_that_is_missing():
    with pytest.raises(CliError, match="ANTHROPIC_API_KEY"):
        require_api_key("claude-sonnet-4-20250514", environ={})
    with pytest.raises(CliError, match="OPENAI_API_KEY"):
        require_api_key("gpt-5", environ={})
    require_api_key("gpt-5", environ={"OPENAI_API_KEY": "x"})


def test_load_factory_imports_a_flow_and_reports_every_way_it_can_fail():
    assert isinstance(load_factory("tests.test_cli:flow_factory"), Flow)

    with pytest.raises(CliError, match="module:factory"):
        load_factory("tests.test_cli")
    with pytest.raises(CliError, match="cannot import"):
        load_factory("no.such.module:factory")
    with pytest.raises(CliError, match="has no 'missing'"):
        load_factory("tests.test_cli:missing")
    with pytest.raises(CliError, match="not callable"):
        load_factory("tests.test_cli:NOT_A_FLOW")
    with pytest.raises(CliError, match="not a Flow"):
        load_factory("tests.test_cli:not_a_flow_factory")


def test_load_root_rejects_a_path_that_is_not_a_run(tmp_path):
    with pytest.raises(CliError, match="path not found"):
        load_root(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CliError, match="cannot read a run"):
        load_root(empty)


def test_run_headless_streams_checkpoints_and_prints_the_answer(tmp_path, capsys):
    flow = Flow(ScriptedLLM(), runtime=LocalRuntime(working_directory=tmp_path))

    run_agent("say done", RunOptions(workdir=str(tmp_path)), headless=True, flow=flow)

    out = capsys.readouterr().out
    assert "done" in out
    assert (tmp_path / "graph").is_dir()
    assert flow.runtime.repls == {}, "the run should not leave a REPL behind"


def test_run_opens_the_dashboard_on_the_flow_and_cleans_up(tmp_path, monkeypatch):
    """The default surface: the TUI gets the flow, a checkpointer, and the query."""
    import rlmflow.consumers.tui as tui

    opened = {}

    class FakeTUI:
        def __init__(self, root=None, *, sink=None):
            opened["root"], opened["sink"] = root, sink

        def init(self, flow):
            opened["flow"] = flow

        def run(self, drive=None, *, query=None):
            opened["query"] = query

    monkeypatch.setattr(tui, "FlowTUI", FakeTUI)
    flow = Flow(ScriptedLLM(), runtime=LocalRuntime(working_directory=tmp_path))

    run_agent("start here", RunOptions(workdir=str(tmp_path)), flow=flow)

    assert opened["flow"] is flow
    assert opened["query"] == "start here"
    assert opened["sink"].path == tmp_path / "graph"
    assert flow.runtime.repls == {}


def test_run_headless_needs_something_to_run(tmp_path):
    flow = Flow(ScriptedLLM(), runtime=LocalRuntime(working_directory=tmp_path))

    with pytest.raises(CliError, match="nothing to run"):
        run_agent(None, RunOptions(workdir=str(tmp_path)), headless=True, flow=flow)


def test_main_returns_one_and_prints_the_error(capsys, tmp_path):
    assert main(["view", "show", str(tmp_path / "missing")]) == 1

    err = capsys.readouterr().err
    assert "error: path not found" in err


def test_main_help_lists_every_command_and_group(capsys):
    assert main(["--help"]) == 0

    captured = capsys.readouterr()
    help_text = captured.out + captured.err
    for command in ("run", "tui", "view", "version", "config", "render"):
        assert command in help_text


def test_run_help_lists_the_tui_and_print_verbs(capsys):
    assert main(["run", "--help"]) == 0

    captured = capsys.readouterr()
    help_text = captured.out + captured.err
    assert "tui" in help_text
    assert "print" in help_text


def test_run_print_streams_through_the_class(tmp_path, capsys):
    assert (
        main(
            [
                "run",
                "print",
                "say done",
                "--agent",
                "tests.test_cli:flow_factory",
                "--workdir",
                str(tmp_path),
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "done" in out
    assert (tmp_path / "graph").is_dir()


def test_render_group_writes_a_figure(ran_graph, tmp_path, capsys):
    out = tmp_path / "run.svg"

    assert main(["render", "svg", str(ran_graph), str(out)]) == 0

    assert out.is_file()
    assert "wrote" in capsys.readouterr().out


def test_render_reports_a_bad_path_once(ran_graph, tmp_path, capsys):
    assert main(["render", "svg", str(tmp_path / "missing"), str(tmp_path / "x.svg")]) == 1

    assert "error: path not found" in capsys.readouterr().err


def test_version_and_config_report_the_basics(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("RLMFLOW_MODEL", "env-model")
    monkeypatch.chdir(tmp_path)

    assert main(["version"]) == 0
    assert main(["config", "show"]) == 0
    assert main(["config", "path"]) == 0

    out = capsys.readouterr().out
    assert "rlmflow " in out and "python " in out
    assert "env-model" in out and "RLMFLOW_MODEL" in out
    assert "project" in out and "user" in out


def test_config_init_writes_a_starter_file_and_refuses_to_clobber(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RLMFLOW_MODEL", "env-model")

    assert main(["config", "init"]) == 0

    written = (tmp_path / "rlmflow.toml").read_text()
    assert 'model = "env-model"' in written
    assert main(["config", "init"]) == 1
