"""The Flow owns configs; the graph only records which one ran."""

import pytest
from helpers import StubLLM, TestRuntime, counting_replies
from helpers import reasoning_repl as block

from rlmflow import AgentConfig, Flow, Graph
from rlmflow.cli import resolve_options


def system_message(call_messages):
    return call_messages[0]["content"]


class RecordingLLM:
    def __init__(self, *replies):
        self.reply = counting_replies(*replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.reply(messages)


def test_resuming_runs_the_root_with_this_runs_config_not_the_recorded_one(tmp_path):
    llm = RecordingLLM(block("submit('one')"), block("submit('two')"))
    g = Graph()
    root = g.create_agent("answer")
    Flow(llm, runtime=TestRuntime()).run(g, root_config=AgentConfig(system_prompt="FIRST RULES"))
    assert root.metadata["config"]["system_prompt"] == "FIRST RULES"
    g.save(tmp_path)

    loaded = Graph.load(tmp_path)
    again = loaded.node(root.id)
    loaded.send(again, "and again")
    Flow(llm, runtime=TestRuntime()).run(
        loaded, root_config=AgentConfig(system_prompt="SECOND RULES")
    )

    assert "SECOND RULES" in system_message(llm.calls[-1])
    assert "FIRST RULES" not in system_message(llm.calls[-1])
    assert again.metadata["config"]["system_prompt"] == "SECOND RULES"
    assert loaded.result(again) == "two"


def test_a_registered_config_wins_over_what_the_graph_recorded():
    llm = RecordingLLM(block("submit('done')"))
    flow = Flow(
        llm, runtime=TestRuntime(), configs=[AgentConfig(name="worker", system_prompt="CURRENT")]
    )
    g = Graph()
    worker = g.spawn("work", config="worker")
    worker.annotate("config", AgentConfig(name="worker", system_prompt="STALE").to_dict())

    flow.run(g)

    assert "CURRENT" in system_message(llm.calls[0])
    assert worker.config.system_prompt == "CURRENT"


def test_an_unknown_config_name_raises():
    flow = Flow(StubLLM(lambda _: block("submit(1)")), runtime=TestRuntime())
    g = Graph()
    g.spawn("work", config="nope")

    with pytest.raises(KeyError, match="no config named 'nope'"):
        flow.run(g)
    with pytest.raises(KeyError, match="known: default"):
        flow.config("nope")


def test_launch_agent_without_a_config_uses_default_not_the_roots():
    llm = RecordingLLM(
        block("h = launch_agent('sub task')\nsubmit(await h)"), block("submit('child')")
    )
    flow = Flow(llm, runtime=TestRuntime())
    g = Graph()
    root = g.create_agent("delegate")

    flow.run(g, root_config=AgentConfig(system_prompt="ROOT ONLY", max_depth=1))

    [child] = g.children(root)
    assert child.config_name == "default"
    assert child.config == flow.configs["default"]
    assert "ROOT ONLY" in system_message(llm.calls[0])
    assert "ROOT ONLY" not in system_message(llm.calls[1])
    assert g.result(root) == "child"


def test_a_config_model_is_a_model_name(monkeypatch):
    named = RecordingLLM(block("submit('from gpt-x')"))
    built = []

    def client_for(model):
        built.append(model)
        return named

    monkeypatch.setattr("rlmflow.flow.client_for", client_for)
    flow = Flow(StubLLM(lambda _: block("submit('from default')")), runtime=TestRuntime())

    assert flow.run("which model", root_config=AgentConfig(model="gpt-x")) == "from gpt-x"
    flow.llm("gpt-x")
    assert built == ["gpt-x"]


def test_a_flow_without_a_default_model_says_so():
    with pytest.raises(ValueError, match="no default model"):
        Flow().llm("default")


def test_the_flow_factory_comes_from_rlmflow_toml(tmp_path):
    (tmp_path / "rlmflow.toml").write_text('[run]\nflow = "mypkg.agents:build"\n')

    options, origins = resolve_options({}, cwd=tmp_path, home=tmp_path / "home", environ={})

    assert options.flow == "mypkg.agents:build"
    assert origins["flow"] == "project config"
