"""The Flow owns agents; the graph only names the one each run runs with."""

import pytest
from helpers import ScriptedFlow, StubLLM, TestRuntime, counting_replies, name_root
from helpers import reasoning_repl as block

from rlmflow import Agent, Flow, Graph
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


def test_resuming_runs_the_root_with_this_flows_agent(tmp_path):
    llm = RecordingLLM(block("submit('one')"), block("submit('two')"))
    g = Graph()
    root = g.start("answer")
    first = ScriptedFlow(llm, runtime=TestRuntime())
    name_root(first, g, Agent(prompt={"rules": "FIRST RULES"}))
    first.run(g)
    assert "FIRST RULES" in system_message(llm.calls[0])

    loaded = Graph.load(g.save(tmp_path / "run"))
    again = loaded.agent_root(root.id)
    assert again.agent == "root"
    loaded.send(again, "and again")
    second = ScriptedFlow(llm, runtime=TestRuntime())
    second.agents["root"] = Agent(prompt={"rules": "SECOND RULES"})
    second.run(loaded)

    assert "SECOND RULES" in system_message(llm.calls[-1])
    assert "FIRST RULES" not in system_message(llm.calls[-1])
    assert again.final_result() == "two"


def test_an_unknown_agent_name_raises():
    flow = ScriptedFlow(StubLLM(lambda _: block("submit(1)")), runtime=TestRuntime())
    g = Graph()
    g.start("work", agent="nope")

    with pytest.raises(KeyError, match="no agent named 'nope'.*known: default"):
        flow.run(g)
    with pytest.raises(KeyError, match="no agent named 'nope'"):
        flow.start("nope", "work")


def test_launch_agent_without_an_agent_uses_the_launchers_own():
    llm = RecordingLLM(
        block("h = launch_agent('sub task')\nsubmit(await h)"), block("submit('child')")
    )
    flow = ScriptedFlow(llm, runtime=TestRuntime())
    g = Graph()
    root = g.start("delegate")

    name_root(flow, g, Agent(prompt={"rules": "ROOT ONLY"}, max_depth=1))
    flow.run(g)

    [child] = root.children()
    assert child.agent == "root"
    assert flow.agent(child) is flow.agents["root"]
    assert "ROOT ONLY" in system_message(llm.calls[1])
    assert root.final_result() == "child"


def test_launch_agent_names_another_agent():
    llm = RecordingLLM(
        block("h = launch_agent('sub task', agent='worker')\nsubmit(await h)"),
        block("submit('child')"),
    )
    flow = ScriptedFlow(
        llm,
        runtime=TestRuntime(),
        agents={"default": Agent(), "worker": Agent(prompt={"rules": "WORKER"}, max_depth=0)},
    )
    g = flow.start("default", "delegate")
    flow.run(g)

    [child] = g.root.children()
    assert child.agent == "worker"
    assert "WORKER" in system_message(llm.calls[1])
    assert g.root.final_result() == "child"


def test_an_agents_model_is_a_model_name(monkeypatch):
    named = RecordingLLM(block("submit('from terra')"))
    built = []

    def client_for(model):
        built.append(model)
        return named

    monkeypatch.setattr("rlmflow.flow.flow.client_for", client_for)
    flow = Flow(runtime=TestRuntime(), agents={"x": Agent(model="gpt-6.1-terra")})

    assert flow.run("x", "which model") == "from terra"
    flow.llm("gpt-6.1-terra")
    assert built == ["gpt-6.1-terra"]


def test_an_agent_without_a_model_says_so():
    flow = Flow(runtime=TestRuntime(), agents={"default": Agent(model=None)})
    with pytest.raises(ValueError, match=r"set Agent\(model=\.\.\.\)"):
        flow.run("default", "which model")


def test_a_custom_client_is_named_like_a_model():
    custom = StubLLM(lambda _: block("submit('custom')"))
    flow = Flow(
        llm_clients={"tuned": custom},
        agents={"default": Agent(model="tuned")},
        runtime=TestRuntime(),
    )
    assert flow.run("default", "which model") == "custom"


def test_the_flow_factory_comes_from_rlmflow_toml(tmp_path):
    (tmp_path / "rlmflow.toml").write_text('[run]\nflow = "mypkg.agents:build"\n')

    options, origins = resolve_options({}, cwd=tmp_path, home=tmp_path / "home", environ={})

    assert options.flow == "mypkg.agents:build"
    assert origins["flow"] == "project config"
