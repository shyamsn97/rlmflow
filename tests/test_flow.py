import asyncio
import base64
import json
import threading
import time
from types import SimpleNamespace

import pytest
from helpers import (
    ScriptedFlow,
    StubLLM,
    TestRuntime,
    agent_graph,
    code_nodes,
    counting_replies,
    first_user,
    own,
    ran,
    run_result,
    stepping,
    trailing_user_text,
)
from helpers import reasoning_repl as block

from rlmflow import (
    RETURN,
    SPAWN,
    Agent,
    AgentConfig,
    Input,
    LLMChunk,
    LLMUsage,
    Node,
    attach_image,
)
from rlmflow.agent_tools import tools as agent_tools
from rlmflow.agent_tools.view import AgentHandle
from rlmflow.graph import Graph, Step, calls
from rlmflow.graph.nodes import READY
from rlmflow.graph.persistence import FORMAT_VERSION, node_record
from rlmflow.harnesses import repl
from rlmflow.llm.anthropic import anthropic_content
from rlmflow.runtime import LocalRuntime
from rlmflow.tools import check_args, repl_tools, tool, tool_from_schema
from rlmflow.utils import truncate_output


class ScriptedLLM:
    """Replies by matching the trailing user messages against scripted needles."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        last = trailing_user_text(messages)
        for needle, reply in self.script:
            if needle in last:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        raise AssertionError(f"no scripted reply for {last!r}")


def model_turns(nodes):
    return [node for node in nodes if "model" in node.data]


def messages_in(nodes):
    return [node for node in nodes if node.role == "user" and not isinstance(node, Agent)]


def ends(node) -> bool:
    return bool(node.data.get("terminal"))


def test_truncate_output_preserves_traceback_tail():
    output = "printed preview data\n" + "x" * 100 + "\nValueError: boxes has no items"

    truncated = truncate_output(output, 40)

    assert truncated.startswith("printed preview data")
    assert "truncated" in truncated
    assert truncated.endswith("boxes has no items")


def test_reassigned_arguments_stick_across_turns():
    llm = StubLLM(counting_replies(block("x = 999\ntask = 'changed'"), block("submit([x, task])")))
    flow = ScriptedFlow(llm)
    assert flow.run("keep it", x=1) == [999, "changed"]


def test_submit():
    llm = ScriptedLLM([("count", block("x = 2\nprint(x)")), ("2", block("submit(x * 3)"))])
    flow = ScriptedFlow(llm)
    assert flow.run("count for me") == 6
    assert llm.calls[0][0]["role"] == "system"


def reload(g, path):
    """Save ``g`` to ``path`` and load it back as a cold graph."""
    g.save(path)
    return Graph.load(path)


def repl_prompt(flow, task, **config):
    g, agent = agent_graph(task)
    agent.bind_config(AgentConfig(**config))
    return repl.ReplHarness().prompt(flow, Step(g, agent, agent))


def test_system_prompt_states_repl_action_and_both_ways_to_finish():
    prompt = repl_prompt(ScriptedFlow(ScriptedLLM([])), "write a poem")

    assert "To act, a response contains exactly one non-empty fenced ```repl``` block" in prompt
    assert "A response with no ```repl``` block is your final answer" in prompt
    assert "call `submit(value)` inside a block" in prompt
    assert "reply in plain text with no ```repl``` block" in prompt
    assert "Do not reply in plain text while work remains" in prompt
    assert "Avoid calling `submit(...)` in a block that prints output" in prompt
    assert "Printing and submitting together is allowed" in prompt
    assert "read the observation, then submit in the next turn" in prompt


def test_config_prompt_overrides_sections_by_key():
    prompt = repl_prompt(
        ScriptedFlow(ScriptedLLM([])),
        "x",
        prompt={
            "protocol": "Be terse.",
            "agents": None,
            "orchestration": None,
            "style": "No emoji.",
        },
    )
    assert prompt.startswith("Be terse.")
    assert "submit(" in prompt
    assert "AGENTS" not in prompt
    assert prompt.endswith("No emoji.")


def test_only_agents_that_can_launch_get_the_orchestration_skill():
    flow = ScriptedFlow(ScriptedLLM([]))
    assert "# Orchestrating agents in rlmflow" in repl_prompt(flow, "x")
    assert "name: orchestrate" not in repl_prompt(flow, "x")
    assert "# Orchestrating agents" not in repl_prompt(flow, "x", max_depth=0)


def test_a_harness_subclass_replaces_the_whole_prompt():
    class Terse(repl.ReplHarness):
        def prompt_sections(self, flow, step):
            return {"mine": "ONLY THIS"}

    g, agent = agent_graph("x")
    agent.bind_config(AgentConfig(prompt={"extra": "AND THIS"}))
    assert (
        Terse().prompt(ScriptedFlow(ScriptedLLM([])), Step(g, agent, agent))
        == "ONLY THIS\n\nAND THIS"
    )


def test_prose_before_repl_action_remains_in_following_turn_context():
    llm = ScriptedLLM(
        [
            ("inspect", "I need to inspect first.\n```repl\nprint('observed')\n```"),
            ("observed", block("submit('The final answer.')")),
        ]
    )

    assert ScriptedFlow(llm).run("inspect this") == "The final answer."
    assert any(
        message
        == {
            "role": "assistant",
            "content": "I need to inspect first.\n```repl\nprint('observed')\n```",
        }
        for message in llm.calls[1]
    )


def test_error_then_recovery():
    llm = ScriptedLLM(
        [("boom", block("1 / 0")), ("ZeroDivisionError", block("submit('recovered')"))]
    )
    g, root = agent_graph("boom")
    assert run_result(ScriptedFlow(llm), g, root) == "recovered"
    assert any(node.runs_code and node.error for node in g.nodes())


def test_max_iters():
    llm = ScriptedLLM(
        [
            ("last turn", block("submit('best effort')")),
            ("", block("print('again')")),
        ]
    )
    g, root = agent_graph("loop")
    assert (
        run_result(ScriptedFlow(llm), g, root, root_config=AgentConfig(max_iters=3))
        == "best effort"
    )
    assert len(model_turns(g.nodes())) == 3


def test_iters_and_budget_are_unbounded_unless_set():
    llm = ScriptedLLM([("loop", block("print('again')")), ("again", block("submit('ok')"))])
    g, root = agent_graph("loop")
    assert root.config.max_iters is None
    assert root.config.max_budget is None
    assert root.config.max_output_length == 20_000
    assert root.config.max_query_chars == 20_000
    bounded = AgentConfig(max_iters=30, max_budget=100_000)
    assert bounded.max_iters == 30
    assert bounded.max_budget == 100_000
    assert run_result(ScriptedFlow(llm), g, root) == "ok"
    assert root.config.max_iters is None


def test_direct_arguments():
    llm = ScriptedLLM([("", block("submit(doc.upper())"))])
    assert ScriptedFlow(llm).run("shout", doc="hi") == "HI"


def test_described_direct_argument_is_unwrapped_in_the_repl():
    llm = ScriptedLLM([("", block("submit(doc.upper())"))])

    result = ScriptedFlow(llm).run(
        "shout",
        doc=Input("hi", description="A short lowercase document."),
    )

    assert result == "HI"


def test_described_large_argument_survives_save_and_load(tmp_path):
    document = "x" * 70_000
    g, root = agent_graph(
        "read",
        args={"document": Input(document, description="A large source document.")},
    )

    run = g.save(tmp_path / "described-input")
    loaded = Graph.load(run).agent(root.id)

    assert loaded.input == {"task": "read", "document": document}
    assert loaded.arg_descriptions == {"document": "A large source document."}
    assert len(list((run / "blobs").iterdir())) == 1


def test_run_task_string_takes_a_root_config():
    flow = ScriptedFlow(
        ScriptedLLM([("dig in", block("submit(task)"))]),
        llm_clients={"fast": ScriptedLLM([("dig in", block("submit('fast ' + task)"))])},
    )

    assert (
        flow.run("dig in", root_config=AgentConfig(max_depth=1, max_iters=5, model="fast"))
        == "fast dig in"
    )


def test_graph_spawn_stores_a_config_name_without_a_flow(tmp_path):
    from pydantic import BaseModel

    class Label(BaseModel):
        label: str

    g = Graph()
    agent = g.start("classify", config="classifier", text="hi")

    assert agent.input == {"task": "classify", "text": "hi"}
    assert agent.name == "classifier"
    assert agent.config_name == "classifier"
    loaded = reload(g, tmp_path / "run").agent(agent.id)
    assert loaded.config_name == "classifier"

    flow = ScriptedFlow(ScriptedLLM([("classify", block("submit({'label': text})"))]))
    config = flow.add_config(AgentConfig(name="classifier", output_schema=Label))
    assert config.output_schema["properties"]["label"]["type"] == "string"
    graph = Graph()
    graph.start("classify", config="classifier", text="hi")
    flow.run(graph)
    [classified] = graph.agents()
    assert classified.config is config
    assert classified.final_result() == {"label": "hi"}


def test_graph_edited_without_a_flow_runs_on_a_fresh_flow(tmp_path):
    """Start, append and fork with ops alone; a new Flow builds every REPL."""
    g = Graph()
    a = g.start("count", name="a")
    turn = g.append(a, ran("n = 41"))
    branch = g.fork(turn)
    g.send(a, "finish a")
    branch.send(branch.agent(a.id), "finish b")

    script = [("finish a", block("submit(n + 1)")), ("finish b", block("submit(n + 2)"))]
    loaded = reload(g, tmp_path / "run")
    loaded_branch = reload(branch, tmp_path / "branch")
    ScriptedFlow(ScriptedLLM(list(script))).run(loaded)
    ScriptedFlow(ScriptedLLM(list(script))).run(loaded_branch)

    assert loaded.agent(a.id).final_result() == 42
    assert loaded_branch.agent(a.id).final_result() == 43
    nodes = loaded_branch.agent(a.id).nodes()
    assert nodes[1].code == "n = 41" and nodes[2].text == "finish b"


def test_replay_rebuilds_a_fork_that_never_stepped():
    flow = ScriptedFlow(ScriptedLLM([]))
    g = Graph()
    a = g.start("count", name="a")
    g.append(a, ran("n = 41"))
    branch = g.fork(g.send(a, "go on"))
    copy = branch.agent(a.id)

    asyncio.run(flow.replay(branch, copy))

    assert copy.repl_id != a.repl_id
    assert flow.runtime.get(copy).namespace["n"] == 41 and flow.runtime.get(a) is None
    asyncio.run(flow.aclose())


def test_a_fork_keeps_the_live_repl_or_gets_its_own():
    flow = ScriptedFlow(ScriptedLLM([]))
    g = Graph()
    a = g.start("count", name="a")
    g.append(a, ran("n = 41"))
    tail = g.send(a, "go on")
    asyncio.run(flow.replay(g, a))
    live = flow.runtime.get(a)
    live.namespace["n"] = 50

    same = g.fork(tail, repl="same")
    flow.config_for(same.agent(a.id))
    asyncio.run(flow.prepare_repl(same, same.agent(a.id)))
    assert flow.runtime.get(same.agent(a.id)) is live and live.namespace["n"] == 50

    copy = g.fork(tail)
    flow.config_for(copy.agent(a.id))
    asyncio.run(flow.prepare_repl(copy, copy.agent(a.id)))
    assert flow.runtime.get(copy.agent(a.id)).namespace["n"] == 41 and live.namespace["n"] == 50
    asyncio.run(flow.aclose())


def test_agent_added_mid_stream_is_scheduled():
    llm = ScriptedLLM([("first", block("submit('one')")), ("second", block("submit('two')"))])
    flow = ScriptedFlow(llm)
    g = Graph()
    first = g.start("first")
    added = []

    async def drive():
        async for _step in flow.run_streaming(g):
            if not added:
                added.append(g.start("second"))

    asyncio.run(drive())

    assert first.final_result() == "one"
    assert added[0].final_result() == "two"


def test_keep_n_messages_keeps_the_tail_and_a_truncation_notice():
    llm = ScriptedLLM(
        [
            ("last turn", block("submit('best effort')")),
            ("", block("print('again')")),
        ]
    )
    g, root = agent_graph("loop")
    flow = ScriptedFlow(llm)
    assert (
        run_result(flow, g, root, root_config=AgentConfig(max_iters=6, keep_n_messages=3))
        == "best effort"
    )

    messages = repl.ReplHarness().messages(flow, Step.latest(g, root))
    assert messages[0]["role"] == "system"
    assert len(messages) == 5  # system, truncation notice, then the last three
    assert messages[1]["content"] == repl.TRUNCATION_TEXT
    assert any("again" in message["content"] for message in messages)
    assert all(len(sent) <= 5 for sent in llm.calls)


async def stream(flow, g, until, **kwargs):
    """The nodes a stream yields."""
    return [step.node async for step in flow.run_streaming(g, until=until, **kwargs)]


def test_streaming_until_idle():
    llm = ScriptedLLM([("look", block("print('peek')")), ("peek", block("submit('finished')"))])
    flow = ScriptedFlow(llm)
    g, root = agent_graph("look")
    seen = asyncio.run(stream(flow, g, "idle"))
    assert seen[-1].runs_code and seen[-1].output.strip() == "peek"
    assert not root.status() == "done"

    rest = asyncio.run(stream(flow, g, "done"))
    assert ends(rest[-1]) and root.final_result() == "finished"


def test_each_streamed_step_names_its_agent_and_node():
    llm = ScriptedLLM([("look", block("print('peek')")), ("peek", block("submit('finished')"))])
    flow = ScriptedFlow(llm)
    g, root = agent_graph("look")

    async def collect():
        return [step async for step in flow.run_streaming(g)]

    steps = asyncio.run(collect())
    assert [step.node for step in steps] == own(root)
    assert [step.node.output.strip() for step in steps] == ["peek", ""]
    assert ends(steps[-1].node)
    assert all(step.agent is root and step.graph is g for step in steps)


def test_children_run_in_parallel():
    parent = block(
        "handles = [launch_agent('do a'), launch_agent('do b')]\n"
        "results = await asyncio.gather(*handles)\n"
        "print(results)"
    )
    llm = ScriptedLLM(
        [
            ("split", parent),
            ("do a", block("submit('A')")),
            ("do b", block("submit('B')")),
            ("['A', 'B']", block("submit('AB')")),
        ]
    )
    g, root = agent_graph("split this up")
    flow = ScriptedFlow(llm)
    assert run_result(flow, g, root) == "AB"
    children = root.children()
    assert [child.name for child in children] == ["default-0", "default-1"]
    assert [child.final_result() for child in children] == ["A", "B"]

    mine = {node.id for node in root.nodes()}
    assert all(node.owner == root.id for node in root.nodes())
    assert mine.isdisjoint(node.id for node in children[0].nodes())
    assert len(own(children[0])) == 1 and ends(children[0].nodes()[-1])

    # Children are created during the model turn's turn; the turn records them.
    [turn] = [n for n in root.nodes() if calls.launched(n)]
    assert calls.launched(turn) == [child.id for child in children]
    assert "model" in turn.data
    spawns = g.edges(kind=SPAWN, source=turn.id)
    assert [edge.target for edge in spawns] == [child.id for child in children]
    assert [child.parent() for child in children] == [root, root]

    # Each awaited result returns to the turn, so none is also sent as a message.
    returns = g.edges(kind=RETURN, target=turn.id)
    assert {edge.source for edge in returns} == {child.nodes()[-1].id for child in children}
    assert not messages_in(g.nodes())

    streamed = asyncio.run(stream(ScriptedFlow(llm), agent_graph("split this up")[0], "done"))
    assert sum(ends(node) for node in streamed) == 3


def test_unawaited_children_report_in_one_batched_message():
    class FanOutLLM:
        """Children finish while the parent's next model call is in flight."""

        async def chat(self, messages):
            task, last = first_user(messages), trailing_user_text(messages)
            if task.startswith("child"):
                while not own(root):
                    await asyncio.sleep(0.01)
                return block(f"submit({task.upper()!r})")
            if "Agents you started have finished" in last:
                return block("submit('collected')")
            if "launched" in last:
                while not all(child.status() == "done" for child in root.children()):
                    await asyncio.sleep(0.01)
                return block("print('between turns')")
            return block("launch_agent('child a')\nlaunch_agent('child b')\nprint('launched')")

    g, root = agent_graph("fan out")

    assert run_result(ScriptedFlow(FanOutLLM()), g, root) == "collected"
    children = root.children()
    assert [child.final_result() for child in children] == ["CHILD A", "CHILD B"]
    [message] = messages_in(root.nodes())
    assert message.producer == "rlmflow"
    assert message.input == {"default_0": "CHILD A", "default_1": "CHILD B"}
    assert "[result root.default-0]" in message.text
    assert "[result root.default-1]" in message.text
    returns = g.edges(kind=RETURN, target=message.id)
    assert {edge.source for edge in returns} == {child.nodes()[-1].id for child in children}


def test_awaiting_an_agent_you_did_not_launch_raises_in_the_turn():
    llm = ScriptedLLM(
        [
            ("you can only await agents you launched", block("submit('refused')")),
            ("curious", block("print(await AGENTS['peer'])")),
            ("peer task", block("submit('peer answer')")),
        ]
    )
    g = Graph()
    peer = g.start("peer task", name="peer")
    curious = g.start("curious", name="curious")

    flow = ScriptedFlow(llm)
    flow.run(g)

    assert peer.final_result() == "peer answer"
    assert curious.final_result() == "refused"
    turn = code_nodes(g, curious)[0]
    assert turn.error
    assert "NotYourAgent" in turn.output
    assert not g.edges(kind=RETURN, target=turn.id)


def test_a_worker_repl_raises_host_errors_as_their_own_types():
    code = (
        "try:\n    await AGENTS['peer']\nexcept NotYourAgent:\n    print('not mine')\n"
        "try:\n    launch_agent('x', spec=AGENTS['peer'])\nexcept TypeError:\n    print('bad arg')"
    )
    llm = ScriptedLLM(
        [
            ("not mine", block("submit('ok')")),
            ("curious", block(code)),
            ("peer task", block("submit('peer answer')")),
        ]
    )
    g = Graph()
    g.start("peer task", name="peer")
    curious = g.start("curious", name="curious")

    ScriptedFlow(llm, runtime=LocalRuntime(repl_timeout=10)).run(g)

    turn = code_nodes(g, curious)[0]
    assert turn.output.split() == ["not", "mine", "bad", "arg"]
    assert curious.final_result() == "ok"


def test_a_launch_argument_must_be_a_result_not_an_agent():
    llm = ScriptedLLM(
        [
            (
                "await the agent first",
                block(
                    "consumer = launch_agent('consume-source', source=await source)\n"
                    "print(await consumer)"
                ),
            ),
            (
                "build-dag",
                block(
                    "source = launch_agent('produce-source')\n"
                    "launch_agent('consume-source', source=source)"
                ),
            ),
            ("produce-source", block("submit('A')")),
            ("consume-source", block("submit(source + '!')")),
            ("A!", block("submit('done')")),
        ]
    )
    g, root = agent_graph("build-dag")

    assert run_result(ScriptedFlow(llm), g, root) == "done"
    source, consumer = root.children()
    assert source.task == "produce-source"
    assert consumer.input == {"task": "consume-source", "source": "A"}
    [refused] = [node for node in root.nodes() if node.error]
    assert "TypeError" in refused.output
    assert calls.launched(refused) == [source.id]


@pytest.mark.parametrize(
    "args",
    [
        {"nested": {"handle": "not-an-agent-ref"}},
        {"nested": ["not", "top-level"]},
        {"number": 3},
        {"described": Input({"rows": [1, 2]}, description="Rows.")},
    ],
)
def test_launch_args_accept_nested_json_values(args):
    assert check_args(args) == args


@pytest.mark.parametrize("value", [{1, 2}, object(), {1: "bad key"}])
def test_launch_args_reject_non_json_values(value):
    with pytest.raises(TypeError, match="JSON-compatible|non-string"):
        check_args({"value": value})


@pytest.mark.parametrize("name", ["bad-name", "class", "task"])
def test_launch_args_require_python_variable_names(name):
    with pytest.raises(ValueError, match="invalid invocation|'task' is the invocation"):
        check_args({name: "value"})


@pytest.mark.parametrize(
    "value",
    [
        AgentHandle("agent_source", "source", "root.source"),
        [AgentHandle("agent_source", "source", "root.source")],
        {"ref": {"$rlmflow_agent": "agent_source"}},
        Input(AgentHandle("agent_source", "source", "root.source"), description="Source."),
    ],
)
def test_launch_args_reject_agents_anywhere(value):
    with pytest.raises(TypeError, match="await the agent first and pass its result"):
        check_args({"evidence": value})


def test_plan_query_allows_iterative_investigation_before_delegation():
    launch = block(
        "handles = [launch_agent('alpha-workstream'), launch_agent('beta-workstream')]\n"
        "print(await asyncio.gather(*handles))",
    )
    llm = ScriptedLLM(
        [
            ("build the requested system", block("print(brief[:7])")),
            ("preview", block("requirements = brief.splitlines()[1:]\nprint(requirements)")),
            ("requirement A", launch),
            ("alpha-workstream", block("submit('A')")),
            ("beta-workstream", block("submit('B')")),
            ("['A', 'B']", block("submit('done')")),
        ]
    )
    g, root = agent_graph(
        "build the requested system",
        args={"brief": "preview\nrequirement A\nrequirement B"},
    )

    assert run_result(ScriptedFlow(llm), g, root) == "done"
    assert len(model_turns(root.nodes())) == 4
    assert len(root.children()) == 2
    outputs = [node.output.strip() for node in code_nodes(g, root)]
    assert outputs[:2] == ["preview", "['requirement A', 'requirement B']"]


def test_save_writes_a_run_directory(tmp_path):
    parent = block("child = launch_agent('do a')\nprint(await child)")
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("submit('A')")), ("A", block("submit('AB')"))]
    )
    g, root = agent_graph("split this up", args={"doc": "hi"})
    assert run_result(ScriptedFlow(llm), g, root) == "AB"

    run = g.save(tmp_path / "run")
    saved = json.loads((run / "graph.json").read_text())
    assert set(saved) == {"format", "id", "metadata", "nodes", "edges"}
    assert saved["format"] == FORMAT_VERSION == 7
    assert saved["id"] == g.id
    assert [record["id"] for record in saved["nodes"]] == [node.id for node in g.nodes()]
    assert [record["id"] for record in saved["edges"]] == [edge.id for edge in g.edges()]
    root_record = saved["nodes"][0]
    assert root_record["input"] == {"task": "split this up", "doc": "hi"}
    assert "history" not in root_record["agent"]
    owned = [record["id"] for record in saved["nodes"] if record["owner"] == root.id]
    assert owned == [node.id for node in root.nodes()]

    child = root.children()[0]
    [turn] = [node for node in own(root) if calls.launched(node)]
    assert {"kind": SPAWN, "source": turn.id, "target": child.id} in [
        {key: edge[key] for key in ("kind", "source", "target")} for edge in saved["edges"]
    ]
    child_record = next(record for record in saved["nodes"] if record["id"] == child.id)
    assert child_record["agent"]["task"] == "do a"
    assert child_record["agent"]["config"] == "default"
    child_nodes = [record for record in saved["nodes"] if record["owner"] == child.id]
    assert [record["id"] for record in child_nodes] == [node.id for node in child.nodes()]


def test_run_records_seq_and_timing():
    parent = block(
        "handles = [launch_agent('do a'), launch_agent('do b')]\n"
        "print(await asyncio.gather(*handles))"
    )
    llm = ScriptedLLM(
        [
            ("split", parent),
            ("do a", block("submit('A')")),
            ("do b", block("submit('B')")),
            ("['A', 'B']", block("submit('AB')")),
        ]
    )
    g, root = agent_graph("split this up")
    assert run_result(ScriptedFlow(llm), g, root) == "AB"

    # seq is the graph's log position, so each agent's nodes are strictly increasing and
    # the log order is the whole run, however the two children raced.
    for agent in (root, *root.children()):
        seqs = [node.seq for node in agent.nodes()]
        assert seqs == sorted(set(seqs))
    every = [*g.nodes(), *g.edges()]
    assert sorted(item.seq for item in every) == list(range(len(every)))

    # Model turns and code_nodes are timed; messages are written by senders.
    stepped = model_turns(g.nodes())
    assert stepped and all(node.finished_at >= node.started_at for node in stepped)
    written = g.send(root, "and then?")
    assert written.started_at is None
    assert written.seq == len(every)


def test_structured_output_is_stored_parsed():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM([("count", block("submit({'n': 41})"))])
    g, root = agent_graph("count")
    assert run_result(
        ScriptedFlow(llm), g, root, root_config=AgentConfig(output_schema=schema)
    ) == {"n": 41}
    assert node_record(root.nodes()[-1])["result"] == {"n": 41}


def test_plain_text_json_reply_is_validated_against_the_schema():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM([("count", '{"n": 41}')])
    g, root = agent_graph("count")

    assert run_result(
        ScriptedFlow(llm), g, root, root_config=AgentConfig(output_schema=schema)
    ) == {"n": 41}
    assert not any(node.runs_code for node in g.nodes())


def test_plain_text_reply_that_fails_the_schema_is_a_code_error():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM(
        [
            ("count", "forty-one"),
            ("not a valid final answer", block("submit({'n': 41})")),
        ]
    )
    g, root = agent_graph("count")

    assert run_result(
        ScriptedFlow(llm), g, root, root_config=AgentConfig(output_schema=schema)
    ) == {"n": 41}
    [rejected] = [n for n in g.nodes() if n.error]
    assert not rejected.runs_code


def test_unstructured_submission_keeps_json_values():
    expected = {"n": [41, True, None, "é"]}
    llm = ScriptedLLM([("value", block(f"submit({expected!r})"))])
    g, root = agent_graph("return value")

    assert run_result(ScriptedFlow(llm), g, root) == expected
    assert node_record(root.nodes()[-1])["result"] == expected


def test_unstructured_submission_rejects_non_json_values():
    llm = ScriptedLLM(
        [
            ("return value", block("submit({1, 2})")),
            ("JSON", block("submit('recovered')")),
        ]
    )
    g, root = agent_graph("return value")

    assert run_result(ScriptedFlow(llm), g, root) == "recovered"
    assert any(n.runs_code and n.error and "JSON" in n.output for n in root.nodes())


def test_json_submission_crosses_worker_and_persistence_boundary(tmp_path):
    g, root = agent_graph("return value")
    flow = ScriptedFlow(
        ScriptedLLM([("return value", block("submit({'n': [1, True, None]})"))]),
        runtime=LocalRuntime(repl_timeout=10),
    )

    try:
        assert run_result(flow, g, root) == {"n": [1, True, None]}
    finally:
        asyncio.run(flow.aclose())
    loaded = reload(g, tmp_path / "json-submission")
    assert loaded.agent(root.id).final_result() == {"n": [1, True, None]}


def test_worker_submission_rejects_non_json_values_and_recovers():
    llm = ScriptedLLM(
        [
            ("return value", block("submit({1, 2})")),
            ("JSON", block("submit({'ok': True})")),
        ]
    )
    flow = ScriptedFlow(llm, runtime=LocalRuntime(repl_timeout=10))

    g, root = agent_graph("return value")
    try:
        assert run_result(flow, g, root) == {"ok": True}
    finally:
        asyncio.run(flow.aclose())


def test_plain_text_reply_finishes_the_agent():
    llm = ScriptedLLM([("poem", "The moon writes silver on the sea.")])
    g, root = agent_graph("write a poem")

    assert run_result(ScriptedFlow(llm), g, root) == "The moon writes silver on the sea."
    assert not any(node.runs_code for node in g.nodes())


def test_plain_text_reply_may_contain_non_repl_code_fences():
    answer = "Use a list comprehension:\n```python\nprint([x * 2 for x in xs])\n```"
    llm = ScriptedLLM([("explain", answer)])
    g, root = agent_graph("explain doubling")

    assert run_result(ScriptedFlow(llm), g, root) == answer
    assert not any(node.runs_code for node in g.nodes())


@pytest.mark.parametrize(
    ("invalid_reply", "reason"),
    [
        ("```repl\n```", "the ```repl block is empty"),
        ("```repl\nunclosed = True", "the ```repl block was never closed"),
        ("", "the reply is empty"),
        ("```repl\nmarker = 'first'\n```\n```repl\nmarker = 'second'\n```", "found 2 ```repl"),
    ],
)
def test_invalid_reply_becomes_a_code_error_and_recovers(invalid_reply, reason):
    llm = ScriptedLLM(
        [
            ("bad response", invalid_reply),
            (
                "Nothing ran",
                block("submit('dirty' if {'unclosed', 'marker'} & set(globals()) else 'clean')"),
            ),
        ]
    )
    g, root = agent_graph("bad response")

    assert run_result(ScriptedFlow(llm), g, root) == "clean"
    [error] = [node for node in g.nodes() if node.error]
    assert not error.runs_code
    assert error.output.startswith(reason)
    assert "Nothing ran" in error.output
    assert "reply in plain text with no ```repl block" in error.output


def test_protocol_error_survives_save_load_and_rebuild(tmp_path):
    llm = ScriptedLLM(
        [
            ("count", "```repl\n```"),
            ("Nothing ran", block("secret = 41\nprint('stored')")),
        ]
    )
    g, root = agent_graph("count for me")
    asyncio.run(
        stream(
            ScriptedFlow(llm),
            g,
            lambda step: step.node.runs_code and not step.node.error,
        )
    )
    loaded = reload(g, tmp_path / "protocol-error")

    resumed = ScriptedLLM([("stored", block("submit(secret + 1)"))])
    assert run_result(ScriptedFlow(resumed), loaded, loaded.agent(root.id)) == 42
    assert sum(1 for n in loaded.nodes() if n.error) == 1


def test_plain_reply_on_final_turn_finishes_the_run():
    g, root = agent_graph("answer now")

    llm = ScriptedLLM([("last turn", "best supported answer"), ("answer now", block("x = 1"))])
    assert (
        run_result(ScriptedFlow(llm), g, root, root_config=AgentConfig(max_iters=2))
        == "best supported answer"
    )


def test_final_turn_without_submit_ends_at_max_iters():
    g, root = agent_graph("answer now")

    llm = ScriptedLLM([("last turn", block("y = 2")), ("answer now", block("x = 1"))])
    assert (
        run_result(ScriptedFlow(llm), g, root, root_config=AgentConfig(max_iters=2))
        == "[max_iters exceeded]"
    )
    assert len(llm.calls) == 2


def test_save_load_roundtrip(tmp_path):
    parent = block("child = launch_agent('do a')\nprint(await child)")
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("submit('A')")), ("A", block("submit('AB')"))]
    )
    g, root = agent_graph("split this up", args={"doc": "hi"})
    run_result(ScriptedFlow(llm), g, root)

    loaded = reload(g, tmp_path / "run")
    loaded_root = loaded.agent(root.id)
    assert [agent.id for agent in loaded.agents()] == [agent.id for agent in g.agents()]
    assert [node.id for node in loaded.nodes()] == [node.id for node in g.nodes()]
    assert [edge.id for edge in loaded.edges()] == [edge.id for edge in g.edges()]
    assert (loaded_root.final_result(), loaded_root.status() == "done") == ("AB", True)
    [child] = loaded_root.children()
    assert child.path() == "root.default-0"
    assert child.final_result() == "A"
    assert loaded.save(tmp_path / "again") and (tmp_path / "again" / "graph.json").is_file()


def paused_run(tmp_path, name="run"):
    """A saved run that set a variable and stopped before using it."""
    llm = ScriptedLLM([("count", block("secret = 41\nprint('stored')"))])
    g, root = agent_graph("count for me")
    asyncio.run(stream(ScriptedFlow(llm), g, "idle"))
    assert root.nodes()[-1].runs_code
    loaded = reload(g, tmp_path / name)
    return loaded, loaded.agent(root.id)


def test_resuming_a_saved_run_rebuilds_its_repl(tmp_path):
    loaded, root = paused_run(tmp_path)
    llm = ScriptedLLM([("stored", block("submit(secret + 1)"))])
    assert run_result(ScriptedFlow(llm), loaded, root) == 42
    assert not any(n.error for n in loaded.nodes())
    assert len(llm.calls) == 1  # the rebuild itself asks the model nothing


def test_the_rebuild_waits_until_code_needs_the_repl(tmp_path):
    loaded, root = paused_run(tmp_path)
    flow = ScriptedFlow(ScriptedLLM([("stored", "plain answer")]))

    assert run_result(flow, loaded, root) == "plain answer"
    assert flow.runtime.get(root) is None

    loaded.send(root, "use it")
    flow.test_llm = ScriptedLLM([("use it", block("submit(secret + 1)"))])
    assert run_result(flow, loaded, root) == 42
    assert flow.runtime.get_var(root, "secret") == 41


def test_halting_mid_delegation_resumes_without_relaunching():
    def reply(messages):
        task = first_user(messages)
        if task == "parent":
            return block(
                "seed = 40\n"
                "slow = launch_agent('slow', name='slow')\n"
                "fast = launch_agent('fast', name='fast')\n"
                "results = await asyncio.gather(slow, fast)\n"
                "submit(f'{seed}:' + ''.join(results))"
            )
        calls.append(task)
        if task == "slow":
            return block("await asyncio.sleep(0.2)\nsubmit('S')")
        return block("submit('F')")

    calls: list[str] = []
    g, root = agent_graph("parent")
    flow = ScriptedFlow(StubLLM(reply), runtime=LocalRuntime(repl_timeout=10))

    async def halt_at_first_child():
        async for _ in flow.run_streaming(g, until=lambda step: step.agent is not root):
            pass

    try:
        asyncio.run(halt_at_first_child())
        [turn] = code_nodes(g, root)
        assert turn.state != "done"
        kept = list(root.children())
        assert kept
        assert run_result(flow, g, root) == "40:SF"
        assert sorted(child.name for child in root.children()) == ["fast", "slow"]
        assert [node.id for node in code_nodes(g, root)] == [turn.id]
        assert set(kept) <= set(root.children())
        assert sorted(calls) == ["fast", "slow"]
    finally:
        asyncio.run(flow.aclose())


def test_rebuild_reads_recorded_child_answers_instead_of_relaunching(tmp_path):
    parent = block("child = launch_agent('add up')\nanswer = await child\nprint(answer)")
    llm = ScriptedLLM([("delegate", parent), ("add up", block("submit('20')"))])
    g, root = agent_graph("delegate")

    def parent_turn_ended(step):
        return step.node.runs_code and step.agent is root

    asyncio.run(
        stream(ScriptedFlow(llm), g, parent_turn_ended, root_config=AgentConfig(max_depth=2))
    )
    loaded = reload(g, tmp_path / "run")
    loaded_root = loaded.agent(root.id)

    resumed = ScriptedLLM([("20", block("submit('parent saw ' + answer)"))])
    assert run_result(ScriptedFlow(resumed), loaded, loaded_root) == "parent saw 20"
    # The child ran once, when it was recorded: no second launch, no second model call.
    assert len(loaded_root.children()) == 1
    assert len(resumed.calls) == 1


def test_rebuild_restores_llm_query_result_without_another_model_call(tmp_path):
    setup = ScriptedLLM(
        [
            ("delegate", block("answer = await llm_query('extract')\nprint(answer)")),
            ("extract", "recorded"),
        ]
    )
    g, root = agent_graph("delegate")

    setup_flow = ScriptedFlow(setup, runtime=LocalRuntime(repl_timeout=10))
    asyncio.run(stream(setup_flow, g, lambda step: step.node.runs_code))
    asyncio.run(setup_flow.aclose())
    loaded = reload(g, tmp_path / "llm-query-run")
    resumed = ScriptedLLM([("recorded", block("submit(answer)"))])

    flow = ScriptedFlow(resumed, runtime=LocalRuntime(repl_timeout=10))
    try:
        assert run_result(flow, loaded, loaded.agent(root.id)) == "recorded"
    finally:
        asyncio.run(flow.aclose())
    assert len(resumed.calls) == 1


GATHER_THREE = block(
    "kids = [launch_agent(f'child {i}') for i in range(3)]\nprint(await asyncio.gather(*kids))"
)


def gathered_run():
    """A finished run whose root awaited three children in one turn."""
    llm = ScriptedLLM(
        [
            ("'CHILD 2']", block("submit('original')")),
            ("fan out", GATHER_THREE),
            ("child", block("submit(task.upper())")),
        ]
    )
    g, root = agent_graph("fan out")
    assert run_result(ScriptedFlow(llm), g, root) == "original"
    turn = code_nodes(g, root)[0]
    return g, root, turn


def test_replacing_an_awaited_result_reruns_the_turn_in_place():
    g, root, turn = gathered_run()
    children = root.children()
    saved = turn.calls
    assert [entry["tool"] for entry in saved] == ["launch_agent"] * 3 + ["await"] * 3
    assert saved[4]["agent"] == children[1].id

    finished = own(root)
    preview = g.replace(turn, 4, "SWAPPED", dry_run=True)
    assert preview.rerun == [turn.id]
    assert own(root) == finished and "value" not in turn.calls[4]

    invalidation = g.replace(turn, 4, "SWAPPED")
    assert invalidation.rerun == [turn.id]
    assert invalidation.rewound == [root.id]
    assert invalidation.removed == []
    assert own(root) == [turn]
    assert turn.state == READY and root.status() == "running"
    assert turn.calls[4]["value"] == "SWAPPED"

    resumed = ScriptedLLM([("SWAPPED", block("submit('swapped')"))])
    assert run_result(ScriptedFlow(resumed), g, root) == "swapped"
    assert own(root)[0] is turn and len(own(root)) == 2
    assert turn.state == "done"
    assert turn.output.strip() == "['CHILD 0', 'SWAPPED', 'CHILD 2']"
    # Nothing relaunched, and the swapped-out child's result isn't delivered as a message.
    assert root.children() == children
    assert len(resumed.calls) == 1
    assert not messages_in(g.nodes())
    assert len(g.edges(kind=RETURN, target=turn.id)) == 3


def test_editing_a_childs_result_reruns_the_turn_that_awaited_it(tmp_path):
    g, root, turn = gathered_run()
    loaded = reload(g, tmp_path / "run")
    loaded_root, loaded_turn = loaded.agent(root.id), loaded.node(turn.id)
    first = loaded_root.children()[0]

    invalidation = loaded.edit(first.nodes()[-1], result="EDITED")
    assert invalidation.rerun == [turn.id]
    assert root.id in invalidation.rewound
    assert first.final_result() == "EDITED"
    assert own(loaded_root) == [loaded_turn]
    assert loaded_turn.state == READY

    resumed = ScriptedLLM([("EDITED", block("submit('edited')"))])
    assert run_result(ScriptedFlow(resumed), loaded, loaded_root) == "edited"
    assert loaded_turn.output.strip() == "['EDITED', 'CHILD 1', 'CHILD 2']"
    assert len(resumed.calls) == 1
    assert loaded.edges(kind=RETURN, source=first.nodes()[-1].id, target=turn.id)
    assert not messages_in(loaded.nodes())


JOIN_CHILDREN = block("print([await c for c in AGENTS.children])")


def test_host_spawned_child_runs_and_returns_to_its_parent(tmp_path):
    g, root = agent_graph("parent")
    branch = g.start("child", parent=root, name="branch0")
    [spawn] = g.edges(kind=SPAWN, target=branch.id)
    assert spawn.source == root.id
    assert branch.config_name == "default"

    loaded = reload(g, tmp_path / "run")
    loaded_root, loaded_branch = loaded.agent(root.id), loaded.agent(branch.id)
    assert loaded_root.children() == [loaded_branch]

    flow = ScriptedFlow(
        ScriptedLLM(
            [
                ("['finished']", block("submit('finished')")),
                ("parent", JOIN_CHILDREN),
                ("child", block("submit('finished')")),
            ]
        )
    )

    assert run_result(flow, loaded, loaded_root) == "finished"
    assert loaded_branch.path() == "root.branch0"
    turn = code_nodes(loaded, loaded_root)[0]
    [returned] = loaded.edges(kind=RETURN, target=turn.id)
    assert returned.source == loaded_branch.nodes()[-1].id


def test_stopping_on_a_childs_done_still_delivers_its_result():
    g = Graph()
    root = g.start("parent")
    g.append(root, Node("assistant", data={"seen": 1, "idle": True}))
    child = g.start("child", parent=root, name="child")
    flow = ScriptedFlow(ScriptedLLM([("child", block("submit('finished')"))]))

    async def drive() -> None:
        async for _ in flow.run_streaming(g, until=lambda s: ends(s.node)):
            pass

    asyncio.run(drive())

    message = root.nodes()[-1]
    assert "finished" in message.text and message.input == {"child": "finished"}
    [returned] = g.edges(kind=RETURN, target=message.id)
    assert returned.source == child.nodes()[-1].id


def test_host_adds_prepared_children_between_streams():
    llm = ScriptedLLM(
        [
            ("['A', 'B']", block("submit('AB')")),
            ("prepared", JOIN_CHILDREN),
            ("prepare", block("print('prepared')")),
            ("child a", block("submit(value)")),
            ("child b", block("submit(value)")),
        ]
    )
    flow = ScriptedFlow(llm)
    g, root = agent_graph("prepare")

    def prepared(task, name, code):
        """A child with a recorded turn, as a controller would prepare it."""
        agent = g.start(task, parent=root, name=name)
        g.append(agent, ran(code, task))
        return agent

    async def main():
        async for _step in flow.run_streaming(g, until="idle"):
            pass

        prepared("child a", "a", "value = 'A'")
        prepared("child b", "b", "value = 'B'")

        async for _step in flow.run_streaming(g):
            pass

    asyncio.run(main())

    assert root.final_result() == "AB"
    assert [child.name for child in root.children()] == ["a", "b"]
    assert [child.final_result() for child in root.children()] == ["A", "B"]


def test_rebuilding_agents_can_skip_their_own_unrepeatable_work(tmp_path):
    trace: list[str] = []

    @tool("Record a live side effect.")
    def log(value):
        trace.append(value)

    code = block("if ENV['RLMFLOW_REPLAY'] == '0':\n    log('live')\nprint('set up')")
    flow = ScriptedFlow(ScriptedLLM([("count", code)]))
    flow.inject("log", log)
    g, root = agent_graph("count for me")
    asyncio.run(stream(flow, g, "idle"))
    assert trace == ["live"]

    loaded = reload(g, tmp_path / "run")
    resumed = ScriptedFlow(ScriptedLLM([("set up", block("submit('ok')"))]))
    resumed.inject("log", log)
    assert run_result(resumed, loaded, loaded.agent(root.id)) == "ok"
    assert trace == ["live"]  # the rebuild saw RLMFLOW_REPLAY=1 and skipped it


def test_a_finished_run_resumes_to_its_recorded_answer(tmp_path):
    llm = ScriptedLLM([("count", block("submit('six')"))])
    g, root = agent_graph("count for me")
    assert run_result(ScriptedFlow(llm), g, root) == "six"

    loaded = reload(g, tmp_path / "run")
    idle = ScriptedLLM([])
    assert run_result(ScriptedFlow(idle), loaded, loaded.agent(root.id)) == "six"
    assert idle.calls == []  # nothing left to step


def test_a_dead_repl_is_an_observation_and_is_rebuilt():
    class OneDeadRepl(LocalRuntime):
        """The first REPL dies on its second turn, the way a sandbox does."""

        opened = 0

        def open(self, agent):
            self.opened += 1
            repl = super().open(agent)
            if self.opened == 1:
                run, code_nodes = repl.run, []

                async def die_on_second(code, **kwargs):
                    code_nodes.append(code)
                    if len(code_nodes) == 2:
                        raise ConnectionResetError("sandbox went away")
                    return await run(code, **kwargs)

                repl.run = die_on_second
            return repl

    llm = ScriptedLLM(
        [
            ("count", block("x = 1\nprint('first')")),
            ("first", block("x += 1\nprint('again')")),
            ("sandbox went away", block("print(x)")),
            ("1", block("submit('recovered')")),
        ]
    )
    runtime = OneDeadRepl()
    g, root = agent_graph("count for me")
    flow = ScriptedFlow(llm, runtime=runtime)
    try:
        assert run_result(flow, g, root) == "recovered"
    finally:
        asyncio.run(flow.aclose())

    [dead] = [node for node in g.nodes() if node.data.get("dead")]
    assert "REPL execution failed" in dead.output
    assert "rebuilt from the recorded nodes" in dead.output
    assert runtime.opened == 2  # rebuilt from the first turn, without the dead one


def test_each_launch_call_in_one_turn_creates_a_distinct_child():
    parent = block(
        "first = launch_agent('do it')\n"
        "second = launch_agent('again')\n"
        "assert first.id != second.id\n"
        "print(await asyncio.gather(first, second))"
    )
    llm = ScriptedLLM(
        [
            ("delegate", parent),
            ("do it", block("submit('once')")),
            ("again", block("submit('twice')")),
            ("['once', 'twice']", block("submit('done')")),
        ]
    )
    g, root = agent_graph("delegate")
    assert run_result(ScriptedFlow(llm), g, root, root_config=AgentConfig(max_depth=2)) == "done"
    assert [child.name for child in root.children()] == ["default-0", "default-1"]


def test_launch_agent_accepts_a_stable_child_name():
    parent = block(
        "worker = launch_agent('do it', name='worker_events_01')\n"
        "assert worker.name == 'worker_events_01'\n"
        "print(await worker)"
    )
    llm = ScriptedLLM(
        [
            ("delegate", parent),
            ("do it", block("submit('done')")),
            ("done", block("submit('complete')")),
        ]
    )
    g, root = agent_graph("delegate")
    assert run_result(ScriptedFlow(llm), g, root) == "complete"
    assert [child.name for child in root.children()] == ["worker_events_01"]


def test_saving_again_rewrites_the_whole_log(tmp_path):
    parent = block("child = launch_agent('do a')\nprint(await child)")
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("submit('A')")), ("A", block("submit('AB')"))]
    )
    g, root = agent_graph("split this up")
    run_result(ScriptedFlow(llm), g, root)
    run = g.save(tmp_path / "run")

    follow_up = g.send(root, "and then?")
    g.save(run)

    loaded = Graph.load(run)
    assert [node.id for node in loaded.nodes()] == [node.id for node in g.nodes()]
    assert [edge.id for edge in loaded.edges()] == [edge.id for edge in g.edges()]
    assert loaded.agent(root.id).nodes()[-1] is loaded.node(follow_up.id)


def test_max_depth_refusal():
    llm = ScriptedLLM(
        [
            ("deep", block("print(launch_agent('x'))")),
            ("is not defined", block("submit('stopped')")),
        ]
    )
    shallow = AgentConfig(max_depth=0)
    g, root = agent_graph("go deep")
    root.bind_config(shallow)
    flow = ScriptedFlow(llm)
    with stepping(g, root):
        assert "launch_agent" not in repl_tools(flow, Step(g, root, root))
    assert run_result(flow, g, root, root_config=shallow) == "stopped"
    assert root.children() == []


def test_grandchildren():
    llm = ScriptedLLM(
        [
            ("plan the work", block("mid = launch_agent('go mid')\nprint(await mid)")),
            ("go mid", block("leaf = launch_agent('go leaf')\nprint(await leaf)")),
            ("go leaf", block("submit('leaf')")),
            ("leaf", block("submit('mid')")),
            ("mid", block("submit('root')")),
        ]
    )
    g, root = agent_graph("plan the work")
    assert run_result(ScriptedFlow(llm, configs=[AgentConfig(max_depth=2)]), g, root) == "root"
    mid = root.children()[0]
    assert [agent.path() for agent in (mid, mid.children()[0])] == [
        "root.default-0",
        "root.default-0.default-0",
    ]


class BarrierLLM:
    """Blocks each caller until every expected child has asked for a reply."""

    def __init__(self, parties, replies):
        self.barrier = asyncio.Barrier(parties)
        self.replies = replies

    async def chat(self, messages):
        last = trailing_user_text(messages)
        for needle, reply in self.replies:
            if needle in last:
                if needle.startswith("child"):
                    await self.barrier.wait()
                return reply
        raise AssertionError(f"no scripted reply for {last!r}")


def test_children_step_concurrently():
    calls = ", ".join(f"launch_agent('child {i}')" for i in range(3))
    llm = BarrierLLM(
        3,
        [
            ("fan out", block(f"handles = [{calls}]\nprint(await asyncio.gather(*handles))")),
            *[(f"child {i}", block(f"submit({i})")) for i in range(3)],
            ("[0, 1, 2]", block("submit('all')")),
        ],
    )
    g, root = agent_graph("fan out")

    async def main():
        # Sequential children would never clear the barrier, so this would hang.
        await asyncio.wait_for(ScriptedFlow(llm).arun(g), timeout=5)
        return root.final_result()

    assert asyncio.run(main()) == "all"
    assert [child.final_result() for child in root.children()] == [0, 1, 2]


def test_background_child_outlives_root_and_stream_drains_it():
    release_child = threading.Event()

    class BackgroundLLM:
        def chat(self, messages):
            last = "\n\n".join(
                message["content"] for message in messages if message["role"] == "user"
            )
            if "slow child" in last:
                assert release_child.wait(timeout=5)
                return block("submit('child finished')")
            return block("child = launch_agent('slow child')\nsubmit(child.status)")

    flow = ScriptedFlow(BackgroundLLM(), workers=2)
    g = Graph()
    root = g.start("launch without joining")

    async def run():
        observed = []
        async for step in flow.run_streaming(g):
            observed.append(step.node)
            if ends(step.node) and step.agent is root:
                assert root.final_result() == "running"
                assert not root.children()[0].status() == "done"
                release_child.set()
        await flow.aclose()
        return observed

    observed = asyncio.run(run())
    child = root.children()[0]
    assert child.final_result() == "child finished"
    assert root.final_result() == "running"  # done is final: the child's result doesn't reopen it
    root_done = next(i for i, node in enumerate(observed) if node.owner == root.id and ends(node))
    child_done = next(i for i, node in enumerate(observed) if node.owner == child.id and ends(node))
    assert root_done < child_done


def test_child_handle_waits_on_a_later_turn():
    wait = block("submit(await later)")
    llm = ScriptedLLM(
        [
            (
                "start later",
                block("later = launch_agent('child task')\nprint('queued', later.name)"),
            ),
            ("child task", block("submit('child answer')")),
            ("queued default-0", wait),
            ("[result root.default-0]", wait),
        ]
    )
    g, root = agent_graph("start later")

    assert run_result(ScriptedFlow(llm), g, root) == "child answer"
    assert len(root.children()) == 1
    assert root.children()[0].final_result() == "child answer"


def test_completed_child_result_access():
    llm = ScriptedLLM(
        [
            ("start child", block("child = launch_agent('child task')\nprint('queued')")),
            ("child task", block("submit(42)")),
            ("result ready", block("submit(child.result())")),
            ("queued", block("print(child.status)")),
            ("running", block("print(child.status)")),
        ]
    )
    g, root = agent_graph("start child")

    assert run_result(ScriptedFlow(llm), g, root) == 42
    assert root.children()[0].final_result() == 42


def test_launch_agent_binds_task_and_json_arguments_directly():
    llm = ScriptedLLM(
        [
            ("delegate", block("child = launch_agent('child', wait=False)\nprint(await child)")),
            ("child:False", block("submit('recovered')")),
            ("child", block("submit(task + ':' + str(wait))")),
        ]
    )
    g, root = agent_graph("delegate")

    assert run_result(ScriptedFlow(llm), g, root) == "recovered"
    assert root.children()[0].input == {"task": "child", "wait": False}


def test_launch_agent_accepts_described_inputs():
    class InputLLM:
        def __init__(self):
            self.child_system_prompt = ""

        def chat(self, messages):
            if first_user(messages) == "child":
                self.child_system_prompt = messages[0]["content"]
                return block("submit(rows[0]['label'])")
            return block(
                "child = launch_agent(\n"
                "    'child',\n"
                "    rows=Input(rows, description='Records with label fields.'),\n"
                ")\n"
                "submit(await child)"
            )

    llm = InputLLM()
    g, root = agent_graph("delegate", args={"rows": [{"label": "private-marker-7"}]})

    assert run_result(ScriptedFlow(llm), g, root) == "private-marker-7"
    child = root.children()[0]
    assert child.input == {"task": "child", "rows": [{"label": "private-marker-7"}]}
    assert child.arg_descriptions == {"rows": "Records with label fields."}
    assert "Records with label fields." in llm.child_system_prompt
    assert "private-marker-7" not in llm.child_system_prompt


def test_agents_result_matches_a_collected_child():
    llm = ScriptedLLM(
        [
            ("inspect waited", block("waited = launch_agent('child task')\nprint(await waited)")),
            ("child task", block("submit('child answer')")),
            ("child answer", block("submit(await AGENTS[waited])")),
        ]
    )
    flow = ScriptedFlow(llm)
    g = Graph()
    root = g.start("inspect waited")

    assert run_result(flow, g, root) == "child answer"


def test_break_cancels_the_run():
    llm = ScriptedLLM([("", block("print('again')"))])
    flow = ScriptedFlow(llm)

    async def main():
        stream = flow.run_streaming(agent_graph("loop forever")[0])
        async for _step in stream:
            break
        await stream.aclose()
        assert not flow.tasks
        await flow.aclose()

    asyncio.run(main())


def test_step_failure_reaches_the_caller():
    class BrokenLLM:
        def chat(self, messages):
            raise RuntimeError("model is down")

    g, root = agent_graph("anything")
    with pytest.raises(RuntimeError, match="model is down"):
        run_result(ScriptedFlow(BrokenLLM()), g, root)
    assert root.final_result() == "[run failed: RuntimeError: model is down]"


def test_custom_tool_is_injected():
    @tool("Upper-case a string.", where="repl")
    def shout(text: str) -> str:
        return text.upper()

    llm = ScriptedLLM([("", block("submit(shout('hi'))"))])
    flow = ScriptedFlow(llm, tools=[shout])
    g, root = agent_graph("say it")
    assert run_result(flow, g, root) == "HI"


def prompt_text(llm) -> str:
    return "\n".join(message["content"] for message in llm.calls[0])


def test_a_plain_function_is_a_host_tool_described_by_its_docstring():
    lock = threading.Lock()  # unpicklable, so this only works if the call stays on the host
    calls: list[str] = []

    def lookup(word: str) -> str:
        """Look up a word in the host's dictionary."""
        with lock:
            calls.append(word)
        return word[::-1]

    llm = ScriptedLLM([("", block("submit(lookup('abc'))"))])
    flow = ScriptedFlow(llm, tools=[lookup])
    g, root = agent_graph("look it up")
    assert run_result(flow, g, root) == "cba"
    assert calls == ["abc"]
    assert "`lookup(word: str) -> str`: Look up a word in the host's dictionary." in prompt_text(
        llm
    )


def test_a_tool_from_a_json_schema_gets_a_signature_and_dict_arguments():
    received: list[dict] = []

    async def call(arguments: dict) -> str:
        received.append(arguments)
        return f"{arguments['city']} x{arguments.get('days', 1)}"

    forecast = tool_from_schema(
        "forecast",
        "Weather for a city.",
        {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 1},
                "city": {"type": "string", "description": "City name."},
            },
            "required": ["city"],
        },
        call,
    )
    llm = ScriptedLLM([("", block("submit(await forecast('Austin', days=3))"))])
    g, root = agent_graph("weather")
    assert run_result(ScriptedFlow(llm, tools=[forecast]), g, root) == "Austin x3"
    assert received == [{"city": "Austin", "days": 3}]
    assert (
        "`await forecast(city: str, days: int = 1)`: Weather for a city. Arguments: `city`: City name."
        in prompt_text(llm)
    )


def test_a_toolkit_is_opened_for_the_run_and_closed_after_it():
    events: list[str] = []

    class Counter:
        async def open(self):
            events.append("open")

            def count() -> int:
                """Count calls."""
                events.append("count")
                return 7

            return [count]

        async def close(self):
            events.append("close")

    flow = ScriptedFlow(ScriptedLLM([("", block("submit(count())"))]), tools=[Counter()])
    g, root = agent_graph("count")
    assert run_result(flow, g, root) == 7
    assert events == ["open", "count", "close"]
    assert "count" not in flow.tools


def test_an_attached_image_reaches_the_model_with_the_code_output(tmp_path):
    picture = tmp_path / "frame.png"
    picture.write_bytes(b"\x89PNG fake")

    def look() -> str:
        """Look at the frame."""
        attach_image(picture)
        return "looked"

    llm = ScriptedLLM([("looked", block("submit('ok')")), ("look", block("print(look())"))])
    g, root = agent_graph("look")
    assert run_result(ScriptedFlow(llm, tools=[look]), g, root) == "ok"

    code = code_nodes(g, root)[0]
    assert code.data["images"] == [str(picture)]
    text, image = llm.calls[1][-1]["content"]
    assert text["text"].startswith("looked")
    assert image == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"\x89PNG fake").decode()},
    }


def test_anthropic_gets_image_parts_as_base64_blocks():
    content = [
        {"type": "text", "text": "out"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
    ]
    assert anthropic_content(content) == [
        {"type": "text", "text": "out"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
    ]


def test_child_config_output_schema_comes_back_parsed():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    parent = block(
        "child = launch_agent('count', config='counter')\nresult = await child\nprint(result['n'] + 1)"
    )
    llm = ScriptedLLM(
        [
            ("measure", parent),
            ("42", block("submit('42')")),
            ("count", block("submit({'n': 41})")),
        ]
    )
    g, root = agent_graph("measure it")
    flow = ScriptedFlow(llm, configs=[AgentConfig(name="counter", output_schema=schema)])
    assert run_result(flow, g, root) == "42"
    child = root.children()[0]
    assert root.config.output_schema is None
    assert child.config.output_schema == schema


def test_named_model_routes_to_another_client():
    default = ScriptedLLM([("", block("submit('wrong client')"))])
    fast = ScriptedLLM([("", block("submit('fast client')"))])
    g, root = agent_graph("who answers")
    flow = ScriptedFlow(default, llm_clients={"fast": fast})
    assert run_result(flow, g, root, root_config=AgentConfig(model="fast")) == "fast client"
    assert default.calls == []


def test_child_config_model_routes_children():
    default = ScriptedLLM(
        [
            ("delegate", block("child = launch_agent('do a', config='cheap')\nprint(await child)")),
            ("cheap result", block("submit('parent saw cheap result')")),
        ]
    )
    fast = ScriptedLLM([("do a", block("submit('cheap result')"))])
    flow = ScriptedFlow(
        default, llm_clients={"fast": fast}, configs=[AgentConfig(name="cheap", model="fast")]
    )

    g = Graph()
    root = g.start("delegate")

    assert run_result(flow, g, root) == "parent saw cheap result"
    assert root.children()[0].config.model == "fast"
    assert len(fast.calls) == 1


def test_child_launch_uses_the_flow_default_config():
    llm = ScriptedLLM(
        [
            ("delegate", block("child = launch_agent('do a')\nprint(await child)")),
            ("do a", block("submit('child')")),
            ("child", block("submit('done')")),
        ]
    )
    flow = ScriptedFlow(llm, configs=[AgentConfig(max_iters=7)])
    g, root = agent_graph("delegate")

    assert run_result(flow, g, root, root_config=AgentConfig(max_iters=5)) == "done"
    child = root.children()[0]
    assert root.config.max_iters == 5
    assert child.config_name == "default"
    assert child.config is flow.configs["default"]


def test_launch_model_keyword_is_an_ordinary_child_argument():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block("child = launch_agent('do a', model='input value')\nprint(await child)"),
            ),
            ("do a", block("submit(model)")),
            ("input value", block("submit('done')")),
        ]
    )
    g, root = agent_graph("delegate")

    assert run_result(ScriptedFlow(llm), g, root) == "done"
    assert root.children()[0].config.model is None


def test_long_child_query_is_refused():
    llm = ScriptedLLM(
        [
            ("delegate", block("print(launch_agent('far too long'))")),
            ("exceeds", block("submit('stopped')")),
        ]
    )
    g, root = agent_graph("delegate this")
    assert (
        run_result(ScriptedFlow(llm), g, root, root_config=AgentConfig(max_query_chars=4))
        == "stopped"
    )
    assert root.children() == []


def final_messages(g, agent):
    return [n for n in agent.nodes() if n.data.get("final")]


def test_a_nearly_spent_budget_asks_for_an_answer_instead_of_dropping_the_run():
    llm = ScriptedLLM(
        [
            ("last turn", block("submit('best guess')")),
            ("one", block("print('two')")),
            ("", block("print('one')")),
        ]
    )
    llm.last_usage = SimpleNamespace(input_tokens=4, output_tokens=4)
    g, root = agent_graph("spend most of it")

    assert (
        run_result(ScriptedFlow(llm), g, root, root_config=AgentConfig(max_budget=20))
        == "best guess"
    )
    [final] = final_messages(g, root)
    assert final.producer == "rlmflow"


def test_a_failed_final_turn_under_a_token_budget_ends_instead_of_looping():
    llm = ScriptedLLM(
        [
            ("last turn", block("y = 2")),
            ("one", block("print('two')")),
            ("", block("print('one')")),
        ]
    )
    llm.last_usage = SimpleNamespace(input_tokens=4, output_tokens=4)
    g, root = agent_graph("spend most of it")

    assert (
        run_result(ScriptedFlow(llm), g, root, root_config=AgentConfig(max_budget=20))
        == "[max_budget exceeded]"
    )
    assert len(final_messages(g, root)) == 1
    assert len(llm.calls) == 3


def test_the_budget_gate_never_discards_an_answer_already_paid_for():
    llm = ScriptedLLM([("", block("submit('landed')"))])
    llm.last_usage = SimpleNamespace(input_tokens=40, output_tokens=40)
    g, root = agent_graph("answer immediately")

    assert (
        run_result(ScriptedFlow(llm), g, root, root_config=AgentConfig(max_budget=10)) == "landed"
    )


def test_exhausted_budget_finalizes_children_before_the_parent():
    class ConcurrentBudgetLLM:
        def __init__(self):
            self.children_ready = asyncio.Barrier(2)

        async def stream(self, messages):
            query = first_user(messages)
            final = "last turn" in messages[-1]["content"]
            if final:
                if query == "parent":
                    assert any("a-final,b-final" in message["content"] for message in messages)
                    reply = block("submit('root-final')")
                else:
                    reply = block(f"submit({f'{query}-final'!r})")
                usage = LLMUsage(output_tokens=1)
            elif query == "parent":
                reply = block(
                    "handles = [launch_agent('a'), launch_agent('b')]\n"
                    "answers = await asyncio.gather(*handles)\n"
                    "print(','.join(answers))"
                )
                usage = LLMUsage(output_tokens=1)
            else:
                await self.children_ready.wait()
                reply = block(f"draft = {query!r}\nprint(draft)")
                usage = LLMUsage(output_tokens=8)
            yield LLMChunk(text=reply, usage=usage)

    g, root = agent_graph("parent")
    flow = ScriptedFlow(
        ConcurrentBudgetLLM(), configs=[AgentConfig(max_budget=10)], runtime=TestRuntime()
    )
    result = run_result(flow, g, root)

    assert result == "root-final"
    assert g.usage().total > root.config.max_budget
    assert len(root.children()) == 2
    for agent in (root, *root.children()):
        assert len(final_messages(g, agent)) == 1


def test_exhausted_budget_rejects_new_children():
    flow = ScriptedFlow(ScriptedLLM([]))
    g, root = agent_graph("parent")
    root.bind_config(AgentConfig(max_budget=1))
    turn = g.append(root, Node("assistant", data={"usage": {"output_tokens": 1}}))

    with pytest.raises(RuntimeError, match="max_budget"):
        agent_tools.spawn(flow, Step(g, root, turn), "too late")


def test_child_failure_reaches_the_parent():
    llm = ScriptedLLM(
        [
            ("delegate", block("child = launch_agent('boom')\nprint(await child)")),
            ("boom", RuntimeError("child is down")),
            ("child is down", block("submit('recovered')")),
        ]
    )
    g, root = agent_graph("delegate this")
    assert run_result(ScriptedFlow(llm), g, root) == "recovered"
    [child] = root.children()
    assert child.final_result() == "[child failed: RuntimeError: child is down]"


def test_aclose_drops_the_work_and_the_repls():
    class ClosingLLM(ScriptedLLM):
        closed = False

        async def aclose(self):
            self.closed = True

    llm = ClosingLLM([("count", block("submit('six')"))])
    flow = ScriptedFlow(llm)

    async def run_then_close():
        await flow.arun(agent_graph("count for me")[0])
        assert flow.runtime.repls
        await flow.aclose()

    asyncio.run(run_then_close())
    assert llm.closed
    assert not flow.runtime.repls
    assert not flow.tasks


def test_one_flow_rejects_competing_stream_drivers():
    llm = ScriptedLLM([("one", block("submit('first')")), ("two", block("submit('second')"))])
    flow = ScriptedFlow(llm)
    (first, _), (second, _) = agent_graph("one"), agent_graph("two")

    async def both():
        left = flow.run_streaming(first)
        right = flow.run_streaming(second)
        await anext(left)
        with pytest.raises(RuntimeError, match="already driving"):
            await anext(right)
        await left.aclose()
        await right.aclose()

    asyncio.run(both())


def peaked(delay=0.02):
    """A blocking reply, plus the most calls it ever saw running at once."""
    lock = threading.Lock()
    seen = {"active": 0, "peak": 0}

    def reply(messages):
        with lock:
            seen["active"] += 1
            seen["peak"] = max(seen["peak"], seen["active"])
        try:
            time.sleep(delay)
            return messages[-1]["content"].upper()
        finally:
            with lock:
                seen["active"] -= 1

    return reply, seen


def test_workers_bounds_blocking_model_calls():
    reply, seen = peaked()
    flow = ScriptedFlow(StubLLM(reply), workers=2)

    async def query_all():
        return await asyncio.gather(*(flow.llm_query(value) for value in ("a", "b", "c", "d")))

    assert asyncio.run(query_all()) == ["A", "B", "C", "D"]
    assert seen["peak"] == 2


def test_a_client_request_timeout_gives_up_on_a_slow_call():
    class SlowLLM:
        request_timeout = 0.01

        async def chat(self, messages):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        ScriptedFlow(SlowLLM()).run("hang")


def test_a_client_request_timeout_is_handed_to_clients_that_take_one():
    llm = StubLLM(lambda _messages: block("submit('ok')"))
    llm.request_timeout = 5
    assert ScriptedFlow(llm).run("go") == "ok"
    assert [call["timeout"] for call in llm.calls] == [5]

    # A client with no timeout parameter is called the way it was written.
    lean = ScriptedLLM([("", block("submit('ok')"))])
    lean.request_timeout = 5
    assert ScriptedFlow(lean).run("go") == "ok"
