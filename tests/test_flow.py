import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from helpers import (
    StubLLM,
    TestRuntime,
    agent_graph,
    first_user,
    next_node,
    run_result,
    stepping,
    trailing_user_text,
)
from helpers import reasoning_repl as block

from rlmflow import (
    DEPENDS_ON,
    FROM,
    NEXT,
    AgentConfig,
    AgentHandle,
    AgentStart,
    AppendChild,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    Flow,
    Input,
    LLMChunk,
    LLMOutput,
    LLMUsage,
    Node,
    PromptProfile,
    ReplDead,
    SequentialPool,
    StaleFrontier,
    UserQuery,
    VerificationFailure,
    reads,
)
from rlmflow.graph import Graph, Step, ops
from rlmflow.graph.persistence import node_record
from rlmflow.runtime import LocalRuntime
from rlmflow.tools import launch, tool
from rlmflow.tools.launch import split_args
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


def test_truncate_output_preserves_traceback_tail():
    output = "printed preview data\n" + "x" * 100 + "\nValueError: boxes has no items"

    truncated = truncate_output(output, 40)

    assert truncated.startswith("printed preview data")
    assert "truncated" in truncated
    assert truncated.endswith("boxes has no items")


def test_submit():
    llm = ScriptedLLM([("count", block("x = 2\nprint(x)")), ("2", block("submit(x * 3)"))])
    flow = Flow(llm)
    assert flow.run("count for me") == 6
    assert llm.calls[0][0]["role"] == "system"


def depends(g, agent):
    """``(as, producer id)`` for each dependency edge, in write order."""
    return tuple((e.data.get("as"), e.target) for e in g.edges(kind=DEPENDS_ON, source=agent.id))


def reload(g, path):
    """Save ``g`` to ``path`` and load it back as a cold graph."""
    g.save(path)
    return Graph.load(path)


def system_prompt(flow, task, **config):
    g, agent = agent_graph(task, **config)
    with stepping(g, agent):
        return flow.build_system_prompt(agent).render()


def test_spawn_capable_prompt_explains_delegation_economics():
    prompt = system_prompt(Flow(ScriptedLLM([])), "build an app", max_depth=1)

    assert "fresh context" in prompt
    assert "cheaper models" in prompt
    assert "run independent\n  work concurrently" in prompt
    assert "reduce token cost and wall-clock time" in prompt


def test_system_prompt_states_repl_action_and_both_ways_to_finish():
    prompt = system_prompt(Flow(ScriptedLLM([])), "write a poem")

    assert "To act, a response contains exactly one non-empty fenced ```repl``` block" in prompt
    assert "A response with no ```repl``` block is your final answer" in prompt
    assert "call `submit(value)` inside a block" in prompt
    assert "reply in plain text with no ```repl``` block" in prompt
    assert "Do not reply in plain text while work remains" in prompt
    assert "Avoid calling `submit(...)` in a block that prints output" in prompt
    assert "Printing and submitting together is allowed" in prompt
    assert "read the observation, then submit in the next turn" in prompt


def test_prose_before_repl_action_remains_in_following_turn_context():
    llm = ScriptedLLM(
        [
            ("inspect", "I need to inspect first.\n```repl\nprint('observed')\n```"),
            ("observed", block("finish('The final answer.')")),
        ]
    )

    assert Flow(llm).run("inspect this") == "The final answer."
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
        [("boom", block("1 / 0")), ("ZeroDivisionError", block("finish('recovered')"))]
    )
    g, root = agent_graph("boom")
    assert run_result(Flow(llm), g, root) == "recovered"
    types = [type(n).__name__ for n in g.nodes()]
    assert "ErrorOutput" in types


def test_execution_guard_rejects_code_before_it_reaches_the_runtime():
    seen = []

    async def guard(action):
        seen.append(action.code)
        if "forbidden" in action.code:
            return "rejected by execution guard"
        return None

    llm = ScriptedLLM(
        [
            ("start", block("forbidden()")),
            ("rejected by execution guard", block("finish('safe')")),
        ]
    )
    g, root = agent_graph("start")

    assert run_result(Flow(llm, execution_guard=guard), g, root) == "safe"
    assert seen == ["forbidden()", "finish('safe')"]
    assert any(
        isinstance(node, ErrorOutput) and node.content == "rejected by execution guard"
        for node in g.nodes()
    )


def test_max_iters():
    llm = ScriptedLLM(
        [
            ("last turn", block("finish('best effort')")),
            ("", block("print('again')")),
        ]
    )
    g, root = agent_graph("loop", max_iters=3)
    assert run_result(Flow(llm), g, root) == "best effort"
    assert sum(isinstance(n, LLMOutput) for n in g.nodes()) == 3


def test_iters_and_budget_are_unbounded_unless_set():
    llm = ScriptedLLM([("loop", block("print('again')")), ("again", block("finish('ok')"))])
    g, root = agent_graph("loop")
    assert root.config.max_iters is None
    assert root.config.max_budget is None
    assert root.config.max_output_length == 20_000
    assert root.config.max_query_chars == 20_000
    assert root.config.resume_on_child_done is True
    _, bounded = agent_graph("bounded", max_iters=30, max_budget=100_000)
    assert bounded.config.max_iters == 30
    assert bounded.config.max_budget == 100_000
    assert run_result(Flow(llm), g, root) == "ok"


def test_direct_arguments():
    llm = ScriptedLLM([("", block("finish(doc.upper())"))])
    assert Flow(llm).run("shout", doc="hi") == "HI"


def test_described_direct_argument_is_unwrapped_in_the_repl():
    llm = ScriptedLLM([("", block("finish(doc.upper())"))])

    result = Flow(llm).run(
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
    loaded = Graph.load(run).node(root.id)

    assert loaded.resolved_args() == {"document": document}
    assert loaded.argument_descriptions() == {"document": "A large source document."}
    assert len(list((run / "blobs").iterdir())) == 1


def test_run_task_string_carries_the_flow_defaults():
    flow = Flow(
        ScriptedLLM([("dig in", block("finish(task)"))]),
        root_config=AgentConfig(max_depth=1, max_iters=5, model="fast"),
        llm_clients={"fast": ScriptedLLM([("dig in", block("finish('fast ' + task)"))])},
    )

    assert flow.run("dig in") == "fast dig in"


def test_graph_create_agent_stores_json_schema_without_a_flow(tmp_path):
    from pydantic import BaseModel

    class Label(BaseModel):
        label: str

    g = Graph()
    agent = g.create_agent("classify", config=AgentConfig(output_schema=Label), text="hi")

    assert agent.args == {"text": "hi"}
    assert agent.config.output_schema["properties"]["label"]["type"] == "string"
    loaded = reload(g, tmp_path / "run")
    assert loaded.node(agent.id).config.output_schema == agent.config.output_schema


def test_graph_edited_without_a_flow_runs_on_a_fresh_flow(tmp_path):
    """Spawn, append history, and fork with ops alone; a new Flow builds every REPL."""
    g = Graph()
    a = g.create_agent("count", config=AgentConfig(name="a"))
    ops.append(g, a, ExecAction(code="n = 41"))
    observed = ops.append(g, a, ExecOutput(content=""))
    b = ops.fork(g, observed, name="b")
    ops.send(g, a, "finish a")
    ops.send(g, b, "finish b")

    loaded = reload(g, tmp_path / "run")
    flow = Flow(
        ScriptedLLM(
            [
                ("finish a", block("finish(n + 1)")),
                ("finish b", block("finish(n + 2)")),
            ]
        )
    )
    flow.run(loaded)

    assert reads.result(loaded, loaded.node(a.id)) == 42
    assert reads.result(loaded, loaded.node(b.id)) == 43


def test_agent_added_mid_stream_is_scheduled():
    llm = ScriptedLLM([("first", block("finish('one')")), ("second", block("finish('two')"))])
    flow = Flow(llm)
    g = Graph()
    first = g.create_agent("first")
    added = []

    async def drive():
        async for _item in flow.run_streaming(g):
            if not added:
                added.append(g.create_agent("second"))

    asyncio.run(drive())

    assert reads.result(g, first) == "one"
    assert reads.result(g, added[0]) == "two"


def test_keep_n_messages_keeps_the_tail_and_truncation_summary():
    from rlmflow import TruncationSummary

    llm = ScriptedLLM(
        [
            ("last turn", block("finish('best effort')")),
            ("", block("print('again')")),
        ]
    )
    g, root = agent_graph("loop", max_iters=6, keep_n_messages=3)
    flow = Flow(llm)
    assert run_result(flow, g, root) == "best effort"
    assert any(isinstance(node, TruncationSummary) for node in reads.history(g, root))

    with stepping(g, root):
        messages = flow.build_messages(reads.frontier(g, root))
    assert messages[0]["role"] == "system"
    assert any("again" in message["content"] for message in messages)
    assert any(TruncationSummary().content in message["content"] for message in messages)


async def stream(flow, g, until):
    """The nodes a stream yields; edges are left out."""
    return [item async for item in flow.run_streaming(g, until=until) if isinstance(item, Node)]


def test_streaming_until_idle():
    llm = ScriptedLLM([("look", block("print('peek')")), ("peek", block("finish('finished')"))])
    flow = Flow(llm)
    g, root = agent_graph("look")
    seen = asyncio.run(stream(flow, g, "idle"))
    assert isinstance(seen[-1], ExecOutput) and seen[-1].content.strip() == "peek"
    assert not reads.terminal(g, root)

    rest = asyncio.run(stream(flow, g, "done"))
    assert isinstance(rest[-1], DoneOutput) and reads.result(g, root) == "finished"


def test_children_run_in_parallel():
    parent = block(
        "handles = [launch_subagent(task='do a'), "
        "launch_subagent(task='do b')]\n"
        "results = await asyncio.gather(*handles)\n"
        "print(results)"
    )
    llm = ScriptedLLM(
        [
            ("split", parent),
            ("do a", block("finish('A')")),
            ("do b", block("finish('B')")),
            ("['A', 'B']", block("finish('AB')")),
        ]
    )
    g, root = agent_graph("split this up")
    flow = Flow(llm)
    assert run_result(flow, g, root) == "AB"
    assert [child.config.name for child in reads.children(g, root)] == ["child0", "child1"]
    assert [reads.result(g, child) for child in reads.children(g, root)] == ["A", "B"]

    own = {node.id for node in reads.history(g, root)}
    assert all(node.agent_id == root.id for node in reads.history(g, root))
    child = reads.children(g, root)[0]
    assert own.isdisjoint(node.id for node in reads.history(g, child))
    assert reads.history(g, child)[0] is child
    assert all(node.agent_id == child.id for node in reads.history(g, child))

    # Children are launched by the action, so the parent's frontier stayed on that
    # action for the whole step, and its sequel is the step's output.
    [action] = (
        n for n in reads.history(g, root) if isinstance(n, ExecAction) and n.launch_agent_ids()
    )
    assert not isinstance(action, AppendChild)
    assert [g.node(e.source) for e in g.edges(kind=FROM, target=action.id)] == reads.children(
        g, root
    )
    assert isinstance(next_node(g, action), ExecOutput)
    assert next_node(g, reads.frontier(g, root)) is None

    streamed = asyncio.run(stream(Flow(llm), agent_graph("split this up")[0], "done"))
    assert sum(isinstance(n, AgentStart) for n in streamed) == 2


def test_handle_valued_args_build_a_flat_dependency_graph(tmp_path):
    parent = block(
        "parser = launch_subagent(task='parse-source')\n"
        "fixtures = launch_subagent(task='inspect-fixtures')\n"
        "reconciler = launch_subagent(\n"
        "    task='reconcile-inputs',\n"
        "    parsed=Input(parser, description='Parsed source counts.'),\n"
        "    fixtures=fixtures,\n"
        ")\n"
        "critic = launch_subagent(\n"
        "    task='critique-answer', reconciler=reconciler)\n"
        "print(await critic)"
    )
    llm = ScriptedLLM(
        [
            ("build-dag", parent),
            ("parse-source", block("finish({'count': 2})")),
            ("inspect-fixtures", block("finish('F')")),
            (
                "reconcile-inputs",
                block("finish(f\"{parsed['count']}:{fixtures}\")"),
            ),
            ("critique-answer", block("finish(reconciler + '!')")),
            ("2:F!", block("finish('done')")),
        ]
    )
    g, root = agent_graph("build-dag", max_depth=1)

    assert run_result(Flow(llm), g, root) == "done"
    assert [child.config.name for child in reads.children(g, root)] == [
        "child0",
        "child1",
        "child2",
        "child3",
    ]
    parser, fixtures, reconciler, critic = reads.children(g, root)
    assert {reads.depth(g, agent) for agent in reads.children(g, root)} == {1}
    assert depends(g, reconciler) == (("parsed", parser.id), ("fixtures", fixtures.id))
    assert depends(g, critic) == (("reconciler", reconciler.id),)
    assert reconciler.args == {}
    assert reads.inputs(g, reconciler) == {"parsed": {"count": 2}, "fixtures": "F"}
    [described] = g.edges(kind=DEPENDS_ON, source=reconciler.id, target=parser.id)
    assert described.data["description"] == "Parsed source counts."
    assert reads.inputs(g, critic) == {"reconciler": "2:F"}

    tasks = [first_user(messages) for messages in llm.calls]
    assert tasks.index("reconcile-inputs") > tasks.index("parse-source")
    assert tasks.index("reconcile-inputs") > tasks.index("inspect-fixtures")
    assert tasks.index("critique-answer") > tasks.index("reconcile-inputs")

    g.save(tmp_path / "dependency-run")
    loaded = Graph.load(tmp_path / "dependency-run")
    assert depends(loaded, loaded.node(reconciler.id)) == depends(g, reconciler)
    assert depends(loaded, loaded.node(critic.id)) == depends(g, critic)


def test_shared_source_feeds_multiple_flat_consumers():
    parent = block(
        "source = launch_subagent(task='shared-source')\n"
        "solver = launch_subagent(task='solve-source', source=source)\n"
        "reviewer = launch_subagent(task='review-source', source=source)\n"
        "print(await asyncio.gather(solver, reviewer))"
    )
    llm = ScriptedLLM(
        [
            ("fan-out", parent),
            ("shared-source", block("finish('evidence')")),
            ("solve-source", block("finish(source + ':solved')")),
            ("review-source", block("finish(source + ':reviewed')")),
            (
                "['evidence:solved', 'evidence:reviewed']",
                block("finish('done')"),
            ),
        ]
    )
    g, root = agent_graph("fan-out", max_depth=2)

    assert run_result(Flow(llm), g, root) == "done"
    assert [agent.config.name for agent in reads.children(g, root)] == [
        "child0",
        "child1",
        "child2",
    ]
    source, solver, reviewer = reads.children(g, root)
    assert depends(g, solver) == (("source", source.id),)
    assert depends(g, reviewer) == (("source", source.id),)
    action = next(
        node
        for node in reads.history(g, root)
        if isinstance(node, ExecAction) and node.launch_agent_ids()
    )
    assert set(action.launch_agent_ids()) == {
        source.id,
        solver.id,
        reviewer.id,
    }


def _launch_step(**config):
    graph = Graph()
    root = ops.spawn(graph, "root", config=AgentConfig(**config))
    action = ops.append(graph, root, ExecAction(code="launch"))
    return Step(graph, root, action)


def test_launch_args_accept_mixed_strings_and_top_level_handles():
    step = _launch_step(max_depth=1)
    source = ops.spawn(step.graph, "source", by=step.node, config=step.agent.config.child("source"))
    handle = AgentHandle(source.id, "source", "root.source")

    plain, inputs = split_args(step, {"prompt": "immediate", "evidence": handle})
    assert plain == {"prompt": "immediate"}
    assert inputs == {"evidence": source}


@pytest.mark.parametrize(
    "args",
    [
        {"nested": {"handle": "not-an-agent-ref"}},
        {"nested": ["not", "top-level"]},
        {"number": 3},
    ],
)
def test_launch_args_accept_nested_json_values(args):
    assert split_args(_launch_step(), args) == (args, {})


@pytest.mark.parametrize("value", [{1, 2}, object(), {1: "bad key"}])
def test_launch_args_reject_non_json_values(value):
    with pytest.raises(TypeError, match="JSON-compatible|non-string"):
        split_args(_launch_step(), {"value": value})


@pytest.mark.parametrize("name", ["bad-name", "class", "task"])
def test_launch_args_require_python_variable_names(name):
    with pytest.raises(ValueError, match="invalid invocation|'task' is the invocation"):
        split_args(_launch_step(), {name: "value"})


def test_launch_args_reject_unknown_and_nested_handles():
    step = _launch_step(max_depth=2)
    source = ops.spawn(step.graph, "source", by=step.node, config=step.agent.config.child("source"))
    source_handle = AgentHandle(source.id, "source", "root.source")

    with pytest.raises(ValueError, match="unknown agent"):
        split_args(step, {"missing": AgentHandle("missing", "missing", "root.missing")})
    with pytest.raises(TypeError, match="nests a handle"):
        split_args(step, {"nested": [source_handle]})


def test_dependency_scheduler_resumes_a_saved_blocked_agent(tmp_path):
    parent = block(
        "source = launch_subagent(task='produce-source')\n"
        "consumer = launch_subagent(task='consume-source', source=source)\n"
        "print(await consumer)"
    )
    llm = ScriptedLLM(
        [
            ("resume-dag", parent),
            ("produce-source", block("finish('A')")),
            ("consume-source", block("finish(source + '!')")),
            ("A!", block("finish('resumed')")),
        ]
    )
    g, root = agent_graph("resume-dag", max_depth=2)
    flow = Flow(llm)

    def source_done(item, graph):
        return isinstance(item, DoneOutput) and graph.node(item.agent_id).config.name == "child0"

    stopped = asyncio.run(stream(flow, g, source_done))
    assert g.node(stopped[-1].agent_id).config.name == "child0"
    assert [child.config.name for child in reads.children(g, root)] == ["child0", "child1"]
    source, consumer = reads.children(g, root)
    assert depends(g, consumer) == (("source", source.id),)
    assert consumer in reads.ready(g)
    assert reads.frontier(g, consumer) is consumer

    g.save(tmp_path / "blocked-run")
    restored = Graph.load(tmp_path / "blocked-run")
    assert run_result(Flow(llm), restored, restored.node(root.id)) == "resumed"
    assert reads.inputs(restored, restored.node(consumer.id)) == {"source": "A"}


def test_plan_query_allows_iterative_investigation_before_delegation():
    launch = block(
        "handles = [launch_subagent(task='alpha-workstream'), "
        "launch_subagent(task='beta-workstream')]\n"
        "print(await asyncio.gather(*handles))",
    )
    llm = ScriptedLLM(
        [
            (
                "build the requested system",
                block("print(brief[:7])"),
            ),
            (
                "preview",
                block("requirements = brief.splitlines()[1:]\nprint(requirements)"),
            ),
            ("requirement A", launch),
            ("alpha-workstream", block("finish('A')")),
            ("beta-workstream", block("finish('B')")),
            ("['A', 'B']", block("finish('done')")),
        ]
    )
    g, root = agent_graph(
        "build the requested system",
        args={"brief": "preview\nrequirement A\nrequirement B"},
        max_depth=1,
    )

    assert run_result(Flow(llm), g, root) == "done"
    assert sum(isinstance(node, LLMOutput) for node in reads.history(g, root)) == 4
    assert [child.config.name for child in reads.children(g, root)] == ["child0", "child1"]
    parent_outputs = [
        node.content.strip() for node in reads.history(g, root) if type(node) is ExecOutput
    ]
    assert parent_outputs[:2] == ["preview", "['requirement A', 'requirement B']"]


def test_save_writes_a_run_directory(tmp_path):
    parent = block("child = launch_subagent(task='do a')\nprint(await child)")
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("finish('A')")), ("A", block("finish('AB')"))]
    )
    g, root = agent_graph("split this up", args={"doc": "hi"})
    assert run_result(Flow(llm), g, root) == "AB"

    run = g.save(tmp_path / "run")
    saved = json.loads((run / "graph.json").read_text())
    assert set(saved) == {"id", "items"}
    assert saved["id"] == g.id
    items = saved["items"]
    assert [item["id"] for item in items] == [item.id for item in g.items()]
    assert items[0]["payload"]["args"] == {"doc": "hi"}

    action = next(
        n for n in reads.history(g, root) if isinstance(n, ExecAction) and n.launch_agent_ids()
    )
    child = reads.children(g, root)[0]
    edges = [item for item in items if "kind" in item]
    assert {"kind": FROM, "source": child.id, "target": action.id} in [
        {key: edge[key] for key in ("kind", "source", "target")} for edge in edges
    ]
    child_nodes = [item for item in items if item.get("agent_id") == child.id]
    assert [item["id"] for item in child_nodes] == [node.id for node in reads.history(g, child)]
    assert child_nodes[0]["payload"]["content"] == "do a"


def test_run_records_prompts_seq_and_timing():
    parent = block(
        "handles = [launch_subagent(task='do a'), "
        "launch_subagent(task='do b')]\n"
        "print(await asyncio.gather(*handles))"
    )
    llm = ScriptedLLM(
        [
            ("split", parent),
            ("do a", block("finish('A')")),
            ("do b", block("finish('B')")),
            ("['A', 'B']", block("finish('AB')")),
        ]
    )
    g, root = agent_graph("split this up")
    flow = Flow(llm)
    assert run_result(flow, g, root) == "AB"

    # seq is the graph's log position, so each history is strictly increasing and
    # the log order is the whole run, however the two children raced.
    for agent in (root, *reads.children(g, root)):
        seqs = [node.seq for node in reads.history(g, agent)]
        assert seqs == sorted(set(seqs))
    assert [item.seq for item in g.items()] == list(range(len(g.items())))

    # Only executed nodes are timed; a node written inside a step is not a step.
    stepped = [node for node in g.nodes() if isinstance(node, (LLMOutput, ExecOutput))]
    assert all(node.timing()["duration_ms"] >= 0 for node in stepped)
    written = [node for node in g.nodes() if isinstance(node, AgentStart)]
    assert written and all(node.timing() == {} for node in written)

    turns = [node for node in reads.history(g, root) if isinstance(node, LLMOutput)]
    assert root.system_prompt_for(turns[0])
    # The text is stored once per distinct prompt, and turns only keep the id.
    assert sorted(root.system_prompts) == sorted({turn.prompt_id for turn in turns})
    assert node_record(turns[0])["metadata"]["system_prompt"] == turns[0].prompt_id
    assert turns[0].timing()["duration_ms"] >= 0


def test_appending_off_the_frontier_is_refused():
    llm = ScriptedLLM([("count", block("x = 2\nprint(x)")), ("2", block("finish(x * 3)"))])
    g, root = agent_graph("count for me")
    assert run_result(Flow(llm), g, root) == 6

    stale = reads.history(g, root)[1]
    with pytest.raises(StaleFrontier, match="frontier"):
        ops.append(g, root, UserQuery(content="too late"), after=stale)
    assert len(g.edges(kind=NEXT, source=stale.id)) == 1


def test_structured_output_is_stored_parsed():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM([("count", block("submit({'n': 41})"))])
    g, root = agent_graph("count", output_schema=schema)
    assert run_result(Flow(llm), g, root) == {"n": 41}
    assert node_record(reads.frontier(g, root))["payload"]["result"] == {"n": 41}


def test_plain_text_json_reply_is_validated_against_the_schema():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM([("count", '{"n": 41}')])
    g, root = agent_graph("count", output_schema=schema)

    assert run_result(Flow(llm), g, root) == {"n": 41}
    assert not g.nodes(ExecAction)


def test_plain_text_reply_that_fails_the_schema_is_repaired():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM(
        [
            ("count", "forty-one"),
            ("was rejected", block("submit({'n': 41})")),
        ]
    )
    g, root = agent_graph("count", output_schema=schema)

    assert run_result(Flow(llm), g, root) == {"n": 41}
    [failure] = g.nodes(VerificationFailure)
    assert (failure.kind, failure.candidate) == ("schema", "'forty-one'")


def test_unstructured_submission_keeps_json_values():
    expected = {"n": [41, True, None, "é"]}
    llm = ScriptedLLM([("value", block(f"submit({expected!r})"))])
    g, root = agent_graph("return value")

    assert run_result(Flow(llm), g, root) == expected
    assert node_record(reads.frontier(g, root))["payload"]["result"] == expected


def test_unstructured_submission_rejects_non_json_values():
    llm = ScriptedLLM(
        [
            ("return value", block("submit((1, 2))")),
            ("requires a JSON value", block("submit('recovered')")),
        ]
    )
    g, root = agent_graph("return value")

    assert run_result(Flow(llm), g, root) == "recovered"
    assert any(
        "requires a JSON value" in node.content
        for node in reads.history(g, root)
        if isinstance(node, VerificationFailure)
    )


def test_json_submission_crosses_worker_and_persistence_boundary(tmp_path):
    g, root = agent_graph("return value")
    flow = Flow(
        ScriptedLLM([("return value", block("submit({'n': [1, True, None]})"))]),
        runtime=LocalRuntime(repl_timeout=10),
    )

    assert run_result(flow, g, root, close_repls=True) == {"n": [1, True, None]}
    loaded = reload(g, tmp_path / "json-submission")
    assert reads.result(loaded, loaded.node(root.id)) == {"n": [1, True, None]}


def test_worker_submission_rejects_non_json_values_and_recovers():
    llm = ScriptedLLM(
        [
            ("return value", block("submit((1, 2))")),
            ("requires a JSON value", block("submit({'ok': True})")),
        ]
    )
    flow = Flow(llm, runtime=LocalRuntime(repl_timeout=10))

    g, root = agent_graph("return value")
    assert run_result(flow, g, root, close_repls=True) == {"ok": True}


def test_plain_text_reply_finishes_the_agent():
    llm = ScriptedLLM([("poem", "The moon writes silver on the sea.")])
    g, root = agent_graph("write a poem")

    assert run_result(Flow(llm), g, root) == "The moon writes silver on the sea."
    assert not g.nodes(ExecAction)
    assert not g.nodes(ErrorOutput)


def test_plain_text_reply_may_contain_non_repl_code_fences():
    answer = "Use a list comprehension:\n```python\nprint([x * 2 for x in xs])\n```"
    llm = ScriptedLLM([("explain", answer)])
    g, root = agent_graph("explain doubling")

    assert run_result(Flow(llm), g, root) == answer
    assert not g.nodes(ExecAction)


def protocol_failure(g):
    [action] = [node for node in g.nodes(ExecAction) if "protocol_error" in node.metadata]
    [error] = [node for node in g.nodes(ErrorOutput) if reads.prev(g, node) is action]
    return action, error


@pytest.mark.parametrize(
    ("invalid_reply", "reason"),
    [
        ("```repl\n```", "the ```repl block is empty"),
        ("```repl\nunclosed = True", "the ```repl block was never closed"),
        ("", "the reply is empty"),
        ("```repl\nmarker = 'first'\n```\n```repl\nmarker = 'second'\n```", "found 2 ```repl"),
    ],
)
def test_invalid_reply_becomes_a_raising_action_and_recovers(invalid_reply, reason):
    llm = ScriptedLLM(
        [
            ("bad response", invalid_reply),
            (
                "MissingReplError",
                block("submit('dirty' if {'unclosed', 'marker'} & set(globals()) else 'clean')"),
            ),
        ]
    )
    g, root = agent_graph("bad response")

    assert run_result(Flow(llm), g, root) == "clean"
    action, error = protocol_failure(g)
    assert action.code.startswith("raise type('MissingReplError', (Exception,), {})(")
    assert action.metadata["protocol_error"].startswith(reason)
    assert error.content.startswith(f"MissingReplError: {reason}")
    assert "Nothing ran" in error.content
    assert "reply in plain text with no ```repl block" in error.content


def test_protocol_error_action_survives_save_load_and_replay(tmp_path):
    llm = ScriptedLLM(
        [
            ("count", "```repl\n```"),
            ("MissingReplError", block("secret = 41\nprint('stored')")),
        ]
    )
    g, root = agent_graph("count for me")
    asyncio.run(stream(Flow(llm), g, lambda item, _g: isinstance(item, ExecOutput)))
    loaded = reload(g, tmp_path / "protocol-error")
    action, _error = protocol_failure(loaded)
    assert action.metadata["protocol_error"] == "the ```repl block is empty"

    resumed = ScriptedLLM([("stored", block("finish(secret + 1)"))])
    assert run_result(Flow(resumed), loaded, loaded.node(root.id)) == 42
    assert len(loaded.nodes(ErrorOutput)) == 1


def test_plain_reply_on_final_turn_finishes_the_run():
    g, root = agent_graph("answer now", max_iters=2)

    llm = ScriptedLLM([("last turn", "best supported answer"), ("answer now", block("x = 1"))])
    assert run_result(Flow(llm), g, root) == "best supported answer"


def test_final_action_without_submit_is_an_error_not_a_crash():
    g, root = agent_graph("answer now", max_iters=2)

    llm = ScriptedLLM([("last turn", block("y = 2")), ("answer now", block("x = 1"))])
    assert run_result(Flow(llm), g, root) == "[max_iters exceeded]"
    [error] = [node for node in g.nodes(ErrorOutput) if "MissingSubmitError" in node.content]
    assert error.content.endswith("MissingSubmitError: the final turn must call submit(answer)")


def test_save_load_roundtrip(tmp_path):
    parent = block("child = launch_subagent(task='do a')\nprint(await child)")
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("finish('A')")), ("A", block("finish('AB')"))]
    )
    g, root = agent_graph("split this up", args={"doc": "hi"})
    run_result(Flow(llm), g, root)

    loaded = reload(g, tmp_path / "run")
    loaded_root = loaded.node(root.id)
    assert [item.id for item in loaded.items()] == [item.id for item in g.items()]
    assert (reads.result(loaded, loaded_root), reads.terminal(loaded, loaded_root)) == ("AB", True)
    [child] = reads.children(loaded, loaded_root)
    assert reads.path(loaded, child) == "root.child0"
    assert reads.result(loaded, child) == "A"
    assert loaded.save(tmp_path / "again") and (tmp_path / "again" / "graph.json").is_file()


def paused_run(tmp_path, name="run"):
    """A saved run that set a variable and stopped before using it."""
    llm = ScriptedLLM([("count", block("secret = 41\nprint('stored')"))])
    g, root = agent_graph("count for me")
    asyncio.run(stream(Flow(llm), g, "idle"))
    assert isinstance(reads.frontier(g, root), ExecOutput)
    loaded = reload(g, tmp_path / name)
    return loaded, loaded.node(root.id)


def test_resuming_a_saved_run_replays_its_repl(tmp_path):
    loaded, root = paused_run(tmp_path)
    llm = ScriptedLLM([("stored", block("finish(secret + 1)"))])
    assert run_result(Flow(llm), loaded, root) == 42
    assert not loaded.nodes(ErrorOutput)
    assert len(llm.calls) == 1  # the replay itself asks the model nothing


def test_lazy_restore_waits_until_code_needs_the_repl(tmp_path):
    loaded, root = paused_run(tmp_path)
    llm = ScriptedLLM([("stored", block("finish(secret + 1)"))])
    flow = Flow(llm, restore="lazy")

    async def stop_before_execution():
        async for _ in flow.run_streaming(
            loaded,
            until=lambda item, _graph: isinstance(item, LLMOutput),
        ):
            pass

    asyncio.run(stop_before_execution())
    assert flow.runtime.get(root) is None
    assert run_result(flow, loaded, root) == 42
    assert flow.runtime.get_var(root, "secret") == 41


def test_halting_mid_delegation_resumes_without_relaunching():
    def reply(messages):
        task = first_user(messages)
        if task == "parent":
            return block(
                "seed = 40\n"
                "slow = launch_agent(task='slow', name='slow')\n"
                "fast = launch_agent(task='fast', name='fast')\n"
                "results = await asyncio.gather(slow, fast)\n"
                "finish(f'{seed}:' + ''.join(results))"
            )
        if task == "slow":
            return block("await asyncio.sleep(0.2)\nfinish('S')")
        return block("finish('F')")

    g, root = agent_graph("parent", max_depth=1)
    flow = Flow(StubLLM(reply), runtime=LocalRuntime(repl_timeout=10))

    async def halt_at_first_child():
        async for _ in flow.run_streaming(
            g, until=lambda item, _g: isinstance(item, AgentStart) and item is not root
        ):
            pass

    asyncio.run(halt_at_first_child())
    assert run_result(flow, g, root) == "40:SF"
    assert sorted(child.config.name for child in reads.children(g, root)) == ["fast", "slow"]
    flow.runtime.close_repls()


def colocated_child(*, parent_done: bool, **config):
    """A parent and a worker-sharing child, each with recorded code."""
    g, root = agent_graph("parent", max_depth=1, **config)
    ops.append(g, root, ExecAction(code="shared = ['parent']"))
    action = reads.frontier(g, root)
    child = ops.spawn(g, "child", by=action, config=root.config.child("child", reuse_worker=True))
    if parent_done:
        ops.append(g, root, DoneOutput(result="parent done"))
    else:
        ops.append(g, root, ExecOutput(content="child done"))
    ops.append(g, child, ExecAction(code="child_state = ['child']"))
    return g, root, child


@pytest.mark.parametrize("restore_mode", ["replay", "lazy"])
def test_restore_rebuilds_terminal_worker_owner_for_unfinished_colocated_child(
    tmp_path,
    restore_mode,
):
    """Cold graph whose unfinished child still needs its terminal parent's worker."""
    g, root, child = colocated_child(parent_done=True, resume_on_child_done=False)
    ops.append(g, child, ExecOutput(content="stored"))
    loaded = reload(g, tmp_path / restore_mode)
    flow = Flow(
        ScriptedLLM([("stored", block("finish(task)"))]),
        runtime=LocalRuntime(repl_timeout=10),
        restore=restore_mode,
    )

    assert run_result(flow, loaded, loaded.node(root.id), close_repls=True) == "parent done"
    assert reads.result(loaded, loaded.node(child.id)) == "child"


def test_replay_keeps_colocated_agent_namespaces_isolated(
    tmp_path,
):
    g, root, child = colocated_child(parent_done=False)
    ops.append(g, child, DoneOutput(result="child done"))
    loaded = reload(g, tmp_path / "terminal-shared-child")

    flow = Flow(
        ScriptedLLM([("child done", block("finish(shared)"))]),
        runtime=LocalRuntime(repl_timeout=10),
    )

    assert run_result(flow, loaded, loaded.node(root.id), close_repls=True) == ["parent"]


def test_replay_reads_recorded_child_answers_instead_of_relaunching(tmp_path):
    parent = block("child = launch_subagent(task='add up')\nanswer = await child\nprint(answer)")
    llm = ScriptedLLM([("delegate", parent), ("add up", block("finish('20')"))])
    g, root = agent_graph("delegate", max_depth=2)

    def parent_turn_ended(item, _graph):
        return type(item) is ExecOutput and item.agent_id == root.id

    asyncio.run(stream(Flow(llm), g, parent_turn_ended))
    loaded = reload(g, tmp_path / "run")
    loaded_root = loaded.node(root.id)

    resumed = ScriptedLLM([("20", block("finish('parent saw ' + answer)"))])
    assert run_result(Flow(resumed), loaded, loaded_root) == "parent saw 20"
    # The child ran once, when it was recorded: no second launch, no second model call.
    assert [child.config.name for child in reads.children(loaded, loaded_root)] == ["child0"]
    assert len(resumed.calls) == 1


def test_replay_restores_llm_query_result_without_another_model_call(tmp_path):
    setup = ScriptedLLM(
        [
            ("delegate", block("answer = await llm_query('extract')\nprint(answer)")),
            ("extract", "recorded"),
        ]
    )
    g, root = agent_graph("delegate")

    def query_finished(item, _graph):
        return type(item) is ExecOutput

    setup_flow = Flow(
        setup,
        runtime=LocalRuntime(repl_timeout=10),
        use_llm_query=True,
    )
    asyncio.run(stream(setup_flow, g, query_finished))
    asyncio.run(setup_flow.aclose())
    loaded = reload(g, tmp_path / "llm-query-run")
    resumed = ScriptedLLM([("recorded", block("finish(answer)"))])

    flow = Flow(resumed, runtime=LocalRuntime(repl_timeout=10), use_llm_query=True)
    assert run_result(flow, loaded, loaded.node(root.id), close_repls=True) == "recorded"
    assert len(resumed.calls) == 1


def test_append_child_runs_a_prebuilt_subtree():
    g, root = agent_graph("parent", max_depth=1)
    branch = ops.spawn(g, "child")
    action = ops.attach(g, root, branch, names=["branch0"])

    flow = Flow(
        ScriptedLLM(
            [
                ("child", block("finish('finished')")),
                ("finished", block("finish('finished')")),
            ]
        )
    )

    assert run_result(flow, g, root) == "finished"
    assert isinstance(action, AppendChild)
    assert reads.children(g, root) == [branch]
    assert reads.path(g, branch) == "root.branch0"


def test_controller_attaches_multiple_subtrees_between_streams():
    llm = ScriptedLLM(
        [
            ("['A', 'B']", block("finish('AB')")),
            ("prepare", block("print('prepared')")),
            ("child a", block("finish(value)")),
            ("child b", block("finish(value)")),
        ]
    )
    flow = Flow(llm)
    g, root = agent_graph("prepare", max_depth=1)

    def prepared(task, code):
        """A subtree built in its own graph, as a controller would prepare it."""
        side = Graph()
        agent = ops.spawn(side, task)
        ops.append(side, agent, ExecAction(code=code))
        ops.append(side, agent, ExecOutput(content=task))
        g.merge(side)
        return agent

    async def main():
        async for _item in flow.run_streaming(g, until="idle"):
            pass

        children = [prepared("child a", "value = 'A'"), prepared("child b", "value = 'B'")]
        ops.attach(g, root, *children, names=["a", "b"])

        async for _item in flow.run_streaming(g):
            pass

    asyncio.run(main())

    action = next(node for node in reads.history(g, root) if isinstance(node, AppendChild))
    assert reads.result(g, root) == "AB"
    assert [child.config.name for child in reads.children(g, root)] == ["a", "b"]
    assert action.code == (
        "_handles = [\n"
        "    launch_agent(task=''),\n"
        "    launch_agent(task=''),\n"
        "]\n"
        "print(await asyncio.gather(*_handles))"
    )


def test_resume_uses_an_unfinished_subtree_already_on_the_launch_action():
    g, root = agent_graph("parent", max_depth=1)
    child = ops.spawn(g, "child")
    ops.attach(g, root, child, names=["kid"])

    llm = ScriptedLLM(
        [
            ("finished child", block("finish('parent saw finished child')")),
            ("child", block("finish('finished child')")),
        ]
    )
    assert run_result(Flow(llm), g, root) == "parent saw finished child"
    assert reads.result(g, child) == "finished child"
    assert len(llm.calls) == 2


def test_append_child_type_survives_save_load(tmp_path):
    g, root = agent_graph("parent")
    branch = ops.spawn(g, "child")
    action = ops.attach(g, root, branch, names=["branch0"])
    ops.append(g, branch, DoneOutput(result="finished"))
    ops.append(g, root, ExecOutput(content="['finished']"))

    loaded = reload(g, tmp_path / "run")
    [loaded_action] = loaded.nodes(AppendChild)

    assert loaded_action.code == action.code
    assert loaded_action.launch_agent_ids() == [branch.id]
    assert [c.config.name for c in reads.children(loaded, loaded.node(root.id))] == ["branch0"]


def test_replaying_agents_can_skip_their_own_unrepeatable_work(tmp_path):
    trace: list[str] = []

    @tool("Record a live side effect.", proxy=True)
    def log(value):
        trace.append(value)

    code = block("if ENV['RLMFLOW_REPLAY'] == '0':\n    log('live')\nprint('set up')")
    flow = Flow(ScriptedLLM([("count", code)]))
    flow.inject("log", log)
    g, root = agent_graph("count for me")
    asyncio.run(stream(flow, g, "idle"))
    assert trace == ["live"]

    loaded = reload(g, tmp_path / "run")
    resumed = Flow(ScriptedLLM([("set up", block("finish('ok')"))]))
    resumed.inject("log", log)
    assert run_result(resumed, loaded, loaded.node(root.id)) == "ok"
    assert trace == ["live"]  # the replay saw RLMFLOW_REPLAY=1 and skipped it


def test_a_finished_run_resumes_to_its_recorded_answer(tmp_path):
    llm = ScriptedLLM([("count", block("finish('six')"))])
    g, root = agent_graph("count for me")
    assert run_result(Flow(llm), g, root) == "six"

    loaded = reload(g, tmp_path / "run")
    idle = ScriptedLLM([])
    assert run_result(Flow(idle), loaded, loaded.node(root.id)) == "six"
    assert idle.calls == []  # nothing left to step


def test_a_dead_repl_is_an_observation_and_is_replaced():
    class OneDeadRepl(LocalRuntime):
        """Whatever the first agent gets, its REPL dies the way a sandbox does."""

        opened = 0

        def open(self, agent, graph=None):
            self.opened += 1
            repl = super().open(agent, graph)
            if self.opened == 1:

                async def die(code):
                    raise ConnectionResetError("sandbox went away")

                repl.run = die
            return repl

    llm = ScriptedLLM(
        [
            ("count", block("x = 1\nprint('recovery ready')")),
            ("sandbox went away", block("x = 1\nprint('recovery ready')")),
            ("recovery ready", block("finish('recovered')")),
        ]
    )
    runtime = OneDeadRepl()
    g, root = agent_graph("count for me")
    assert run_result(Flow(llm, runtime=runtime), g, root) == "recovered"

    errors = [node for node in g.nodes() if isinstance(node, ErrorOutput)]
    assert len(errors) == 1 and "REPL execution failed" in errors[0].content
    assert runtime.opened == 2  # the dead one was dropped, not handed out again

    # A dead REPL reads differently from code that raised: the agent is told its
    # namespace went with it, and the record says which kind of failure it was.
    assert errors[0].error == "repl"
    assert errors[0].content.endswith(ReplDead().content)
    assert node_record(errors[0])["payload"]["error"] == "repl"


def test_each_launch_call_on_one_action_creates_a_distinct_child():
    parent = block(
        "first = launch_subagent(task='do it')\n"
        "second = launch_subagent(task='again')\n"
        "assert first.id != second.id\n"
        "print(await asyncio.gather(first, second))"
    )
    llm = ScriptedLLM(
        [
            ("delegate", parent),
            ("do it", block("finish('once')")),
            ("again", block("finish('twice')")),
            ("['once', 'twice']", block("finish('done')")),
        ]
    )
    g, root = agent_graph("delegate", max_depth=2)
    assert run_result(Flow(llm), g, root) == "done"
    assert [child.config.name for child in reads.children(g, root)] == ["child0", "child1"]


def test_launch_subagent_accepts_a_stable_child_name():
    parent = block(
        "worker = launch_subagent(name='worker-events-01', task='do it')\n"
        "assert worker.name == 'worker-events-01'\n"
        "print(await worker)"
    )
    llm = ScriptedLLM(
        [
            ("delegate", parent),
            ("do it", block("finish('done')")),
            ("done", block("finish('complete')")),
        ]
    )
    g, root = agent_graph("delegate", max_depth=1)
    assert run_result(Flow(llm), g, root) == "complete"
    assert [child.config.name for child in reads.children(g, root)] == ["worker-events-01"]


def test_saving_again_rewrites_the_whole_log(tmp_path):
    parent = block("child = launch_subagent(task='do a')\nprint(await child)")
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("finish('A')")), ("A", block("finish('AB')"))]
    )
    g, root = agent_graph("split this up")
    run_result(Flow(llm), g, root)
    run = g.save(tmp_path / "run")

    follow_up = ops.send(g, root, "and then?")
    g.save(run)

    loaded = Graph.load(run)
    assert [item.id for item in loaded.items()] == [item.id for item in g.items()]
    assert reads.inbox(loaded, loaded.node(root.id)) == [loaded.node(follow_up.id)]


def test_max_depth_refusal():
    llm = ScriptedLLM(
        [
            ("deep", block("print(launch_subagent(task='x'))")),
            ("is not defined", block("finish('stopped')")),
        ]
    )
    g, root = agent_graph("go deep", max_depth=0)
    flow = Flow(llm)
    with stepping(g, root):
        assert "launch_subagent" not in flow.build_tools(root)
    assert run_result(flow, g, root) == "stopped"
    assert reads.children(g, root) == []


def test_grandchildren():
    llm = ScriptedLLM(
        [
            (
                "plan the work",
                block("mid = launch_subagent(task='go mid')\nprint(await mid)"),
            ),
            (
                "go mid",
                block("leaf = launch_subagent(task='go leaf')\nprint(await leaf)"),
            ),
            ("go leaf", block("finish('leaf')")),
            ("leaf", block("finish('mid')")),
            ("mid", block("finish('root')")),
        ]
    )
    g, root = agent_graph("plan the work", max_depth=2)
    assert run_result(Flow(llm), g, root) == "root"
    mid = reads.children(g, root)[0]
    assert [reads.path(g, agent) for agent in (mid, reads.children(g, mid)[0])] == [
        "root.child0",
        "root.child0.child0",
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
    calls = ", ".join(f"launch_subagent(task='child {i}')" for i in range(3))
    llm = BarrierLLM(
        3,
        [
            (
                "fan out",
                block(f"handles = [{calls}]\nprint(await asyncio.gather(*handles))"),
            ),
            *[(f"child {i}", block(f"finish({i})")) for i in range(3)],
            ("[0, 1, 2]", block("finish('all')")),
        ],
    )
    g, root = agent_graph("fan out", max_depth=1)

    async def main():
        # Sequential children would never clear the barrier, so this would hang.
        await asyncio.wait_for(Flow(llm).arun(g), timeout=5)
        return reads.result(g, root)

    assert asyncio.run(main()) == "all"
    assert [reads.result(g, child) for child in reads.children(g, root)] == [0, 1, 2]


def test_background_child_outlives_root_and_stream_drains_it():
    release_child = threading.Event()

    class BackgroundLLM:
        def chat(self, messages):
            last = "\n\n".join(
                message["content"] for message in messages if message["role"] == "user"
            )
            if "slow child" in last:
                assert release_child.wait(timeout=5)
                return block("finish('child finished')")
            return block("child = launch_subagent(task='slow child')\nfinish(child.status)")

    flow = Flow(BackgroundLLM(), workers=2)
    g = Graph()
    root = g.create_agent(
        "launch without joining",
        config=AgentConfig(max_depth=1, resume_on_child_done=False),
    )

    async def run():
        observed = []
        async for item in flow.run_streaming(g):
            if not isinstance(item, Node):
                continue
            observed.append(item)
            if isinstance(item, DoneOutput) and item.agent_id == root.id:
                assert reads.result(g, root) == "running"
                assert not reads.terminal(g, reads.children(g, root)[0])
                release_child.set()
        await flow.aclose()
        return observed

    observed = asyncio.run(run())
    child = reads.children(g, root)[0]
    assert reads.result(g, child) == "child finished"
    root_done = next(
        i
        for i, node in enumerate(observed)
        if node.agent_id == root.id and isinstance(node, DoneOutput)
    )
    child_done = next(
        i
        for i, node in enumerate(observed)
        if node.agent_id == child.id and isinstance(node, DoneOutput)
    )
    assert root_done < child_done


def test_child_handle_waits_on_a_later_turn():
    wait = block("finish(await later)")
    llm = ScriptedLLM(
        [
            (
                "start later",
                block("later = launch_subagent(task='child task')\nprint('queued', later.name)"),
            ),
            ("child task", block("finish('child answer')")),
            ("queued child0", wait),
            ("[handoff root.child0]", wait),
        ]
    )
    g, root = agent_graph("start later", max_depth=1)

    assert run_result(Flow(llm), g, root) == "child answer"
    assert len(reads.children(g, root)) == 1
    assert reads.result(g, reads.children(g, root)[0]) == "child answer"


def test_completed_child_result_access_satisfies_finish_gate():
    llm = ScriptedLLM(
        [
            (
                "start child",
                block("child = launch_subagent(task='child task')\nprint('queued')"),
            ),
            ("child task", block("finish(42)")),
            ("result ready", block("finish(child.result())")),
            ("queued", block("print(child.status)")),
            ("running", block("print(child.status)")),
        ]
    )
    g, root = agent_graph("start child", max_depth=1)

    assert run_result(Flow(llm), g, root) == 42
    assert reads.result(g, reads.children(g, root)[0]) == 42


def test_launch_subagent_binds_task_and_json_arguments_directly():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block("child = launch_subagent(task='child', wait=False)\nprint(await child)"),
            ),
            ("child:False", block("finish('recovered')")),
            ("child", block("finish(task + ':' + str(wait))")),
        ]
    )
    g, root = agent_graph("delegate", max_depth=1)

    assert run_result(Flow(llm), g, root) == "recovered"
    assert reads.children(g, root)[0].resolved_args() == {"wait": False}


def test_launch_subagent_accepts_described_inputs():
    class InputLLM:
        def __init__(self):
            self.child_system_prompt = ""

        def chat(self, messages):
            if first_user(messages) == "child":
                self.child_system_prompt = messages[0]["content"]
                return block("finish(rows[0]['label'])")
            return block(
                "child = launch_subagent(\n"
                "    task='child',\n"
                "    rows=Input(rows, description='Records with label fields.'),\n"
                ")\n"
                "finish(await child)"
            )

    llm = InputLLM()
    g, root = agent_graph(
        "delegate",
        args={"rows": [{"label": "private-marker-7"}]},
        max_depth=1,
    )

    assert run_result(Flow(llm), g, root) == "private-marker-7"
    child = reads.children(g, root)[0]
    assert child.resolved_args() == {"rows": [{"label": "private-marker-7"}]}
    assert child.argument_descriptions() == {"rows": "Records with label fields."}
    assert "Records with label fields." in llm.child_system_prompt
    assert "private-marker-7" not in llm.child_system_prompt


def test_agents_result_matches_a_collected_child():
    llm = ScriptedLLM(
        [
            (
                "inspect waited",
                block("waited = launch_subagent(task='child task')\nprint(await waited)"),
            ),
            ("child task", block("finish('child answer')")),
            (
                "child answer",
                block("finish(await AGENTS.get(waited).wait_for_result())"),
            ),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    g = Graph()
    root = g.create_agent("inspect waited", config=AgentConfig(max_depth=1))

    assert run_result(flow, g, root) == "child answer"


def test_break_cancels_the_run():
    llm = ScriptedLLM([("", block("print('again')"))])
    flow = Flow(llm)

    async def main():
        stream = flow.run_streaming(agent_graph("loop forever")[0])
        async for _node in stream:
            break
        await stream.aclose()
        assert flow.queue is None
        await flow.aclose()

    asyncio.run(main())


def test_step_failure_reaches_the_caller():
    class BrokenLLM:
        def chat(self, messages):
            raise RuntimeError("model is down")

    g, root = agent_graph("anything")
    try:
        run_result(Flow(BrokenLLM()), g, root)
    except RuntimeError as exc:
        assert str(exc) == "model is down"
    else:
        raise AssertionError("expected the step failure to propagate")
    assert reads.result(g, root) == "[run failed: RuntimeError: model is down]"


def test_custom_tool_is_injected():
    @tool("Upper-case a string.")
    def shout(text: str) -> str:
        return text.upper()

    llm = ScriptedLLM([("", block("finish(shout('hi'))"))])
    flow = Flow(llm, tools=[shout])
    g, root = agent_graph("say it")
    assert run_result(flow, g, root) == "HI"


def test_child_output_schema_comes_back_parsed():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "number"}},
        "required": ["n"],
    }
    parent = block(
        f"child = launch_subagent(task='count', schema={schema!r})\n"
        "result = await child\n"
        "print(result['n'] + 1)"
    )
    llm = ScriptedLLM(
        [
            ("measure", parent),
            ("count", block("finish({'n': 41})")),
            ("42", block("finish('42')")),
        ]
    )
    g, root = agent_graph("measure it", max_depth=1)
    assert run_result(Flow(llm), g, root) == "42"
    assert reads.children(g, root)[0].config.output_schema == schema


def test_child_config_output_schema_applies_when_launch_omits_it():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "number"}},
        "required": ["n"],
    }
    parent = block(
        "child = launch_subagent(task='count')\nresult = await child\nprint(result['n'] + 1)"
    )
    llm = ScriptedLLM(
        [
            ("measure", parent),
            ("count", block("finish({'n': 41})")),
            ("42", block("finish('42')")),
        ]
    )
    g, root = agent_graph("measure it", max_depth=1, child_config={"output_schema": schema})
    assert run_result(Flow(llm), g, root) == "42"
    child = reads.children(g, root)[0]
    assert root.config.output_schema is None
    assert child.config.output_schema == schema
    assert child.config.output_schema is not root.config.child_config["output_schema"]
    assert child.config.child_config == {"output_schema": schema}


def test_launch_output_schema_overrides_child_config():
    inherited = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    }
    schema = {
        "type": "object",
        "properties": {"n": {"type": "number"}},
        "required": ["n"],
    }
    parent = block(
        f"child = launch_subagent(task='count', schema={schema!r})\n"
        "result = await child\n"
        "print(result['n'])"
    )
    llm = ScriptedLLM(
        [
            ("measure", parent),
            ("count", block("finish({'n': 41})")),
            ("41", block("finish('ok')")),
        ]
    )
    g, root = agent_graph("measure it", max_depth=1, child_config={"output_schema": inherited})
    assert run_result(Flow(llm), g, root) == "ok"
    child = reads.children(g, root)[0]
    assert child.config.output_schema == schema
    assert child.config.child_config == {"output_schema": inherited}


def test_named_model_routes_to_another_client():
    default = ScriptedLLM([("", block("finish('wrong client')"))])
    fast = ScriptedLLM([("", block("finish('fast client')"))])
    g, root = agent_graph("who answers", model="fast")
    assert run_result(Flow(default, llm_clients={"fast": fast}), g, root) == "fast client"
    assert default.calls == []


def test_child_config_model_routes_children():
    default = ScriptedLLM(
        [
            (
                "delegate",
                block("child = launch_subagent(task='do a')\nprint(await child)"),
            ),
            ("cheap result", block("finish('parent saw cheap result')")),
        ]
    )
    fast = ScriptedLLM([("do a", block("finish('cheap result')"))])
    flow = Flow(default, llm_clients={"fast": fast})

    g = Graph()
    root = g.create_agent(
        "delegate", config=AgentConfig(max_depth=1, child_config={"model": "fast"})
    )

    assert run_result(flow, g, root) == "parent saw cheap result"
    assert reads.children(g, root)[0].config.model == "fast"
    assert len(fast.calls) == 1


def test_child_config_profile_and_placement_route_children():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block("child = launch_subagent(task='do a')\nprint(await child)"),
            ),
            ("do a", block("finish('child')")),
            ("child", block("finish('done')")),
        ]
    )
    flow = Flow(
        llm,
        prompt_profiles={"worker": PromptProfile()},
        runtime=TestRuntime(),
    )

    g = Graph()
    root = g.create_agent(
        "delegate",
        config=AgentConfig(
            max_depth=1, child_config={"prompt_profile": "worker", "reuse_worker": True}
        ),
    )

    assert run_result(flow, g, root) == "done"
    assert reads.children(g, root)[0].config.prompt_profile == "worker"
    assert reads.children(g, root)[0].config.reuse_worker is True


def test_child_launch_inherits_parent_model_by_default():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block("child = launch_subagent(task='do a')\nprint(await child)"),
            ),
            ("do a", block("finish('child')")),
            ("child", block("finish('done')")),
        ]
    )
    g, root = agent_graph("delegate", max_depth=1)

    assert run_result(Flow(llm), g, root) == "done"
    assert reads.children(g, root)[0].config.model == "default"


def test_launch_model_keyword_is_ordinary_child_argument_not_policy():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block(
                    "child = launch_subagent(task='do a', model='input value')\nprint(await child)"
                ),
            ),
            ("do a", block("finish(model)")),
            ("input value", block("finish('done')")),
        ]
    )
    g, root = agent_graph("delegate", max_depth=1)

    assert run_result(Flow(llm), g, root) == "done"
    assert reads.children(g, root)[0].config.model == "default"


def test_long_child_query_is_refused():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block("print(launch_subagent(task='far too long'))"),
            ),
            ("exceeds", block("finish('stopped')")),
        ]
    )
    g, root = agent_graph("delegate this", max_depth=1, max_query_chars=4)
    assert run_result(Flow(llm), g, root) == "stopped"
    assert reads.children(g, root) == []


def test_a_nearly_spent_budget_asks_for_an_answer_instead_of_dropping_the_run():
    # Iteration exhaustion always got a FinalQuery to answer in, but token
    # exhaustion used to end a run holding nothing even when iterations remained.
    llm = ScriptedLLM(
        [
            ("last turn", block("finish('best guess')")),
            ("one", block("print('two')")),
            ("", block("print('one')")),
        ]
    )
    llm.last_usage = SimpleNamespace(input_tokens=4, output_tokens=4)
    g, root = agent_graph("spend most of it", max_budget=20)

    assert run_result(Flow(llm), g, root) == "best guess"
    assert sum(isinstance(node, FinalQuery) for node in reads.history(g, root)) == 1


def test_a_failed_final_turn_under_a_token_budget_ends_instead_of_looping():
    llm = ScriptedLLM(
        [
            ("last turn", block("y = 2")),
            ("one", block("print('two')")),
            ("", block("print('one')")),
        ]
    )
    llm.last_usage = SimpleNamespace(input_tokens=4, output_tokens=4)
    g, root = agent_graph("spend most of it", max_budget=20)

    assert run_result(Flow(llm), g, root) == "[max_budget exceeded]"
    assert sum(isinstance(node, FinalQuery) for node in reads.history(g, root)) == 1
    assert len(llm.calls) == 3


def test_the_budget_gate_never_discards_an_answer_already_paid_for():
    # The gate used to fire on any node, so a reply that finished the task could be
    # dropped between generation and the exec that lands it.
    llm = ScriptedLLM([("", block("finish('landed')"))])
    llm.last_usage = SimpleNamespace(input_tokens=40, output_tokens=40)
    g, root = agent_graph("answer immediately", max_budget=10)

    assert run_result(Flow(llm), g, root) == "landed"


def test_exhausted_budget_finalizes_children_before_the_parent():
    class ConcurrentBudgetLLM:
        def __init__(self):
            self.children_ready = asyncio.Barrier(2)

        async def stream(self, messages):
            query = first_user(messages)
            final = messages[-1]["content"].startswith("This is your last turn")
            if final:
                if query == "parent":
                    assert any("a-final,b-final" in message["content"] for message in messages)
                    reply = block("finish('root-final')")
                else:
                    reply = block(f"finish({f'{query}-final'!r})")
                usage = LLMUsage(output_tokens=1)
            elif query == "parent":
                reply = block(
                    "handles = [\n"
                    "    launch_subagent(task='a'),\n"
                    "    launch_subagent(task='b'),\n"
                    "]\n"
                    "answers = await asyncio.gather(*handles)\n"
                    "print(','.join(answers))"
                )
                usage = LLMUsage(output_tokens=1)
            else:
                await self.children_ready.wait()
                reply = block(f"draft = {query!r}\nprint(draft)")
                usage = LLMUsage(output_tokens=8)
            yield LLMChunk(text=reply, usage=usage)

    g, root = agent_graph("parent", max_depth=1, max_budget=10)
    result = run_result(Flow(ConcurrentBudgetLLM(), runtime=TestRuntime()), g, root)

    assert result == "root-final"
    assert reads.usage(g).total > root.config.max_budget
    assert len(reads.children(g, root)) == 2
    for agent in (root, *reads.children(g, root)):
        assert sum(isinstance(node, FinalQuery) for node in reads.history(g, agent)) == 1


def test_exhausted_budget_rejects_new_children():
    flow = Flow(ScriptedLLM([]))
    g, root = agent_graph("parent", max_depth=1, max_budget=1)
    ops.append(g, root, LLMOutput(usage=LLMUsage(output_tokens=1)))
    action = ops.append(g, root, ExecAction(code="pass"))

    with stepping(g, root), pytest.raises(RuntimeError, match="cannot launch.*max_budget"):
        launch.new_child(
            flow,
            action,
            "late",
            {"query": "too late", "model": "default"},
        )


def test_child_failure_reaches_the_parent():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block("child = launch_subagent(task='boom')\nprint(await child)"),
            ),
            ("boom", RuntimeError("child is down")),
            ("child is down", block("finish('recovered')")),
        ]
    )
    g, root = agent_graph("delegate this", max_depth=1)
    assert run_result(Flow(llm), g, root) == "recovered"


def test_aclose_drops_the_work_and_the_repls():
    class ClosingLLM(ScriptedLLM):
        closed = False

        async def aclose(self):
            self.closed = True

    llm = ClosingLLM([("count", block("finish('six')"))])
    flow = Flow(llm)

    async def run_then_close():
        await flow.arun(agent_graph("count for me")[0])
        assert flow.repls
        await flow.aclose()

    asyncio.run(run_then_close())
    assert llm.closed
    assert not flow.repls
    assert flow.queue is None


def test_one_flow_rejects_competing_stream_drivers():
    llm = ScriptedLLM([("one", block("finish('first')")), ("two", block("finish('second')"))])
    flow = Flow(llm)
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
    flow = Flow(StubLLM(reply), workers=2)

    async def query_all():
        return await asyncio.gather(*(flow.llm_query(value) for value in ("a", "b", "c", "d")))

    assert asyncio.run(query_all()) == ["A", "B", "C", "D"]
    assert seen["peak"] == 2


def test_a_pool_replaces_the_one_workers_would_have_built():
    reply, seen = peaked()
    flow = Flow(StubLLM(reply), pool=SequentialPool())

    async def query_all():
        return await asyncio.gather(*(flow.llm_query(value) for value in ("a", "b", "c")))

    assert asyncio.run(query_all()) == ["A", "B", "C"]
    assert seen["peak"] == 1

    with pytest.raises(ValueError, match="not both"):
        Flow(StubLLM(reply), workers=2, pool=SequentialPool())


def test_llm_request_timeout_gives_up_on_a_slow_call():
    class SlowLLM:
        async def chat(self, messages):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        Flow(SlowLLM(), llm_request_timeout=0.01).run("hang")


def test_llm_request_timeout_is_handed_to_clients_that_take_one():
    llm = StubLLM(lambda _messages: block("finish('ok')"))
    assert Flow(llm, llm_request_timeout=5).run("go") == "ok"
    assert [call["timeout"] for call in llm.calls] == [5]

    # A client with no timeout parameter is called the way it was written.
    lean = ScriptedLLM([("", block("finish('ok')"))])
    assert Flow(lean, llm_request_timeout=5).run("go") == "ok"
