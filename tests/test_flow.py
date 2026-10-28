import asyncio
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
    first_user,
    run_result,
    stepping,
    trailing_user_text,
)
from helpers import reasoning_repl as block

from rlmflow import (
    AGENT_START,
    CODE,
    DEPENDS_ON,
    DONE,
    FROM,
    LLM_OUTPUT,
    NEXT,
    AgentConfig,
    AgentHandle,
    Input,
    LLMChunk,
    LLMUsage,
    Node,
    SequentialPool,
    StaleFrontier,
)
from rlmflow.agent_tools import tools as agent_tools
from rlmflow.graph import Graph, Step
from rlmflow.graph.persistence import node_record
from rlmflow.harnesses import repl
from rlmflow.runtime import LocalRuntime
from rlmflow.tools import repl_tools, split_args, tool
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
    flow = ScriptedFlow(llm)
    assert flow.run("count for me") == 6
    assert llm.calls[0][0]["role"] == "system"


def depends(g, agent):
    """``(as, producer id)`` for each dependency edge, in write order."""
    return tuple((e.data.get("as"), e.target) for e in g.edges(kind=DEPENDS_ON, source=agent.id))


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
        prompt={"protocol": "Be terse.", "agents": None, "style": "No emoji."},
    )
    assert prompt.startswith("Be terse.")
    assert "submit(" in prompt
    assert "AGENTS" not in prompt
    assert prompt.endswith("No emoji.")


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
    assert any(node.type == CODE and node.data["error"] for node in g.nodes())


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
    assert len(g.nodes(LLM_OUTPUT)) == 3


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
    loaded = Graph.load(run).node(root.id)

    assert loaded.resolved_args() == {"document": document}
    assert loaded.argument_descriptions() == {"document": "A large source document."}
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
    agent = g.spawn("classify", config="classifier", text="hi")

    assert agent.args == {"text": "hi"}
    assert agent.name == "classifier"
    assert agent.config_name == "classifier"
    loaded = reload(g, tmp_path / "run").node(agent.id)
    assert loaded.config_name == "classifier"

    flow = ScriptedFlow(ScriptedLLM([("classify", block("submit({'label': text})"))]))
    config = flow.add_config(AgentConfig(name="classifier", output_schema=Label))
    assert config.output_schema["properties"]["label"]["type"] == "string"
    graph = Graph()
    graph.spawn("classify", config="classifier", text="hi")
    flow.run(graph)
    [classified] = graph.agents()
    assert classified.config is config
    assert graph.result(classified) == {"label": "hi"}


def test_graph_edited_without_a_flow_runs_on_a_fresh_flow(tmp_path):
    """Spawn, append history, and fork with ops alone; a new Flow builds every REPL."""
    g = Graph()
    a = g.create_agent("count", name="a")
    cell = g.append(a, Node(CODE, code="n = 41", output=""))
    b = g.fork(cell, name="b")
    g.send(a, "finish a")
    g.send(b, "finish b")

    loaded = reload(g, tmp_path / "run")
    flow = ScriptedFlow(
        ScriptedLLM(
            [
                ("finish a", block("submit(n + 1)")),
                ("finish b", block("submit(n + 2)")),
            ]
        )
    )
    flow.run(loaded)

    assert loaded.result(loaded.node(a.id)) == 42
    assert loaded.result(loaded.node(b.id)) == 43


def test_replay_rebuilds_a_fork_that_never_stepped():
    flow = ScriptedFlow(ScriptedLLM([]))
    g = Graph()
    a = g.create_agent("count", name="a")
    cell = g.append(a, Node(CODE, code="n = 41", output=""))
    fork = g.fork(cell, name="fork")

    asyncio.run(flow.replay(g, fork))

    assert flow.runtime.get(fork).namespace["n"] == 41
    asyncio.run(flow.aclose())


def test_agent_added_mid_stream_is_scheduled():
    llm = ScriptedLLM([("first", block("submit('one')")), ("second", block("submit('two')"))])
    flow = ScriptedFlow(llm)
    g = Graph()
    first = g.create_agent("first")
    added = []

    async def drive():
        async for _step in flow.run_streaming(g):
            if not added:
                added.append(g.create_agent("second"))

    asyncio.run(drive())

    assert g.result(first) == "one"
    assert g.result(added[0]) == "two"


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

    messages = repl.ReplHarness().messages(flow, Step(g, root, g.frontier(root)))
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
    assert seen[-1].type == CODE and seen[-1].data["output"].strip() == "peek"
    assert not g.terminal(root)

    rest = asyncio.run(stream(flow, g, "done"))
    assert rest[-1].type == DONE and g.result(root) == "finished"


def test_each_streamed_step_names_its_agent_and_node():
    llm = ScriptedLLM([("look", block("print('peek')")), ("peek", block("submit('finished')"))])
    flow = ScriptedFlow(llm)
    g, root = agent_graph("look")

    async def collect():
        return [step async for step in flow.run_streaming(g)]

    steps = asyncio.run(collect())
    assert [step.node.type for step in steps] == [LLM_OUTPUT, CODE, LLM_OUTPUT, DONE]
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
    children = g.children(root)
    assert [child.name for child in children] == ["default-0", "default-1"]
    assert [g.result(child) for child in children] == ["A", "B"]

    own = {node.id for node in g.history(root)}
    assert all(node.agent_id == root.id for node in g.history(root))
    assert own.isdisjoint(node.id for node in g.history(children[0]))
    assert g.history(children[0])[0] is children[0]

    # Children are created during the model turn's cell; the cell records them.
    [cell] = [n for n in g.history(root) if n.type == CODE and n.launched()]
    assert cell.launched() == [child.id for child in children]
    [turn] = {g.node(e.target) for child in children for e in g.edges(kind=FROM, source=child.id)}
    assert turn.type == LLM_OUTPUT and g.prev(cell) is turn

    streamed = asyncio.run(stream(ScriptedFlow(llm), agent_graph("split this up")[0], "done"))
    assert sum(node.type == AGENT_START for node in streamed) == 2


def test_handle_valued_args_build_a_flat_dependency_graph(tmp_path):
    parent = block(
        "parser = launch_agent('parse-source')\n"
        "fixtures = launch_agent('inspect-fixtures')\n"
        "reconciler = launch_agent(\n"
        "    'reconcile-inputs',\n"
        "    parsed=Input(parser, description='Parsed source counts.'),\n"
        "    fixtures=fixtures,\n"
        ")\n"
        "critic = launch_agent('critique-answer', reconciler=reconciler)\n"
        "print(await critic)"
    )
    llm = ScriptedLLM(
        [
            ("build-dag", parent),
            ("parse-source", block("submit({'count': 2})")),
            ("inspect-fixtures", block("submit('F')")),
            ("reconcile-inputs", block("submit(f\"{parsed['count']}:{fixtures}\")")),
            ("critique-answer", block("submit(reconciler + '!')")),
            ("2:F!", block("submit('done')")),
        ]
    )
    g, root = agent_graph("build-dag")

    assert run_result(ScriptedFlow(llm), g, root) == "done"
    parser, fixtures, reconciler, critic = g.children(root)
    assert {g.depth(agent) for agent in g.children(root)} == {1}
    assert depends(g, reconciler) == (("parsed", parser.id), ("fixtures", fixtures.id))
    assert depends(g, critic) == (("reconciler", reconciler.id),)
    assert reconciler.args == {}
    assert g.inputs(reconciler) == {"parsed": {"count": 2}, "fixtures": "F"}
    [described] = g.edges(kind=DEPENDS_ON, source=reconciler.id, target=parser.id)
    assert described.data["description"] == "Parsed source counts."
    assert g.inputs(critic) == {"reconciler": "2:F"}

    tasks = [first_user(messages) for messages in llm.calls]
    assert tasks.index("reconcile-inputs") > tasks.index("parse-source")
    assert tasks.index("reconcile-inputs") > tasks.index("inspect-fixtures")
    assert tasks.index("critique-answer") > tasks.index("reconcile-inputs")

    loaded = reload(g, tmp_path / "dependency-run")
    assert depends(loaded, loaded.node(reconciler.id)) == depends(g, reconciler)
    assert depends(loaded, loaded.node(critic.id)) == depends(g, critic)


def test_shared_source_feeds_multiple_flat_consumers():
    parent = block(
        "source = launch_agent('shared-source')\n"
        "solver = launch_agent('solve-source', source=source)\n"
        "reviewer = launch_agent('review-source', source=source)\n"
        "print(await asyncio.gather(solver, reviewer))"
    )
    llm = ScriptedLLM(
        [
            ("fan-out", parent),
            ("shared-source", block("submit('evidence')")),
            ("solve-source", block("submit(source + ':solved')")),
            ("review-source", block("submit(source + ':reviewed')")),
            ("['evidence:solved', 'evidence:reviewed']", block("submit('done')")),
        ]
    )
    g, root = agent_graph("fan-out")

    assert run_result(ScriptedFlow(llm), g, root, root_config=AgentConfig(max_depth=2)) == "done"
    source, solver, reviewer = g.children(root)
    assert depends(g, solver) == (("source", source.id),)
    assert depends(g, reviewer) == (("source", source.id),)
    [cell] = [n for n in g.history(root) if n.type == CODE and n.launched()]
    assert set(cell.launched()) == {source.id, solver.id, reviewer.id}


def _launch_graph():
    graph = Graph()
    root = graph.create_agent("root")
    cell = graph.append(root, Node(CODE, code="launch"))
    return graph, root, cell


def test_launch_args_accept_mixed_strings_and_top_level_handles():
    g, _root, cell = _launch_graph()
    source = g.spawn("source", by=cell, name="source", config="default")
    handle = AgentHandle(source.id, "source", "root.source")

    plain, inputs = split_args(g, {"prompt": "immediate", "evidence": handle})
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
    assert split_args(_launch_graph()[0], args) == (args, {})


@pytest.mark.parametrize("value", [{1, 2}, object(), {1: "bad key"}])
def test_launch_args_reject_non_json_values(value):
    with pytest.raises(TypeError, match="JSON-compatible|non-string"):
        split_args(_launch_graph()[0], {"value": value})


@pytest.mark.parametrize("name", ["bad-name", "class", "task"])
def test_launch_args_require_python_variable_names(name):
    with pytest.raises(ValueError, match="invalid invocation|'task' is the invocation"):
        split_args(_launch_graph()[0], {name: "value"})


def test_launch_args_reject_unknown_and_nested_handles():
    g, _root, cell = _launch_graph()
    source = g.spawn("source", by=cell, name="source", config="default")
    source_handle = AgentHandle(source.id, "source", "root.source")

    with pytest.raises(ValueError, match="unknown agent"):
        split_args(g, {"missing": AgentHandle("missing", "missing", "root.missing")})
    with pytest.raises(TypeError, match="nests a handle"):
        split_args(g, {"nested": [source_handle]})


def test_dependency_scheduler_resumes_a_saved_blocked_agent(tmp_path):
    parent = block(
        "source = launch_agent('produce-source')\n"
        "consumer = launch_agent('consume-source', source=source)\n"
        "print(await consumer)"
    )
    llm = ScriptedLLM(
        [
            ("resume-dag", parent),
            ("produce-source", block("submit('A')")),
            ("consume-source", block("submit(source + '!')")),
            ("A!", block("submit('resumed')")),
        ]
    )
    g, root = agent_graph("resume-dag")
    flow = ScriptedFlow(llm)
    deep = AgentConfig(max_depth=2)

    def source_done(step):
        return step.node.type == DONE and step.agent.name == "default-0"

    stopped = asyncio.run(stream(flow, g, source_done, root_config=deep))
    assert g.node(stopped[-1].agent_id).name == "default-0"
    source, consumer = g.children(root)
    assert depends(g, consumer) == (("source", source.id),)
    assert consumer in g.ready()
    assert g.frontier(consumer) is consumer

    restored = reload(g, tmp_path / "blocked-run")
    assert (
        run_result(ScriptedFlow(llm), restored, restored.node(root.id), root_config=deep)
        == "resumed"
    )
    assert restored.inputs(restored.node(consumer.id)) == {"source": "A"}
    assert len(restored.children(restored.node(root.id))) == 2


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
    assert sum(node.type == LLM_OUTPUT for node in g.history(root)) == 4
    assert len(g.children(root)) == 2
    outputs = [node.data["output"].strip() for node in g.history(root) if node.type == CODE]
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
    assert set(saved) == {"format", "id", "items"}
    assert saved["id"] == g.id
    items = saved["items"]
    assert [item["id"] for item in items] == [item.id for item in g.items()]
    assert items[0]["data"]["args"] == {"doc": "hi"}

    child = g.children(root)[0]
    [launcher] = g.edges(kind=FROM, source=child.id)
    edges = [item for item in items if "kind" in item]
    assert {"kind": FROM, "source": child.id, "target": launcher.target} in [
        {key: edge[key] for key in ("kind", "source", "target")} for edge in edges
    ]
    child_nodes = [item for item in items if item.get("agent_id") == child.id]
    assert [item["id"] for item in child_nodes] == [node.id for node in g.history(child)]
    assert child_nodes[0]["data"]["task"] == "do a"


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

    # seq is the graph's log position, so each history is strictly increasing and
    # the log order is the whole run, however the two children raced.
    for agent in (root, *g.children(root)):
        seqs = [node.seq for node in g.history(agent)]
        assert seqs == sorted(set(seqs))
    assert [item.seq for item in g.items()] == list(range(len(g.items())))

    # Only nodes a harness returned are timed; agent_start and message are written.
    stepped = [node for node in g.nodes() if node.type in (LLM_OUTPUT, CODE, DONE)]
    assert stepped and all(node.timing()["duration_ms"] >= 0 for node in stepped)
    written = g.agents()
    assert written and all(node.timing() == {} for node in written)


def test_appending_off_the_frontier_is_refused():
    llm = ScriptedLLM([("count", block("x = 2\nprint(x)")), ("2", block("submit(x * 3)"))])
    g, root = agent_graph("count for me")
    assert run_result(ScriptedFlow(llm), g, root) == 6

    stale = g.history(root)[1]
    with pytest.raises(StaleFrontier, match="frontier"):
        g.append(root, Node(CODE, code=""), after=stale)
    assert len(g.edges(kind=NEXT, source=stale.id)) == 1


def test_structured_output_is_stored_parsed():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM([("count", block("submit({'n': 41})"))])
    g, root = agent_graph("count")
    assert run_result(
        ScriptedFlow(llm), g, root, root_config=AgentConfig(output_schema=schema)
    ) == {"n": 41}
    assert node_record(g.frontier(root))["data"]["result"] == {"n": 41}


def test_plain_text_json_reply_is_validated_against_the_schema():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM([("count", '{"n": 41}')])
    g, root = agent_graph("count")

    assert run_result(
        ScriptedFlow(llm), g, root, root_config=AgentConfig(output_schema=schema)
    ) == {"n": 41}
    assert not g.nodes(CODE)


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
    [rejected] = [n for n in g.nodes(CODE) if n.data["error"]]
    assert rejected.data["code"] == ""


def test_unstructured_submission_keeps_json_values():
    expected = {"n": [41, True, None, "é"]}
    llm = ScriptedLLM([("value", block(f"submit({expected!r})"))])
    g, root = agent_graph("return value")

    assert run_result(ScriptedFlow(llm), g, root) == expected
    assert node_record(g.frontier(root))["data"]["result"] == expected


def test_unstructured_submission_rejects_non_json_values():
    llm = ScriptedLLM(
        [
            ("return value", block("submit({1, 2})")),
            ("JSON", block("submit('recovered')")),
        ]
    )
    g, root = agent_graph("return value")

    assert run_result(ScriptedFlow(llm), g, root) == "recovered"
    assert any(
        n.type == CODE and n.data["error"] and "JSON" in n.data["output"] for n in g.history(root)
    )


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
    assert loaded.result(loaded.node(root.id)) == {"n": [1, True, None]}


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
    assert not g.nodes(CODE)


def test_plain_text_reply_may_contain_non_repl_code_fences():
    answer = "Use a list comprehension:\n```python\nprint([x * 2 for x in xs])\n```"
    llm = ScriptedLLM([("explain", answer)])
    g, root = agent_graph("explain doubling")

    assert run_result(ScriptedFlow(llm), g, root) == answer
    assert not g.nodes(CODE)


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
    [error] = [node for node in g.nodes(CODE) if node.data["error"]]
    assert error.data["code"] == ""
    assert error.data["output"].startswith(reason)
    assert "Nothing ran" in error.data["output"]
    assert "reply in plain text with no ```repl block" in error.data["output"]


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
            lambda step: step.node.type == CODE and not step.node.data["error"],
        )
    )
    loaded = reload(g, tmp_path / "protocol-error")

    resumed = ScriptedLLM([("stored", block("submit(secret + 1)"))])
    assert run_result(ScriptedFlow(resumed), loaded, loaded.node(root.id)) == 42
    assert sum(1 for n in loaded.nodes(CODE) if n.data["error"]) == 1


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
    loaded_root = loaded.node(root.id)
    assert [item.id for item in loaded.items()] == [item.id for item in g.items()]
    assert (loaded.result(loaded_root), loaded.terminal(loaded_root)) == ("AB", True)
    [child] = loaded.children(loaded_root)
    assert loaded.path(child) == "root.default-0"
    assert loaded.result(child) == "A"
    assert loaded.save(tmp_path / "again") and (tmp_path / "again" / "graph.json").is_file()


def paused_run(tmp_path, name="run"):
    """A saved run that set a variable and stopped before using it."""
    llm = ScriptedLLM([("count", block("secret = 41\nprint('stored')"))])
    g, root = agent_graph("count for me")
    asyncio.run(stream(ScriptedFlow(llm), g, "idle"))
    assert g.frontier(root).type == CODE
    loaded = reload(g, tmp_path / name)
    return loaded, loaded.node(root.id)


def test_resuming_a_saved_run_rebuilds_its_repl(tmp_path):
    loaded, root = paused_run(tmp_path)
    llm = ScriptedLLM([("stored", block("submit(secret + 1)"))])
    assert run_result(ScriptedFlow(llm), loaded, root) == 42
    assert not any(n.data["error"] for n in loaded.nodes(CODE))
    assert len(llm.calls) == 1  # the rebuild itself asks the model nothing


def test_the_rebuild_waits_until_code_needs_the_repl(tmp_path):
    loaded, root = paused_run(tmp_path)
    llm = ScriptedLLM([("stored", block("submit(secret + 1)"))])
    flow = ScriptedFlow(llm)

    async def stop_before_execution():
        async for _ in flow.run_streaming(loaded, until=lambda step: step.node.type == LLM_OUTPUT):
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
                "slow = launch_agent('slow', name='slow')\n"
                "fast = launch_agent('fast', name='fast')\n"
                "results = await asyncio.gather(slow, fast)\n"
                "submit(f'{seed}:' + ''.join(results))"
            )
        if task == "slow":
            return block("await asyncio.sleep(0.2)\nsubmit('S')")
        return block("submit('F')")

    g, root = agent_graph("parent")
    flow = ScriptedFlow(StubLLM(reply), runtime=LocalRuntime(repl_timeout=10))

    async def halt_at_first_child():
        async for _ in flow.run_streaming(g, until=lambda step: step.node.type == AGENT_START):
            pass

    try:
        asyncio.run(halt_at_first_child())
        assert run_result(flow, g, root) == "40:SF"
        assert sorted(child.name for child in g.children(root)) == ["fast", "slow"]
    finally:
        asyncio.run(flow.aclose())


def test_rebuild_reads_recorded_child_answers_instead_of_relaunching(tmp_path):
    parent = block("child = launch_agent('add up')\nanswer = await child\nprint(answer)")
    llm = ScriptedLLM([("delegate", parent), ("add up", block("submit('20')"))])
    g, root = agent_graph("delegate")

    def parent_turn_ended(step):
        return step.node.type == CODE and step.agent is root

    asyncio.run(
        stream(ScriptedFlow(llm), g, parent_turn_ended, root_config=AgentConfig(max_depth=2))
    )
    loaded = reload(g, tmp_path / "run")
    loaded_root = loaded.node(root.id)

    resumed = ScriptedLLM([("20", block("submit('parent saw ' + answer)"))])
    assert run_result(ScriptedFlow(resumed), loaded, loaded_root) == "parent saw 20"
    # The child ran once, when it was recorded: no second launch, no second model call.
    assert len(loaded.children(loaded_root)) == 1
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
    asyncio.run(stream(setup_flow, g, lambda step: step.node.type == CODE))
    asyncio.run(setup_flow.aclose())
    loaded = reload(g, tmp_path / "llm-query-run")
    resumed = ScriptedLLM([("recorded", block("submit(answer)"))])

    flow = ScriptedFlow(resumed, runtime=LocalRuntime(repl_timeout=10))
    try:
        assert run_result(flow, loaded, loaded.node(root.id)) == "recorded"
    finally:
        asyncio.run(flow.aclose())
    assert len(resumed.calls) == 1


JOIN_CHILDREN = block("print([await c for c in AGENTS.get_children()])")


def test_attached_subtree_runs_and_reports_to_its_new_parent():
    g, root = agent_graph("parent")
    branch = g.spawn("child", name="branch0")
    g.attach(root, branch)

    flow = ScriptedFlow(
        ScriptedLLM(
            [
                ("now report to you", JOIN_CHILDREN),
                ("['finished']", block("submit('finished')")),
                ("child", block("submit('finished')")),
            ]
        )
    )

    assert run_result(flow, g, root) == "finished"
    assert g.children(root) == [branch]
    assert g.path(branch) == "root.branch0"


def test_controller_attaches_multiple_subtrees_between_streams():
    llm = ScriptedLLM(
        [
            ("['A', 'B']", block("submit('AB')")),
            ("now report to you", JOIN_CHILDREN),
            ("prepare", block("print('prepared')")),
            ("child a", block("submit(value)")),
            ("child b", block("submit(value)")),
        ]
    )
    flow = ScriptedFlow(llm)
    g, root = agent_graph("prepare")

    def prepared(task, name, code):
        """A subtree built in its own graph, as a controller would prepare it."""
        side = Graph()
        agent = side.spawn(task, name=name)
        side.append(agent, Node(CODE, code=code, output=task))
        g.merge(side)
        return agent

    async def main():
        async for _step in flow.run_streaming(g, until="idle"):
            pass

        children = [
            prepared("child a", "a", "value = 'A'"),
            prepared("child b", "b", "value = 'B'"),
        ]
        g.attach(root, *children)

        async for _step in flow.run_streaming(g):
            pass

    asyncio.run(main())

    assert g.result(root) == "AB"
    assert [child.name for child in g.children(root)] == ["a", "b"]


def test_attach_message_survives_save_load(tmp_path):
    g, root = agent_graph("parent")
    branch = g.spawn("child", name="branch0")
    message = g.attach(root, branch)

    loaded = reload(g, tmp_path / "run")
    [loaded_message] = loaded.inbox(loaded.node(root.id))

    assert loaded_message.data == message.data
    assert [c.name for c in loaded.children(loaded.node(root.id))] == ["branch0"]


def test_rebuilding_agents_can_skip_their_own_unrepeatable_work(tmp_path):
    trace: list[str] = []

    @tool("Record a live side effect.", proxy=True)
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
    assert run_result(resumed, loaded, loaded.node(root.id)) == "ok"
    assert trace == ["live"]  # the rebuild saw RLMFLOW_REPLAY=1 and skipped it


def test_a_finished_run_resumes_to_its_recorded_answer(tmp_path):
    llm = ScriptedLLM([("count", block("submit('six')"))])
    g, root = agent_graph("count for me")
    assert run_result(ScriptedFlow(llm), g, root) == "six"

    loaded = reload(g, tmp_path / "run")
    idle = ScriptedLLM([])
    assert run_result(ScriptedFlow(idle), loaded, loaded.node(root.id)) == "six"
    assert idle.calls == []  # nothing left to step


def test_a_dead_repl_is_an_observation_and_is_rebuilt():
    class OneDeadRepl(LocalRuntime):
        """The first REPL dies on its second cell, the way a sandbox does."""

        opened = 0

        def open(self, agent):
            self.opened += 1
            repl = super().open(agent)
            if self.opened == 1:
                run, cells = repl.run, []

                async def die_on_second(code, **kwargs):
                    cells.append(code)
                    if len(cells) == 2:
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

    [dead] = [node for node in g.nodes(CODE) if node.data.get("dead")]
    assert "REPL execution failed" in dead.data["output"]
    assert "rebuilt from the recorded cells" in dead.data["output"]
    assert runtime.opened == 2  # rebuilt from the first cell, without the dead one


def test_each_launch_call_in_one_cell_creates_a_distinct_child():
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
    assert [child.name for child in g.children(root)] == ["default-0", "default-1"]


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
    assert [child.name for child in g.children(root)] == ["worker_events_01"]


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
    assert [item.id for item in loaded.items()] == [item.id for item in g.items()]
    assert loaded.inbox(loaded.node(root.id)) == [loaded.node(follow_up.id)]


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
    assert g.children(root) == []


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
    mid = g.children(root)[0]
    assert [g.path(agent) for agent in (mid, g.children(mid)[0])] == [
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
        return g.result(root)

    assert asyncio.run(main()) == "all"
    assert [g.result(child) for child in g.children(root)] == [0, 1, 2]


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
    root = g.create_agent("launch without joining")

    async def run():
        observed = []
        async for step in flow.run_streaming(g):
            observed.append(step.node)
            if step.node.type == DONE and step.agent is root:
                assert g.result(root) == "running"
                assert not g.terminal(g.children(root)[0])
                release_child.set()
        await flow.aclose()
        return observed

    observed = asyncio.run(run())
    child = g.children(root)[0]
    assert g.result(child) == "child finished"
    assert g.result(root) == "running"  # done is final: the child's result doesn't reopen it
    root_done = next(
        i for i, node in enumerate(observed) if node.agent_id == root.id and node.type == DONE
    )
    child_done = next(
        i for i, node in enumerate(observed) if node.agent_id == child.id and node.type == DONE
    )
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
            ("[handoff root.default-0]", wait),
        ]
    )
    g, root = agent_graph("start later")

    assert run_result(ScriptedFlow(llm), g, root) == "child answer"
    assert len(g.children(root)) == 1
    assert g.result(g.children(root)[0]) == "child answer"


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
    assert g.result(g.children(root)[0]) == 42


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
    assert g.children(root)[0].resolved_args() == {"wait": False}


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
    child = g.children(root)[0]
    assert child.resolved_args() == {"rows": [{"label": "private-marker-7"}]}
    assert child.argument_descriptions() == {"rows": "Records with label fields."}
    assert "Records with label fields." in llm.child_system_prompt
    assert "private-marker-7" not in llm.child_system_prompt


def test_agents_result_matches_a_collected_child():
    llm = ScriptedLLM(
        [
            ("inspect waited", block("waited = launch_agent('child task')\nprint(await waited)")),
            ("child task", block("submit('child answer')")),
            ("child answer", block("submit(await AGENTS.get(waited).wait_for_result())")),
        ]
    )
    flow = ScriptedFlow(llm)
    g = Graph()
    root = g.create_agent("inspect waited")

    assert run_result(flow, g, root) == "child answer"


def test_break_cancels_the_run():
    llm = ScriptedLLM([("", block("print('again')"))])
    flow = ScriptedFlow(llm)

    async def main():
        stream = flow.run_streaming(agent_graph("loop forever")[0])
        async for _step in stream:
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
    with pytest.raises(RuntimeError, match="model is down"):
        run_result(ScriptedFlow(BrokenLLM()), g, root)
    assert g.result(root) == "[run failed: RuntimeError: model is down]"


def test_custom_tool_is_injected():
    @tool("Upper-case a string.")
    def shout(text: str) -> str:
        return text.upper()

    llm = ScriptedLLM([("", block("submit(shout('hi'))"))])
    flow = ScriptedFlow(llm, tools=[shout])
    g, root = agent_graph("say it")
    assert run_result(flow, g, root) == "HI"


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
    child = g.children(root)[0]
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
    root = g.create_agent("delegate")

    assert run_result(flow, g, root) == "parent saw cheap result"
    assert g.children(root)[0].config.model == "fast"
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
    child = g.children(root)[0]
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
    assert g.children(root)[0].config.model is None


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
    assert g.children(root) == []


def final_messages(g, agent):
    return [n for n in g.history(agent) if n.type == "message" and n.data.get("final")]


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
    assert final.data["sender"] == "rlmflow"


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
    assert len(g.children(root)) == 2
    for agent in (root, *g.children(root)):
        assert len(final_messages(g, agent)) == 1


def test_exhausted_budget_rejects_new_children():
    flow = ScriptedFlow(ScriptedLLM([]))
    g, root = agent_graph("parent")
    root.bind_config(AgentConfig(max_budget=1))
    turn = g.append(root, Node(LLM_OUTPUT, text="", usage={"output_tokens": 1}))

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
    [child] = g.children(root)
    assert g.result(child) == "[child failed: RuntimeError: child is down]"


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
    assert flow.queue is None


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


def test_a_pool_can_stand_in_for_workers():
    reply, seen = peaked()
    flow = ScriptedFlow(StubLLM(reply), workers=SequentialPool())

    async def query_all():
        return await asyncio.gather(*(flow.llm_query(value) for value in ("a", "b", "c")))

    assert asyncio.run(query_all()) == ["A", "B", "C"]
    assert seen["peak"] == 1


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
