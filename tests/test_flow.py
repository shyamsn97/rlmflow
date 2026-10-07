import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
from helpers import StubLLM, TestRuntime, first_user, trailing_user_text
from helpers import reasoning_repl as block

from rlmflow import (
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
    PromptProfile,
    ReplDead,
    SequentialPool,
    UserQuery,
    VerificationFailure,
    persistence,
    start,
)
from rlmflow.engine import delegation
from rlmflow.engine.delegation import resolve_args
from rlmflow.runtime import LocalRuntime
from rlmflow.tools import tool
from rlmflow.utils.helpers import truncate_output


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


def test_spawn_capable_prompt_explains_delegation_economics():
    prompt = Flow(ScriptedLLM([])).build_system_prompt(
        start("build an app", max_depth=1)
    ).render()

    assert "fresh context" in prompt
    assert "cheaper models" in prompt
    assert "run independent\n  work concurrently" in prompt
    assert "reduce token cost and wall-clock time" in prompt


def test_system_prompt_requires_repl_and_submit_protocol():
    prompt = Flow(ScriptedLLM([])).build_system_prompt(start("write a poem")).render()

    assert "Every response must contain exactly one non-empty fenced ```repl``` block" in prompt
    assert "call `submit(value)` inside the block" in prompt
    assert "Avoid calling `submit(...)` in a block that prints output" in prompt
    assert "Printing and submitting together is allowed" in prompt
    assert "read the observation, then submit in the next turn" in prompt
    assert "the only way to complete the run" in prompt
    assert "submitted as the final answer" not in prompt


def test_prose_before_repl_action_remains_in_following_turn_context():
    llm = ScriptedLLM(
        [
            ("inspect", "I need to inspect first.\n```repl\nprint('observed')\n```"),
            ("observed", block("finish('The final answer.')")),
        ]
    )

    assert Flow(llm).run("inspect this") == "The final answer."
    assert any(
        message == {
            "role": "assistant",
            "content": "I need to inspect first.\n```repl\nprint('observed')\n```",
        }
        for message in llm.calls[1]
    )


def test_error_then_recovery():
    llm = ScriptedLLM(
        [("boom", block("1 / 0")), ("ZeroDivisionError", block("finish('recovered')"))]
    )
    root = start("boom")
    assert Flow(llm).run(root) == "recovered"
    types = [type(n).__name__ for n in root.walk()]
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
    root = start("start")

    assert Flow(llm, execution_guard=guard).run(root) == "safe"
    assert seen == ["forbidden()", "finish('safe')"]
    assert any(
        isinstance(node, ErrorOutput) and node.content == "rejected by execution guard"
        for node in root.walk()
    )


def test_max_iters():
    llm = ScriptedLLM(
        [
            ("last turn", block("finish('best effort')")),
            ("", block("print('again')")),
        ]
    )
    root = start("loop", max_iters=3)
    assert Flow(llm).run(root) == "best effort"
    assert sum(isinstance(n, LLMOutput) for n in root.walk()) == 3


def test_iters_and_budget_are_unbounded_unless_set():
    llm = ScriptedLLM([("loop", block("print('again')")), ("again", block("finish('ok')"))])
    root = start("loop")
    assert root.config.max_iters is None
    assert root.config.max_budget is None
    assert root.config.max_output_length == 20_000
    assert root.config.max_query_chars == 20_000
    assert root.config.resume_on_child_done is True
    bounded = start("bounded", max_iters=30, max_budget=100_000)
    assert bounded.config.max_iters == 30
    assert bounded.config.max_budget == 100_000
    assert Flow(llm).run(root) == "ok"


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
    root = start(
        "read",
        args={"document": Input(document, description="A large source document.")},
    )

    run = root.save(tmp_path / "described-input")
    loaded = AgentStart.load(run)

    assert loaded.resolved_args() == {"document": document}
    assert loaded.argument_descriptions() == {"document": "A large source document."}
    assert len(list((run / "arg_blobs").glob("*.txt"))) == 1


def test_flow_start_carries_the_flow_defaults():
    flow = Flow(
        ScriptedLLM([]),
        root_config=AgentConfig(max_depth=1, max_iters=5, model="fast"),
    )

    root = flow.start("dig in")
    overridden = flow.start(
        "dig in",
        config=AgentConfig(max_depth=1, max_iters=9, model="fast"),
    )

    assert root.content == "dig in"
    assert (root.config.max_depth, root.config.max_iters, root.config.model) == (1, 5, "fast")
    assert overridden.config.max_iters == 9
    assert (overridden.config.max_depth, overridden.config.model) == (1, "fast")


def test_keep_n_messages_keeps_the_tail_and_truncation_summary():
    from rlmflow import TruncationSummary

    llm = ScriptedLLM(
        [
            ("last turn", block("finish('best effort')")),
            ("", block("print('again')")),
        ]
    )
    root = start("loop", max_iters=6, keep_n_messages=3)
    flow = Flow(llm)
    assert flow.run(root) == "best effort"
    assert any(isinstance(node, TruncationSummary) for node in root.transcript())

    messages = flow.build_messages(root.frontier)
    assert messages[0]["role"] == "system"
    assert any("again" in message["content"] for message in messages)
    assert any(TruncationSummary().content in message["content"] for message in messages)


async def stream(flow, root, until):
    return [node async for node in flow.run_streaming(root, until=until)]


def test_streaming_until_idle():
    llm = ScriptedLLM([("look", block("print('peek')")), ("peek", block("finish('finished')"))])
    flow = Flow(llm)
    root = start("look")
    seen = asyncio.run(stream(flow, root, "idle"))
    assert isinstance(seen[-1], ExecOutput) and seen[-1].content.strip() == "peek"
    assert not root.terminal

    rest = asyncio.run(stream(flow, root, "done"))
    assert isinstance(rest[-1], DoneOutput) and root.result() == "finished"


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
    root = start("split this up")
    flow = Flow(llm)
    assert flow.run(root) == "AB"
    assert [child.config.name for child in root.sub_agents] == ["child0", "child1"]
    assert [child.result() for child in root.sub_agents] == ["A", "B"]

    own = {id(node) for node in root.transcript()}
    assert all(node.parent_agent is root for node in root.transcript())
    assert own.isdisjoint(id(node) for child in root.sub_agents for node in child.walk())
    child = root.sub_agents[0]
    assert child.transcript()[0] is child
    assert all(node.parent_agent is child for node in child.transcript())

    # Children branch off the action that launched them, so the parent's frontier
    # stayed on that action for the whole step, and its sequel is the step's output.
    action = next(
        n
        for n in root.transcript()
        if isinstance(n, ExecAction)
        and any(isinstance(child, AgentStart) for child in n.children)
    )
    assert not isinstance(action, AppendChild)
    assert action.children[:2] == root.sub_agents
    assert action.next is action.children[-1]
    assert isinstance(action.next, ExecOutput)
    assert root.frontier.next is None

    streamed = asyncio.run(stream(Flow(llm), start("split this up"), "done"))
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
                block(
                    "finish(f\"{parsed['count']}:{fixtures}\")"
                ),
            ),
            ("critique-answer", block("finish(reconciler + '!')")),
            ("2:F!", block("finish('done')")),
        ]
    )
    root = start("build-dag", max_depth=1)

    assert Flow(llm).run(root) == "done"
    assert [child.config.name for child in root.sub_agents] == [
        "child0",
        "child1",
        "child2",
        "child3",
    ]
    parser, fixtures, reconciler, critic = root.sub_agents
    assert {agent.config.depth for agent in root.sub_agents} == {1}
    assert reconciler.argument_references == (
        ("parsed", parser.id),
        ("fixtures", fixtures.id),
    )
    assert critic.argument_references == (("reconciler", reconciler.id),)
    assert reconciler.args == {
        "parsed": Input(
            {"$rlmflow_agent": parser.id},
            description="Parsed source counts.",
        ),
        "fixtures": {"$rlmflow_agent": fixtures.id},
    }
    assert reconciler.resolved_args() == {"parsed": {"count": 2}, "fixtures": "F"}
    assert reconciler.argument_descriptions() == {"parsed": "Parsed source counts."}
    assert critic.resolved_args() == {"reconciler": "2:F"}

    tasks = [first_user(messages) for messages in llm.calls]
    assert tasks.index("reconcile-inputs") > tasks.index("parse-source")
    assert tasks.index("reconcile-inputs") > tasks.index("inspect-fixtures")
    assert tasks.index("critique-answer") > tasks.index("reconcile-inputs")

    loaded = AgentStart.load(root.save(tmp_path / "dependency-run"))
    loaded_parser, loaded_fixtures, loaded_reconciler, loaded_critic = loaded.sub_agents
    assert loaded_reconciler.argument_references == (
        ("parsed", loaded_parser.id),
        ("fixtures", loaded_fixtures.id),
    )
    assert loaded_critic.argument_references == (("reconciler", loaded_reconciler.id),)
    assert loaded_reconciler.args == reconciler.args


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
    root = start("fan-out", max_depth=2)

    assert Flow(llm).run(root) == "done"
    assert [agent.config.name for agent in root.sub_agents] == [
        "child0",
        "child1",
        "child2",
    ]
    source, solver, reviewer = root.sub_agents
    assert solver.argument_references == (("source", source.id),)
    assert reviewer.argument_references == (("source", source.id),)
    action = next(
        node
        for node in root.transcript()
        if isinstance(node, ExecAction) and node.launch_agent_ids()
    )
    assert set(action.launch_agent_ids().values()) == {
        source.id,
        solver.id,
        reviewer.id,
    }


def test_launch_args_accept_mixed_strings_and_top_level_handles():
    root = start("root", max_depth=1)
    action = ExecAction(code="launch")
    root.append(action)
    source = AgentStart(content="source", config=root.config.child("source"))
    action.append(source)
    action.record_tool_call("launch_subagent", 0, {"$rlmflow_agent": source.id})
    handle = AgentHandle(source.id, "source", "root.source")

    assert resolve_args(
        action,
        {"prompt": "immediate", "evidence": handle},
    ) == {
        "prompt": "immediate",
        "evidence": {"$rlmflow_agent": source.id},
    }


@pytest.mark.parametrize(
    "args",
    [
        {"nested": {"handle": "not-an-agent-ref"}},
        {"nested": ["not", "top-level"]},
        {"number": 3},
    ],
)
def test_launch_args_accept_nested_json_values(args):
    root = start("root")
    action = ExecAction(code="launch")
    root.append(action)

    assert resolve_args(action, args) == args


@pytest.mark.parametrize("value", [{1, 2}, object(), {1: "bad key"}])
def test_launch_args_reject_non_json_values(value):
    root = start("root")
    action = ExecAction(code="launch")
    root.append(action)

    with pytest.raises(TypeError, match="JSON-compatible|non-string"):
        resolve_args(action, {"value": value})


@pytest.mark.parametrize("name", ["bad-name", "class", "task"])
def test_launch_args_require_python_variable_names(name):
    root = start("root")
    action = ExecAction(code="launch")
    root.append(action)

    with pytest.raises(ValueError, match="invalid invocation|'task' is the invocation"):
        resolve_args(action, {name: "value"})


def test_launch_args_reject_unknown_duplicate_and_out_of_scope_handles():
    root = start("root", max_depth=2)
    action = ExecAction(code="launch")
    root.append(action)
    source = AgentStart(content="source", config=root.config.child("source"))
    action.append(source)
    action.record_tool_call("launch_subagent", 0, {"$rlmflow_agent": source.id})
    source_handle = AgentHandle(source.id, "source", "root.source")

    with pytest.raises(ValueError, match="unknown agent"):
        resolve_args(
            action,
            {"missing": AgentHandle("missing", "missing", "root.missing")},
        )
    with pytest.raises(ValueError, match="duplicate agent argument"):
        resolve_args(action, {"a": source_handle, "b": source_handle})

    nested_action = ExecAction(code="nested launch")
    source.append(nested_action)
    with pytest.raises(ValueError, match="outside the current agent"):
        resolve_args(nested_action, {"source": source_handle})


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
    root = start("resume-dag", max_depth=2)
    flow = Flow(llm)

    stopped = asyncio.run(
        stream(
            flow,
            root,
            lambda node, _root: (
                isinstance(node, DoneOutput) and node.parent_agent.config.name == "child0"
            ),
        )
    )
    assert stopped[-1].parent_agent.config.name == "child0"
    assert [child.config.name for child in root.sub_agents] == ["child0", "child1"]
    source, consumer = root.sub_agents
    assert consumer.argument_references == (("source", source.id),)
    assert consumer.blocked is False
    assert consumer.frontier is consumer

    restored = AgentStart.load(root.save(tmp_path / "blocked-run"))
    assert Flow(llm).run(restored) == "resumed"
    restored_source, restored_consumer = restored.sub_agents
    assert restored_consumer.args == {"source": {"$rlmflow_agent": restored_source.id}}
    assert restored_consumer.resolved_args() == {"source": "A"}


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
    root = start(
        "build the requested system",
        args={"brief": "preview\nrequirement A\nrequirement B"},
        max_depth=1,
    )

    assert Flow(llm).run(root) == "done"
    assert sum(isinstance(node, LLMOutput) for node in root.transcript()) == 4
    assert [child.config.name for child in root.sub_agents] == ["child0", "child1"]
    parent_outputs = [
        node.content.strip() for node in root.transcript() if type(node) is ExecOutput
    ]
    assert parent_outputs[:2] == ["preview", "['requirement A', 'requirement B']"]


def test_save_writes_a_run_directory(tmp_path):
    parent = block(
        "child = launch_subagent(task='do a')\n"
        "print(await child)"
    )
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("finish('A')")), ("A", block("finish('AB')"))]
    )
    root = start("split this up", args={"doc": "hi"})
    assert Flow(llm).run(root) == "AB"

    run = root.save(tmp_path / "run")
    graph = json.loads((run / "graph.json").read_text())
    assert "version" not in graph
    assert (graph["node_count"], graph["metadata"]) == (14, {})
    assert graph["nodes"][0]["payload"]["args"] == {"doc": "hi"}
    assert graph["nodes"][0]["parent_id"] is None

    summary = json.loads((run / "latest.json").read_text())
    assert "version" not in summary
    assert summary["agent_ids"] == ["root", "root.child0"]
    assert (summary["root_agent_id"], summary["node_count"]) == ("root", 14)
    assert (summary["finished"], summary["result"]) == (True, "AB")

    action = next(
        n
        for n in root.transcript()
        if isinstance(n, ExecAction) and n.launch_agent_ids()
    )
    children = [node for node in graph["nodes"] if node["parent_id"] == action.id]
    assert [node["type"] for node in children] == ["agent_start", "exec_output"]
    assert children[0]["agent_id"] == "root.child0"

    child = root.sub_agents[0]
    child_dir = run / "agents" / "root" / "child0"
    agent_json = json.loads((child_dir / "agent.json").read_text())
    assert agent_json["agent_id"] == "root.child0"
    assert (agent_json["parent_agent_id"], agent_json["depth"]) == ("root", 1)
    assert agent_json["query"] == "do a"

    lines = (child_dir / "session.jsonl").read_text().splitlines()
    session = [json.loads(line) for line in lines]
    assert [node["id"] for node in session] == [node.id for node in child.transcript()]
    assert all(
        node["agent_id"] == "root.child0" and "children" not in node
        for node in session
    )
    assert json.loads((child_dir / "latest.json").read_text())["id"] == child.frontier.id


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
    root = start("split this up")
    flow = Flow(llm)
    assert flow.run(root) == "AB"

    # seq counts within one agent, so every agent starts over at 0. The tree plus
    # these is the whole run: two children racing changes neither.
    for agent in (root, *root.sub_agents):
        assert [node.seq for node in agent.transcript()] == list(range(len(agent.transcript())))

    # Only executed nodes are timed; a node written inside a step is not a step.
    stepped = [node for node in root.walk() if isinstance(node, (LLMOutput, ExecOutput))]
    assert all(node.timing()["duration_ms"] >= 0 for node in stepped)
    written = [node for node in root.walk() if isinstance(node, AgentStart)]
    assert written and all(node.timing() == {} for node in written)

    turns = [node for node in root.transcript() if isinstance(node, LLMOutput)]
    first, last = root.system_prompt_for(turns[0]), root.system_prompt_for(turns[-1])
    assert first
    assert root.latest_system_prompt() == last
    # The text is stored once per distinct prompt, and turns only keep the id.
    assert sorted(root.system_prompts) == sorted({turn.prompt_id for turn in turns})
    assert turns[0].to_record()["metadata"]["system_prompt"] == turns[0].prompt_id
    assert turns[0].timing()["duration_ms"] >= 0


def test_appending_off_the_frontier_is_refused():
    llm = ScriptedLLM([("count", block("x = 2\nprint(x)")), ("2", block("finish(x * 3)"))])
    root = start("count for me")
    assert Flow(llm).run(root) == 6

    stale = root.transcript()[1]
    with pytest.raises(ValueError, match="frontier"):
        stale.append(UserQuery(content="too late"))
    assert stale.children == [stale.next]


def test_structured_output_is_stored_parsed():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM([("count", block("submit({'n': 41})"))])
    root = start("count", output_schema=schema)
    assert Flow(llm).run(root) == {"n": 41}
    assert root.frontier.to_record()["payload"]["result"] == {"n": 41}


def test_structured_output_still_requires_explicit_submission():
    schema = {"type": "object", "properties": {"n": {"type": "number"}}, "required": ["n"]}
    llm = ScriptedLLM(
        [
            ("count", '{"n": 41}'),
            ("MissingReplError", block("submit({'n': 41})")),
        ]
    )
    root = start("count", output_schema=schema)

    assert Flow(llm).run(root) == {"n": 41}
    assert any(isinstance(node, ErrorOutput) for node in root.walk())


def test_unstructured_submission_keeps_json_values():
    expected = {"n": [41, True, None, "é"]}
    llm = ScriptedLLM([("value", block(f"submit({expected!r})"))])
    root = start("return value")

    assert Flow(llm).run(root) == expected
    assert root.frontier.to_record()["payload"]["result"] == expected


def test_unstructured_submission_rejects_non_json_values():
    llm = ScriptedLLM(
        [
            ("return value", block("submit((1, 2))")),
            ("requires a JSON value", block("submit('recovered')")),
        ]
    )
    root = start("return value")

    assert Flow(llm).run(root) == "recovered"
    assert any(
        "requires a JSON value" in node.content
        for node in root.transcript()
        if isinstance(node, VerificationFailure)
    )


def test_json_submission_crosses_worker_and_persistence_boundary(tmp_path):
    root = start("return value")
    flow = Flow(
        ScriptedLLM([("return value", block("submit({'n': [1, True, None]})"))]),
        runtime=LocalRuntime(repl_timeout=10),
    )

    assert flow.run(root, close_repls=True) == {"n": [1, True, None]}
    loaded = AgentStart.load(root.save(tmp_path / "json-submission"))
    assert loaded.result() == {"n": [1, True, None]}


def test_worker_submission_rejects_non_json_values_and_recovers():
    llm = ScriptedLLM(
        [
            ("return value", block("submit((1, 2))")),
            ("requires a JSON value", block("submit({'ok': True})")),
        ]
    )
    flow = Flow(llm, runtime=LocalRuntime(repl_timeout=10))

    root = start("return value")
    assert flow.run(root, close_repls=True) == {"ok": True}


def test_plain_reply_is_a_protocol_error_and_recovers_through_submission():
    llm = ScriptedLLM(
        [
            ("poem", "The moon writes silver on the sea."),
            ("MissingReplError", block("submit('The moon writes silver on the sea.')")),
        ]
    )
    root = start("write a poem")

    assert Flow(llm).run(root) == "The moon writes silver on the sea."
    assert any(isinstance(node, ErrorOutput) for node in root.walk())


@pytest.mark.parametrize(
    "invalid_reply",
    [
        "```python\nprint('not executed')\n```",
        "```repl\n```",
        "```repl\nprint('unclosed')",
    ],
)
def test_invalid_repl_reply_executes_nothing_and_recovers(invalid_reply):
    llm = ScriptedLLM(
        [
            ("bad response", invalid_reply),
            (
                "MissingReplError",
                block("submit('clean' if 'unclosed' not in globals() else 'dirty')"),
            ),
        ]
    )
    root = start("bad response")

    assert Flow(llm).run(root) == "clean"
    failed_action = next(
        node
        for node in root.walk()
        if isinstance(node, ExecAction) and node.code == ""
    )
    assert failed_action.code == ""


def test_multiple_repl_blocks_execute_neither_block():
    reply = (
        "```repl\nmarker = 'first'\n```\n"
        "```repl\nmarker = 'second'\n```"
    )
    llm = ScriptedLLM(
        [
            ("bad response", reply),
            (
                "MissingReplError",
                block("finish('dirty' if 'marker' in globals() else 'clean')"),
            ),
        ]
    )
    root = start("bad response")

    assert Flow(llm).run(root) == "clean"
    failed_action = next(
        node
        for node in root.walk()
        if isinstance(node, ExecAction) and node.code == ""
    )
    assert failed_action.code == ""


def test_empty_reply_uses_missing_repl_recovery():
    llm = ScriptedLLM(
        [
            ("scan", ""),
            ("MissingReplError", block("finish('done')")),
        ]
    )
    root = start("scan")
    assert Flow(llm).run(root) == "done"

    note = next(node for node in root.walk() if isinstance(node, ErrorOutput)).content
    assert "exactly one non-empty" in note
    assert "calls submit(value)" in note


def test_plain_reply_on_final_turn_does_not_complete_the_run():
    root = start("answer now", max_iters=2)

    llm = ScriptedLLM(
        [("last turn", "unsupported prose"), ("answer now", block("x = 1"))]
    )
    assert Flow(llm).run(root) == (
        "[max_iters exceeded]"
    )
    assert any(isinstance(node, ErrorOutput) for node in root.walk())


def test_save_load_roundtrip(tmp_path):
    parent = block(
        "child = launch_subagent(task='do a')\n"
        "print(await child)"
    )
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("finish('A')")), ("A", block("finish('AB')"))]
    )
    root = start("split this up", args={"doc": "hi"})
    Flow(llm).run(root)

    loaded = AgentStart.load(root.save(tmp_path / "run"))
    assert persistence.to_document(loaded) == persistence.to_document(root)
    assert loaded.stats == root.stats
    assert (loaded.result(), loaded.terminal) == ("AB", True)
    assert [sub.config.path for sub in loaded.sub_agents] == ["root.child0"]
    assert loaded.sub_agents[0].result() == "A"
    assert loaded.save(tmp_path / "again") and (tmp_path / "again" / "graph.json").is_file()


def paused_run(tmp_path, name="run"):
    """A saved run that set a variable and stopped before using it."""
    llm = ScriptedLLM([("count", block("secret = 41\nprint('stored')"))])
    root = start("count for me")
    asyncio.run(stream(Flow(llm), root, "idle"))
    assert isinstance(root.frontier, ExecOutput)
    return AgentStart.load(root.save(tmp_path / name))


def test_resuming_a_saved_run_replays_its_repl(tmp_path):
    loaded = paused_run(tmp_path)
    llm = ScriptedLLM([("stored", block("finish(secret + 1)"))])
    assert Flow(llm).run(loaded) == 42
    assert not [node for node in loaded.walk() if isinstance(node, ErrorOutput)]
    assert len(llm.calls) == 1  # the replay itself asks the model nothing


def test_lazy_restore_waits_until_code_needs_the_repl(tmp_path):
    loaded = paused_run(tmp_path)
    llm = ScriptedLLM([("stored", block("finish(secret + 1)"))])
    flow = Flow(llm, restore="lazy")

    async def stop_before_execution():
        async for _ in flow.run_streaming(
            loaded,
            until=lambda node, _root: isinstance(node, LLMOutput),
        ):
            pass

    asyncio.run(stop_before_execution())
    assert flow.runtime.get(loaded) is None
    assert flow.run(loaded) == 42
    assert flow.runtime.get_var(loaded, "secret") == 41


def terminal_parent_with_unfinished_colocated_child(tmp_path, name):
    """Cold graph whose unfinished child still needs its terminal parent's worker."""
    root = start("parent", max_depth=1, resume_on_child_done=False)
    parent_action = ExecAction(
        code="shared = ['parent']",
        repl_execution_order=1,
    )
    root.append(parent_action)
    child = AgentStart(
        content="child",
        config=root.config.child("child", reuse_worker=True),
    )
    parent_action.append(child)
    parent_action.append(DoneOutput(result="parent done"))

    child_action = ExecAction(
        code="child_state = ['child']",
        repl_execution_order=2,
    )
    child.append(child_action)
    child_action.append(ExecOutput(content="stored"))
    return AgentStart.load(root.save(tmp_path / name))


@pytest.mark.parametrize("restore_mode", ["replay", "lazy"])
def test_restore_rebuilds_terminal_worker_owner_for_unfinished_colocated_child(
    tmp_path,
    restore_mode,
):
    loaded = terminal_parent_with_unfinished_colocated_child(tmp_path, restore_mode)
    child = loaded.sub_agents[0]
    flow = Flow(
        ScriptedLLM([("stored", block("finish(task)"))]),
        runtime=LocalRuntime(repl_timeout=10),
        restore=restore_mode,
    )

    assert flow.run(loaded, close_repls=True) == "parent done"
    assert child.result() == "child"


def test_replay_keeps_colocated_agent_namespaces_isolated(
    tmp_path,
):
    root = start("parent", max_depth=1)
    parent_action = ExecAction(
        code="shared = ['parent']",
        repl_execution_order=1,
    )
    root.append(parent_action)
    child = AgentStart(
        content="child",
        config=root.config.child("child", reuse_worker=True),
    )
    parent_action.append(child)
    parent_action.append(ExecOutput(content="child done"))
    child_action = ExecAction(
        code="child_state = ['child']",
        repl_execution_order=2,
    )
    child.append(child_action)
    child_action.append(DoneOutput(result="child done"))
    loaded = AgentStart.load(root.save(tmp_path / "terminal-shared-child"))

    flow = Flow(
        ScriptedLLM([("child done", block("finish(shared)"))]),
        runtime=LocalRuntime(repl_timeout=10),
    )

    assert flow.run(loaded, close_repls=True) == ["parent"]


def test_replay_reads_recorded_child_answers_instead_of_relaunching(tmp_path):
    parent = block(
        "child = launch_subagent(task='add up')\n"
        "answer = await child\nprint(answer)"
    )
    llm = ScriptedLLM([("delegate", parent), ("add up", block("finish('20')"))])
    root = start("delegate", max_depth=2)

    def parent_turn_ended(node, root):
        return type(node) is ExecOutput and node.parent_agent is root

    asyncio.run(stream(Flow(llm), root, parent_turn_ended))
    loaded = AgentStart.load(root.save(tmp_path / "run"))

    resumed = ScriptedLLM([("20", block("finish('parent saw ' + answer)"))])
    assert Flow(resumed).run(loaded) == "parent saw 20"
    # The child ran once, when it was recorded: no second launch, no second model call.
    assert [child.config.name for child in loaded.sub_agents] == ["child0"]
    assert len(resumed.calls) == 1


def test_replay_restores_llm_query_result_without_another_model_call(tmp_path):
    setup = ScriptedLLM(
        [
            ("delegate", block("answer = await llm_query('extract')\nprint(answer)")),
            ("extract", "recorded"),
        ]
    )
    root = start("delegate")

    def query_finished(node, _root):
        return type(node) is ExecOutput

    setup_flow = Flow(
        setup,
        runtime=LocalRuntime(repl_timeout=10),
        use_llm_query=True,
    )
    asyncio.run(stream(setup_flow, root, query_finished))
    asyncio.run(setup_flow.aclose())
    loaded = AgentStart.load(root.save(tmp_path / "llm-query-run"))
    resumed = ScriptedLLM([("recorded", block("finish(answer)"))])

    assert Flow(
        resumed,
        runtime=LocalRuntime(repl_timeout=10),
        use_llm_query=True,
    ).run(loaded, close_repls=True) == "recorded"
    assert len(resumed.calls) == 1


def test_append_child_runs_a_prebuilt_subtree():
    root = start("parent", max_depth=1)
    branch = start("child")
    root.append_child(branch, name="branch0")
    action = root.frontier

    flow = Flow(
        ScriptedLLM(
            [
                ("child", block("finish('finished')")),
                ("finished", block("finish('finished')")),
            ]
        )
    )

    assert flow.run(root) == "finished"
    assert isinstance(action, AppendChild)
    assert action.child_agents == [branch]
    assert branch.config.path == "root.branch0"


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
    root = start("prepare", max_depth=1)

    async def main():
        async for _node in flow.run_streaming(root, until="idle"):
            pass

        anchor = root.frontier
        child_a = start("child a")
        action_a = ExecAction(code="value = 'A'")
        child_a.append(action_a)
        action_a.append(ExecOutput(content="child a"))
        child_b = start("child b")
        action_b = ExecAction(code="value = 'B'")
        child_b.append(action_b)
        action_b.append(ExecOutput(content="child b"))
        anchor.append_child(child_a, name="a")
        anchor.append_child(child_b, name="b")

        async for _node in flow.run_streaming(root):
            pass

    asyncio.run(main())

    action = next(node for node in root.transcript() if isinstance(node, AppendChild))
    assert root.result() == "AB"
    assert [child.config.name for child in action.child_agents] == ["a", "b"]
    assert action.code == (
        "_handles = [\n"
        "    launch_subagent(task=''),\n"
        "    launch_subagent(task=''),\n"
        "]\n"
        "print(await asyncio.gather(*_handles))"
    )


def test_resume_uses_an_unfinished_subtree_already_on_the_launch_action():
    root = start("parent", max_depth=1)
    child = start("child")
    root.append_child(child, name="kid")

    llm = ScriptedLLM(
        [
            ("finished child", block("finish('parent saw finished child')")),
            ("child", block("finish('finished child')")),
        ]
    )
    assert Flow(llm).run(root) == "parent saw finished child"
    assert child.result() == "finished child"
    assert len(llm.calls) == 2


def test_append_child_type_survives_save_load(tmp_path):
    root = start("parent")
    branch = start("child")
    root.append_child(branch, name="branch0")
    action = root.frontier
    branch.append(DoneOutput(result="finished"))
    action.append(ExecOutput(content="['finished']"))

    loaded = AgentStart.load(root.save(tmp_path / "run"))
    loaded_action = next(node for node in loaded.transcript() if isinstance(node, AppendChild))

    assert loaded_action.code == action.code
    assert [child.config.name for child in loaded_action.child_agents] == ["branch0"]


def test_replaying_agents_can_skip_their_own_unrepeatable_work(tmp_path):
    trace: list[str] = []

    @tool("Record a live side effect.", proxy=True)
    def log(value):
        trace.append(value)

    code = block("if ENV['RLMFLOW_REPLAY'] == '0':\n    log('live')\nprint('set up')")
    flow = Flow(ScriptedLLM([("count", code)]))
    flow.inject("log", log)
    root = start("count for me")
    asyncio.run(stream(flow, root, "idle"))
    assert trace == ["live"]

    loaded = AgentStart.load(root.save(tmp_path / "run"))
    resumed = Flow(ScriptedLLM([("set up", block("finish('ok')"))]))
    resumed.inject("log", log)
    assert resumed.run(loaded) == "ok"
    assert trace == ["live"]  # the replay saw RLMFLOW_REPLAY=1 and skipped it


def test_a_finished_run_resumes_to_its_recorded_answer(tmp_path):
    llm = ScriptedLLM([("count", block("finish('six')"))])
    root = start("count for me")
    assert Flow(llm).run(root) == "six"

    loaded = AgentStart.load(root.save(tmp_path / "run"))
    idle = ScriptedLLM([])
    assert Flow(idle).run(loaded) == "six"
    assert idle.calls == []  # nothing left to step


def test_a_dead_repl_is_an_observation_and_is_replaced():
    class OneDeadRepl(LocalRuntime):
        """Whatever the first agent gets, its REPL dies the way a sandbox does."""

        opened = 0

        def open(self, agent):
            self.opened += 1
            repl = super().open(agent)
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
    root = start("count for me")
    assert Flow(llm, runtime=runtime).run(root) == "recovered"

    errors = [node for node in root.walk() if isinstance(node, ErrorOutput)]
    assert len(errors) == 1 and "REPL execution failed" in errors[0].content
    assert runtime.opened == 2  # the dead one was dropped, not handed out again

    # A dead REPL reads differently from code that raised: the agent is told its
    # namespace went with it, and the record says which kind of failure it was.
    assert errors[0].error == "repl"
    assert errors[0].content.endswith(ReplDead().content)
    assert errors[0].to_record()["payload"]["error"] == "repl"


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
    root = start("delegate", max_depth=2)
    assert Flow(llm).run(root) == "done"
    assert [child.config.name for child in root.sub_agents] == ["child0", "child1"]


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
    root = start("delegate", max_depth=1)
    assert Flow(llm).run(root) == "complete"
    assert [child.config.name for child in root.sub_agents] == ["worker-events-01"]


def test_saving_a_smaller_tree_prunes_stale_agents(tmp_path):
    parent = block(
        "child = launch_subagent(task='do a')\n"
        "print(await child)"
    )
    llm = ScriptedLLM(
        [("split", parent), ("do a", block("finish('A')")), ("A", block("finish('AB')"))]
    )
    root = start("split this up")
    Flow(llm).run(root)

    run = root.save(tmp_path / "run")
    assert (run / "agents" / "root" / "child0" / "agent.json").is_file()

    root.sub_agents.clear()
    root.save(run)
    assert not (run / "agents" / "root" / "child0").exists()


def test_max_depth_refusal():
    llm = ScriptedLLM(
        [
            ("deep", block("print(launch_subagent(task='x'))")),
            ("is not defined", block("finish('stopped')")),
        ]
    )
    root = start("go deep", max_depth=0)
    flow = Flow(llm)
    assert "launch_subagent" not in flow.build_tools(root)
    assert flow.run(root) == "stopped"
    assert root.sub_agents == []


def test_grandchildren():
    llm = ScriptedLLM(
        [
            (
                "plan the work",
                block(
                    "mid = launch_subagent(task='go mid')\n"
                    "print(await mid)"
                ),
            ),
            (
                "go mid",
                block(
                    "leaf = launch_subagent(task='go leaf')\n"
                    "print(await leaf)"
                ),
            ),
            ("go leaf", block("finish('leaf')")),
            ("leaf", block("finish('mid')")),
            ("mid", block("finish('root')")),
        ]
    )
    root = start("plan the work", max_depth=2)
    assert Flow(llm).run(root) == "root"
    mid = root.sub_agents[0]
    assert [agent.config.path for agent in (mid, mid.sub_agents[0])] == [
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
    calls = ", ".join(
        f"launch_subagent(task='child {i}')" for i in range(3)
    )
    llm = BarrierLLM(
        3,
        [
            (
                "fan out",
                block(
                    f"handles = [{calls}]\n"
                    "print(await asyncio.gather(*handles))"
                ),
            ),
            *[(f"child {i}", block(f"finish({i})")) for i in range(3)],
            ("[0, 1, 2]", block("finish('all')")),
        ],
    )
    root = start("fan out", max_depth=1)

    async def main():
        # Sequential children would never clear the barrier, so this would hang.
        return await asyncio.wait_for(Flow(llm).arun(root), timeout=5)

    assert asyncio.run(main()) == "all"
    assert [child.result() for child in root.sub_agents] == [0, 1, 2]


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
            return block(
                "child = launch_subagent(task='slow child')\n"
                "finish(child.status)"
            )

    flow = Flow(BackgroundLLM(), workers=2)
    root = flow.start(
        "launch without joining",
        config=AgentConfig(max_depth=1, resume_on_child_done=False),
    )

    async def run():
        observed = []
        async for node in flow.run_streaming(root):
            observed.append(node)
            if isinstance(node, DoneOutput) and node.parent_agent is root:
                assert root.result() == "running"
                assert not root.sub_agents[0].terminal
                release_child.set()
        await flow.aclose()
        return observed

    observed = asyncio.run(run())
    child = root.sub_agents[0]
    assert child.result() == "child finished"
    root_done = next(
        i
        for i, node in enumerate(observed)
        if node.parent_agent is root and isinstance(node, DoneOutput)
    )
    child_done = next(
        i
        for i, node in enumerate(observed)
        if node.parent_agent is child and isinstance(node, DoneOutput)
    )
    assert root_done < child_done
    assert child.id not in root.retrieved_agent_ids()


def test_child_handle_waits_on_a_later_turn():
    wait = block("finish(await later)")
    llm = ScriptedLLM(
        [
            (
                "start later",
                block(
                    "later = launch_subagent(task='child task')\n"
                    "print('queued', later.name)"
                ),
            ),
            ("child task", block("finish('child answer')")),
            ("queued child0", wait),
            ("[handoff root.child0]", wait),
        ]
    )
    root = start("start later", max_depth=1)

    assert Flow(llm).run(root) == "child answer"
    assert len(root.sub_agents) == 1
    assert root.sub_agents[0].result() == "child answer"


def test_completed_child_result_access_satisfies_finish_gate():
    llm = ScriptedLLM(
        [
            (
                "start child",
                block(
                    "child = launch_subagent(task='child task')\n"
                    "print('queued')"
                ),
            ),
            ("child task", block("finish(42)")),
            ("result ready", block("finish(child.result())")),
            ("queued", block("print(child.status)")),
            ("running", block("print(child.status)")),
        ]
    )
    root = start("start child", max_depth=1)

    assert Flow(llm).run(root) == 42
    child = root.sub_agents[0]
    assert child.id in root.retrieved_agent_ids()
    assert any(
        child.id in node.retrieved_agent_ids
        for node in root.transcript()
        if isinstance(node, ExecAction)
    )


def test_launch_subagent_binds_task_and_json_arguments_directly():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block(
                    "child = launch_subagent(task='child', wait=False)\n"
                    "print(await child)"
                ),
            ),
            ("child:False", block("finish('recovered')")),
            ("child", block("finish(task + ':' + str(wait))")),
        ]
    )
    root = start("delegate", max_depth=1)

    assert Flow(llm).run(root) == "recovered"
    assert root.sub_agents[0].resolved_args() == {"wait": False}


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
    root = start(
        "delegate",
        args={"rows": [{"label": "private-marker-7"}]},
        max_depth=1,
    )

    assert Flow(llm).run(root) == "private-marker-7"
    child = root.sub_agents[0]
    assert child.resolved_args() == {"rows": [{"label": "private-marker-7"}]}
    assert child.argument_descriptions() == {"rows": "Records with label fields."}
    assert "Records with label fields." in llm.child_system_prompt
    assert "private-marker-7" not in llm.child_system_prompt


def test_agents_result_matches_a_collected_child():
    llm = ScriptedLLM(
        [
            (
                "inspect waited",
                block(
                    "waited = launch_subagent(task='child task')\n"
                    "print(await waited)"
                ),
            ),
            ("child task", block("finish('child answer')")),
            (
                "child answer",
                block("finish(await AGENTS.get(waited).wait_for_result())"),
            ),
        ]
    )
    flow = Flow(llm, use_agent_tree=True)
    root = flow.start("inspect waited", config=AgentConfig(max_depth=1))

    assert flow.run(root) == "child answer"


def test_break_cancels_the_run():
    llm = ScriptedLLM([("", block("print('again')"))])
    flow = Flow(llm)

    async def main():
        stream = flow.run_streaming(start("loop forever"))
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

    root = start("anything")
    try:
        Flow(BrokenLLM()).run(root)
    except RuntimeError as exc:
        assert str(exc) == "model is down"
    else:
        raise AssertionError("expected the step failure to propagate")
    assert root.result() == "[run failed: RuntimeError: model is down]"


def test_custom_tool_is_injected():
    @tool("Upper-case a string.")
    def shout(text: str) -> str:
        return text.upper()

    llm = ScriptedLLM([("", block("finish(shout('hi'))"))])
    flow = Flow(llm, tools=[shout])
    root = start("say it")
    assert flow.run(root) == "HI"


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
    root = start("measure it", max_depth=1)
    assert Flow(llm).run(root) == "42"
    assert root.sub_agents[0].config.output_schema == schema


def test_child_inherits_child_output_schema_when_launch_omits_it():
    schema = {
        "type": "object",
        "properties": {"n": {"type": "number"}},
        "required": ["n"],
    }
    parent = block(
        "child = launch_subagent(task='count')\n"
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
    root = start("measure it", max_depth=1, child_output_schema=schema)
    assert Flow(llm).run(root) == "42"
    child = root.sub_agents[0]
    assert root.config.output_schema is None
    assert child.config.output_schema == schema
    assert child.config.output_schema is not root.config.child_output_schema
    assert child.config.child_output_schema == schema


def test_launch_output_schema_overrides_child_output_schema():
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
    root = start("measure it", max_depth=1, child_output_schema=inherited)
    assert Flow(llm).run(root) == "ok"
    child = root.sub_agents[0]
    assert child.config.output_schema == schema
    assert child.config.child_output_schema == inherited


def test_named_model_routes_to_another_client():
    default = ScriptedLLM([("", block("finish('wrong client')"))])
    fast = ScriptedLLM([("", block("finish('fast client')"))])
    root = start("who answers", model="fast")
    assert Flow(default, llm_clients={"fast": fast}).run(root) == "fast client"
    assert default.calls == []


def test_flow_owned_delegation_model_routes_children():
    default = ScriptedLLM(
        [
            (
                "delegate",
                block(
                    "child = launch_subagent(task='do a')\n"
                    "print(await child)"
                ),
            ),
            ("cheap result", block("finish('parent saw cheap result')")),
        ]
    )
    fast = ScriptedLLM([("do a", block("finish('cheap result')"))])
    flow = Flow(
        default,
        root_config=AgentConfig(max_depth=1),
        llm_clients={"fast": fast},
        delegation_model="fast",
    )

    root = flow.start("delegate")

    assert flow.run(root) == "parent saw cheap result"
    assert root.sub_agents[0].config.model == "fast"
    assert len(fast.calls) == 1


def test_flow_owned_delegation_profile_and_placement_route_children():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block(
                    "child = launch_subagent(task='do a')\n"
                    "print(await child)"
                ),
            ),
            ("do a", block("finish('child')")),
            ("child", block("finish('done')")),
        ]
    )
    flow = Flow(
        llm,
        root_config=AgentConfig(max_depth=1),
        prompt_profiles={"worker": PromptProfile()},
        delegation_prompt_profile="worker",
        delegation_reuse_worker=True,
        runtime=TestRuntime(),
    )

    root = flow.start("delegate")

    assert flow.run(root) == "done"
    assert root.sub_agents[0].config.prompt_profile == "worker"
    assert root.sub_agents[0].config.reuse_worker is True


def test_child_launch_inherits_parent_model_by_default():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block(
                    "child = launch_subagent(task='do a')\n"
                    "print(await child)"
                ),
            ),
            ("do a", block("finish('child')")),
            ("child", block("finish('done')")),
        ]
    )
    root = start("delegate", max_depth=1)

    assert Flow(llm).run(root) == "done"
    assert root.sub_agents[0].config.model == "default"


def test_launch_model_keyword_is_ordinary_child_argument_not_policy():
    llm = ScriptedLLM(
        [
            (
                "delegate",
                block(
                    "child = launch_subagent(task='do a', model='input value')\n"
                    "print(await child)"
                ),
            ),
            ("do a", block("finish(model)")),
            ("input value", block("finish('done')")),
        ]
    )
    root = start("delegate", max_depth=1)

    assert Flow(llm).run(root) == "done"
    assert root.sub_agents[0].config.model == "default"


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
    root = start("delegate this", max_depth=1, max_query_chars=4)
    assert Flow(llm).run(root) == "stopped"
    assert root.sub_agents == []


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
    root = start("spend most of it", max_budget=20)

    assert Flow(llm).run(root) == "best guess"
    assert sum(isinstance(node, FinalQuery) for node in root.transcript()) == 1


def test_the_budget_gate_never_discards_an_answer_already_paid_for():
    # The gate used to fire on any node, so a reply that finished the task could be
    # dropped between generation and the exec that lands it.
    llm = ScriptedLLM([("", block("finish('landed')"))])
    llm.last_usage = SimpleNamespace(input_tokens=40, output_tokens=40)
    root = start("answer immediately", max_budget=10)

    assert Flow(llm).run(root) == "landed"


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

    root = start("parent", max_depth=1, max_budget=10)
    result = Flow(ConcurrentBudgetLLM(), runtime=TestRuntime()).run(root)

    assert result == "root-final"
    assert root.usage.total > root.config.max_budget
    assert len(root.sub_agents) == 2
    for agent in (root, *root.sub_agents):
        assert sum(isinstance(node, FinalQuery) for node in agent.transcript()) == 1


def test_exhausted_budget_rejects_new_children():
    flow = Flow(ScriptedLLM([]))
    root = start("parent", max_depth=1, max_budget=1)
    reply = LLMOutput(usage=LLMUsage(output_tokens=1))
    root.append(reply)
    action = ExecAction(code="pass")
    reply.append(action)

    with pytest.raises(RuntimeError, match="cannot launch.*max_budget"):
        delegation.new_child(
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
                block(
                    "child = launch_subagent(task='boom')\n"
                    "print(await child)"
                ),
            ),
            ("boom", RuntimeError("child is down")),
            ("child is down", block("finish('recovered')")),
        ]
    )
    root = start("delegate this", max_depth=1)
    assert Flow(llm).run(root) == "recovered"


def test_aclose_drops_the_work_and_the_repls():
    class ClosingLLM(ScriptedLLM):
        closed = False

        async def aclose(self):
            self.closed = True

    llm = ClosingLLM([("count", block("finish('six')"))])
    flow = Flow(llm)

    async def run_then_close():
        await flow.arun(start("count for me"))
        assert flow.repls
        await flow.aclose()

    asyncio.run(run_then_close())
    assert llm.closed
    assert not flow.repls
    assert flow.queue is None


def test_one_flow_rejects_competing_stream_drivers():
    llm = ScriptedLLM([("one", block("finish('first')")), ("two", block("finish('second')"))])
    flow = Flow(llm)
    first, second = start("one"), start("two")

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
