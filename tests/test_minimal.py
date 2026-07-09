import asyncio
import json
import os
import sys
from pathlib import Path

from rflow.minimal import (
    AddChild,
    AppendNode,
    AsyncPool,
    ExecAction,
    ExecOutput,
    Graph,
    GraphCheckpoint,
    GraphCreated,
    LLMOutput,
    LLMUsage,
    LocalRuntime,
    RemoveNode,
    RemoveChild,
    ReplaceNode,
    SequentialPool,
    SubprocessRuntime,
    UserQuery,
    apply_graph_action,
    code_block,
    find_code_blocks,
    Flow,
    group_flows,
    tool,
)
from rflow.minimal.runtime.protocol import (
    CapabilitiesRequest,
    PingRequest,
    ProxyCall,
    ReplResponse,
    RunRequest,
    parse_client_message,
    parse_request,
)
from rflow.minimal.runtime.repl_client import ReplClient
from rflow.minimal.runtime.repl import DoneSignal
from rflow.minimal.utils import repl_key
from rflow.minimal.view.rendering import LiveTreeRenderer, render_tree
from rflow.minimal.runtime import PopenConnection, build_docker_argv


class StubLLM:
    def __init__(self, fn):
        self.fn = fn

    def chat(self, messages):
        return self.fn(messages)


class UsageLLM:
    def __init__(self, reply, input_tokens=0, output_tokens=0):
        self.reply = reply
        self.last_usage = None
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def chat(self, messages):
        self.last_usage = LLMUsage(self.input_tokens, self.output_tokens)
        return self.reply(messages) if callable(self.reply) else self.reply


def first_user(messages):
    return next(m["content"] for m in messages if m["role"] == "user")


def assert_llm_outputs_are_followed_by_exec_actions(graph):
    for agent in graph.walk():
        for index, node in enumerate(agent.nodes):
            if node.type == "llm_output":
                assert index + 1 < len(agent.nodes)
                assert agent.nodes[index + 1].type == "exec_action"


def test_minimal_code_block_extraction_handles_runtime_edge_cases():
    assert code_block("```repl   \ndone('ok')```\ntrailing") == "done('ok')"
    assert code_block("```python\nx = 1\n```") == "x = 1"

    text = '```repl\ns = """\n```bash\nls\n```\n"""\nprint(s)\n```'
    assert "```bash" in code_block(text)
    assert find_code_blocks("no block") == []


def test_minimal_graph_actions_append_replace_and_remove_nodes():
    graph = apply_graph_action(
        None,
        GraphCreated(type="graph_created", graph=Graph(query="q")),
    )
    graph = apply_graph_action(
        graph,
        AppendNode(
            type="append_node",
            agent_id="root",
            node_type="user_query",
            node=UserQuery(content="q"),
        ),
    )
    first = graph.nodes[0]
    graph = apply_graph_action(
        graph,
        ReplaceNode(
            type="replace_node",
            agent_id="root",
            node_id=first.id,
            node_type="llm_output",
            node=LLMOutput(content="replacement"),
        ),
    )
    assert [node.type for node in graph.nodes] == ["llm_output"]
    assert graph.nodes[0].seq == 0

    graph = apply_graph_action(
        graph,
        RemoveNode(
            type="remove_node",
            agent_id="root",
            node_id=graph.nodes[0].id,
        ),
    )
    assert graph.nodes == []


def test_minimal_graph_actions_add_and_remove_child_graph():
    graph = apply_graph_action(
        None,
        GraphCreated(type="graph_created", graph=Graph(query="q")),
    )
    child = Graph(agent_id="root.child", query="child", parent_agent_id="root", depth=1)

    graph = apply_graph_action(
        graph,
        AddChild(type="add_child", parent_agent_id="root", child=child),
    )
    assert "root.child" in graph

    graph = apply_graph_action(
        graph,
        RemoveChild(
            type="remove_child",
            parent_agent_id="root",
            child_agent_id="root.child",
        ),
    )
    assert "root.child" not in graph


def test_minimal_graph_operation_helpers_edit_transcripts():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")
    flow.start(graph)

    injected = graph.inject("please verify")

    assert isinstance(injected, AppendNode)
    assert [node.content for node in graph.nodes] == ["q", "please verify"]

    first_id = graph.nodes[0].id
    graph.replace_node(first_id, UserQuery(content="replacement"))

    assert [node.content for node in graph.nodes] == ["replacement", "please verify"]
    assert [node.seq for node in graph.nodes] == [0, 1]

    graph.rewind(graph.nodes[1].id)

    assert [node.content for node in graph.nodes] == ["replacement"]


def test_minimal_graph_fork_creates_independent_graph_branch():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")
    flow.start(graph)
    graph.inject("branch point")
    child = Graph(
        agent_id="root.child",
        graph_id=graph.graph_id,
        query="child",
        depth=1,
        parent_agent_id="root",
    )
    flow.apply_action(
        graph,
        AddChild(type="add_child", parent_agent_id="root", child=child),
    )

    branch = graph.fork(
        from_node_id=graph.nodes[1].id,
        keep_children=False,
    )

    assert branch is not graph
    assert branch.graph_id != graph.graph_id
    assert all(agent.graph_id == branch.graph_id for agent in branch.walk())
    assert [node.content for node in branch.nodes] == ["q"]
    assert branch.children == {}
    assert "root.child" in graph
    assert [node.content for node in graph.nodes] == ["q", "branch point"]


def test_minimal_graph_fork_can_share_session_graph_id_for_repl_reuse():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")
    flow.repl_for(graph)

    branch = graph.fork(session="shared")

    assert branch is not graph
    assert branch.graph_id == graph.graph_id
    assert all(agent.graph_id == graph.graph_id for agent in branch.walk())
    assert flow.repl_for(branch) is flow.repl_for(graph)


def test_minimal_graph_remove_child_keeps_repl_cleanup_explicit():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")
    child = Graph(
        agent_id="root.child",
        graph_id=graph.graph_id,
        query="child",
        depth=1,
        parent_agent_id="root",
    )
    flow.apply_action(
        graph,
        AddChild(type="add_child", parent_agent_id="root", child=child),
    )
    flow.repl_for(child)

    assert (graph.graph_id, "root.child") in flow.repls

    removed = graph.remove_child("root.child")

    assert isinstance(removed, RemoveChild)
    assert "root.child" not in graph
    assert (graph.graph_id, "root.child") in flow.repls

    flow.close_repl(child)

    assert (graph.graph_id, "root.child") not in flow.repls


def test_minimal_flow_launch_subagents_is_public():
    def reply(messages):
        task = first_user(messages)
        return f'```repl\ndone("{task} done")\n```'

    flow = Flow(StubLLM(reply), max_depth=1)
    graph = Graph(query="parent")
    flow.start(graph)
    repl = flow.repl_for(graph)
    launch_subagents = flow.launch_subagents(graph, repl)

    results = asyncio.run(
        launch_subagents(
            [
                {"name": "a", "query": "task a"},
                {"name": "b", "query": "task b"},
            ],
        )
    )

    assert results == ["task a done", "task b done"]
    assert graph["root.a"].result() == "task a done"
    assert graph["root.b"].result() == "task b done"
    assert graph.nodes[-2].type == "supervising_output"
    assert graph.nodes[-1].type == "resume_action"


def test_minimal_flow_done():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    assert flow.run(graph) == "ok"
    assert [node.type for node in graph.nodes] == [
        "user_query",
        "llm_output",
        "exec_action",
        "done_output",
    ]
    assert_llm_outputs_are_followed_by_exec_actions(graph)


def test_minimal_run_streaming_seeds_graph_inputs_and_output_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    flow = Flow(
        StubLLM(
            lambda _messages: (
                "```repl\n"
                "print(INPUTS['context'])\n"
                "done({'answer': INPUTS['context']})\n"
                "```"
            )
        )
    )
    graph = Graph(query="use the provided context")

    async def drive():
        async for _event in flow.run_streaming(
            graph,
            inputs={"context": "from-inputs"},
            output_schema=schema,
        ):
            pass

    asyncio.run(drive())

    assert graph.inputs == {"context": "from-inputs"}
    assert graph.output_schema == schema
    assert graph.nodes[0].type == "user_query"
    assert graph.nodes[0].content == "use the provided context"
    assert json.loads(graph.result()) == {"answer": "from-inputs"}
    assert "from-inputs" in graph.nodes[-1].output


def test_minimal_flow_delegates_child_and_keeps_graph_shape():
    def reply(messages):
        task = first_user(messages)
        if task == "child task":
            return '```repl\ndone("c")\n```'
        return (
            "```repl\n"
            'results = await launch_subagents([{"name": "child", "query": "child task"}])\n'
            'done("p:" + results[0])\n'
            "```"
        )

    flow = Flow(StubLLM(reply), max_depth=1)
    graph = Graph(query="parent")

    assert flow.run(graph) == "p:c"
    assert [node.type for node in graph.nodes] == [
        "user_query",
        "llm_output",
        "exec_action",
        "supervising_output",
        "resume_action",
        "done_output",
    ]
    assert [node.type for node in graph["root.child"].nodes] == [
        "user_query",
        "llm_output",
        "exec_action",
        "done_output",
    ]
    assert_llm_outputs_are_followed_by_exec_actions(graph)


def test_minimal_async_pool_bounds_child_fanout():
    active = 0
    max_seen = 0

    async def reply(messages):
        nonlocal active, max_seen
        task = first_user(messages)
        if task == "parent":
            specs = [{"name": f"c{i}", "query": f"child {i}"} for i in range(4)]
            return f"```repl\nresults = await launch_subagents({specs!r})\ndone(','.join(results))\n```"
        active += 1
        max_seen = max(max_seen, active)
        try:
            await asyncio.sleep(0.01)
            return f'```repl\ndone("{task}")\n```'
        finally:
            active -= 1

    flow = Flow(StubLLM(reply), max_depth=1, max_concurrency=2)
    graph = Graph(query="parent")

    assert flow.run(graph) == "child 0,child 1,child 2,child 3"
    assert max_seen == 2


def test_minimal_sequential_pool_runs_child_fanout_one_at_a_time():
    active = 0
    max_seen = 0

    async def reply(messages):
        nonlocal active, max_seen
        task = first_user(messages)
        if task == "parent":
            specs = [{"name": f"c{i}", "query": f"child {i}"} for i in range(3)]
            return f"```repl\nresults = await launch_subagents({specs!r})\ndone(','.join(results))\n```"
        active += 1
        max_seen = max(max_seen, active)
        try:
            await asyncio.sleep(0.01)
            return f'```repl\ndone("{task}")\n```'
        finally:
            active -= 1

    flow = Flow(StubLLM(reply), max_depth=1, pool=SequentialPool())
    graph = Graph(query="parent")

    assert flow.run(graph) == "child 0,child 1,child 2"
    assert max_seen == 1


def test_minimal_llm_query_batched_uses_shared_pool():
    active = 0
    max_seen = 0

    async def reply(messages):
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        try:
            await asyncio.sleep(0.01)
            return messages[-1]["content"].upper()
        finally:
            active -= 1

    flow = Flow(StubLLM(reply), max_concurrency=2)

    async def collect():
        return await flow.llm_query_batched(["a", "b", "c", "d"])

    assert asyncio.run(collect()) == ["A", "B", "C", "D"]
    assert max_seen == 2


def test_minimal_llm_query_batched_can_be_used_as_opt_in_tool():
    def reply(messages):
        if len(messages) == 1:
            return messages[0]["content"].upper()
        return (
            "```repl\n"
            'results = await llm_query_batched(["a", "b"])\n'
            'done("|".join(results))\n'
            "```"
        )

    flow = Flow(StubLLM(reply), use_llm_query=True)
    graph = Graph(query="q")

    assert flow.run(graph) == "A|B"


def test_minimal_child_specs_can_route_to_named_model():
    default = StubLLM(
        lambda _messages: (
            "```repl\n"
            'results = await launch_subagents([{"name": "fast", "query": "child", "model": "fast"}])\n'
            'done(results[0])\n'
            "```"
        )
    )
    fast = StubLLM(lambda _messages: '```repl\ndone("fast child")\n```')
    flow = Flow(default, llm_clients={"fast": fast}, max_depth=1)
    graph = Graph(query="parent")

    assert flow.run(graph) == "fast child"
    assert graph["root.fast"].model == "fast"


def test_minimal_usage_accounting_is_stored_in_llm_output_metadata():
    flow = Flow(UsageLLM('```repl\ndone("ok")\n```', input_tokens=3, output_tokens=2))
    graph = Graph(query="q")

    assert flow.run(graph) == "ok"
    llm_output = next(node for node in graph.nodes if isinstance(node, LLMOutput))
    assert llm_output.metadata == {
        "model": "default",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    assert graph.usage() == LLMUsage(input_tokens=3, output_tokens=2)
    assert graph.tokens() == (3, 2)
    assert graph.total_tokens() == 5


def test_minimal_llm_query_batched_routes_model():
    fast = UsageLLM(lambda messages: messages[-1]["content"].upper(), 5, 7)
    flow = Flow(StubLLM(lambda _messages: "unused"), llm_clients={"fast": fast})

    async def collect():
        return await flow.llm_query_batched(["a", "b"], model="fast")

    assert asyncio.run(collect()) == ["A", "B"]


def test_minimal_root_output_schema_validates_done_and_prompts_schema():
    seen = {}
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }

    def reply(messages):
        seen["system"] = messages[0]["content"]
        return '```repl\ndone({"answer": 42})\n```'

    flow = Flow(StubLLM(reply))
    graph = Graph(query="q")

    assert json.loads(flow.run(graph, output_schema=schema)) == {"answer": 42}
    assert graph.output_schema == schema
    assert "JSON Schema" in seen["system"]
    assert '"answer"' in seen["system"]


def test_minimal_output_schema_prompted_and_enforced_regardless_of_flag():
    # An explicit output_schema is a hard requirement, independent of the
    # enable_structured_output capability flag.
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    seen = {}

    def reply(messages):
        seen["system"] = messages[0]["content"]
        return '```repl\ndone({"answer": 5})\n```'

    flow = Flow(StubLLM(reply), enable_structured_output=False)
    graph = Graph(query="q", output_schema=schema)

    assert json.loads(flow.run(graph)) == {"answer": 5}
    assert "This run requires structured output" in seen["system"]


def test_minimal_structured_output_option_teaches_subagent_schema_when_enabled():
    flow = Flow(StubLLM(lambda _messages: "unused"), max_depth=1)
    agent = flow.start("q")
    prompt = flow.build_system_prompt(agent)
    assert "Structured Subagent Output" in prompt
    assert "output_schema" in prompt


def test_minimal_structured_output_option_absent_when_flag_disabled():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        max_depth=1,
        enable_structured_output=False,
    )
    agent = flow.start("q")
    assert "Structured Subagent Output" not in flow.build_system_prompt(agent)


def test_minimal_structured_output_option_absent_without_subagents():
    flow = Flow(StubLLM(lambda _messages: "unused"), max_depth=0)
    agent = flow.start("q")
    assert "Structured Subagent Output" not in flow.build_system_prompt(agent)


def test_minimal_first_turn_note_present_then_drops_after_output():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    agent = flow.start("q")
    assert "First Turn" in flow.build_system_prompt(agent)
    agent.commit(LLMOutput(content="```repl\nprint(1)\n```", code="print(1)"))
    assert "First Turn" not in flow.build_system_prompt(agent)


def test_minimal_first_turn_note_mentions_inputs_when_present():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    with_inputs = flow.start("q", {"doc": "x"})
    assert "INPUTS" in flow.build_system_prompt(with_inputs)


def test_minimal_static_system_prompt_within_size_guard():
    from rflow.minimal.prompts import MAX_STATIC_PROMPT_CHARS, SYSTEM_PROMPT

    assert len(SYSTEM_PROMPT) <= MAX_STATIC_PROMPT_CHARS


def test_minimal_prompt_builder_names_lists_sections_in_order():
    from rflow.minimal.prompts import DEFAULT_BUILDER

    names = DEFAULT_BUILDER.names
    assert names[0] == "role"
    assert names[-1] == "first-turn"
    assert {"tools", "inputs", "status"} <= set(names)


def test_minimal_invalid_structured_done_records_error_then_recovers():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    replies = iter(
        [
            '```repl\ndone({"answer": "wrong"})\n```',
            '```repl\ndone({"answer": 7})\n```',
        ]
    )
    flow = Flow(StubLLM(lambda _messages: next(replies)), max_iters=2)
    graph = Graph(query="q", output_schema=schema)

    assert json.loads(flow.run(graph)) == {"answer": 7}
    assert [node.type for node in graph.nodes] == [
        "user_query",
        "llm_output",
        "exec_action",
        "error_output",
        "llm_output",
        "exec_action",
        "done_output",
    ]
    assert "StructuredOutputError" in graph.nodes[3].output


def test_minimal_child_output_schema_returns_parsed_result_to_parent():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }

    def reply(messages):
        task = first_user(messages)
        if task == "child":
            return '```repl\ndone({"answer": 5})\n```'
        return (
            "```repl\n"
            f'results = await launch_subagents([{{"name": "child", "query": "child", "output_schema": {schema!r}}}])\n'
            'done(str(results[0]["answer"] + 1))\n'
            "```"
        )

    flow = Flow(StubLLM(reply), max_depth=1)
    graph = Graph(query="parent")

    assert flow.run(graph) == "6"
    assert graph["root.child"].output_schema == schema


def test_minimal_llm_query_batched_supports_output_schema():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    flow = Flow(StubLLM(lambda _messages: '{"answer": 3}'))

    async def collect():
        return await flow.llm_query_batched(["q1", "q2"], output_schema=schema)

    assert asyncio.run(collect()) == [{"answer": 3}, {"answer": 3}]


def test_minimal_async_pool_cancels_sibling_work_on_failure():
    cancelled = False

    async def slow():
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise

    async def fail():
        await asyncio.sleep(0)
        raise RuntimeError("boom")

    async def run():
        pool = AsyncPool(max_concurrency=2)
        try:
            await pool.gather(slow(), fail())
        except RuntimeError:
            return
        raise AssertionError("pool.gather should have raised")

    asyncio.run(run())

    assert cancelled


def test_minimal_run_stream_emits_graph_events_only():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))

    async def collect():
        return [event async for event in flow.run_stream("q")]

    events = asyncio.run(collect())
    assert events[0].type == "graph_created"
    assert events[-1].type == "append_node"
    assert events[-1].node_type == "done_output"
    assert {event.type for event in events} == {
        "graph_created",
        "append_node",
    }
    assert [event.node_type for event in events if event.type == "append_node"] == [
        "user_query",
        "llm_output",
        "exec_action",
        "done_output",
    ]


def test_minimal_events_carry_graph_id():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    async def collect():
        return [event async for event in flow.run_streaming(graph)]

    events = asyncio.run(collect())
    appended = [event for event in events if event.type == "append_node"]
    assert appended
    assert all(event.graph_id == graph.graph_id for event in appended)


def test_minimal_group_flows_streams_tagged_events_and_returns_graphs():
    def reply(_messages):
        return '```repl\ndone("ok")\n```'

    a_flow = Flow(StubLLM(reply))
    b_flow = Flow(StubLLM(reply))
    a_graph = Graph(query="a")
    b_graph = Graph(query="b")

    group = group_flows(a=(a_flow, a_graph), b=(b_flow, b_graph))

    async def collect():
        events = [event async for event in group]
        return events

    events = asyncio.run(collect())
    graph_ids = {event.graph_id for event in events}
    assert graph_ids == {a_graph.graph_id, b_graph.graph_id}
    assert a_graph.result() == "ok"
    assert b_graph.result() == "ok"


def test_minimal_group_flows_pool_bounds_concurrent_flows():
    state = {"active": 0, "peak": 0}

    class ConcurrencyProbeLLM:
        async def chat(self, _messages):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0)
            state["active"] -= 1
            return '```repl\ndone("ok")\n```'

    entries = {f"f{i}": (Flow(ConcurrencyProbeLLM()), f"q{i}") for i in range(3)}

    async def peak_with(pool):
        state["active"] = state["peak"] = 0
        await group_flows(pool=pool, **entries).run()
        return state["peak"]

    assert asyncio.run(peak_with(SequentialPool())) == 1
    entries = {f"f{i}": (Flow(ConcurrencyProbeLLM()), f"q{i}") for i in range(3)}
    assert asyncio.run(peak_with(AsyncPool(max_concurrency=2))) == 2


def test_minimal_group_flows_run_returns_graphs_by_label():
    flow_a = Flow(StubLLM(lambda _messages: '```repl\ndone("A")\n```'))
    flow_b = Flow(StubLLM(lambda _messages: '```repl\ndone("B")\n```'))

    group = group_flows(a=(flow_a, "qa"), b=(flow_b, "qb"))
    graphs = asyncio.run(group.run())

    assert set(graphs) == {"a", "b"}
    assert graphs["a"].result() == "A"
    assert graphs["b"].result() == "B"


def test_minimal_run_streaming_emits_events():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))

    async def collect():
        return [event async for event in flow.run_streaming("q")]

    seen = asyncio.run(collect())
    graph = seen[0].graph

    assert graph.result() == "ok"
    assert seen[0].type == "graph_created"
    assert seen[-1].type == "append_node"
    assert seen[-1].node_type == "done_output"
    assert [event.node_type for event in seen if event.type == "append_node"] == [
        "user_query",
        "llm_output",
        "exec_action",
        "done_output",
    ]


def test_minimal_run_streaming_mutates_passed_graph_in_place():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    async def collect():
        return [event async for event in flow.run_streaming(graph)]

    seen = asyncio.run(collect())

    assert graph.result() == "ok"
    assert seen[0].type == "append_node"
    assert {event.type for event in seen} == {"append_node"}
    assert [node.type for node in graph.nodes] == [
        "user_query",
        "llm_output",
        "exec_action",
        "done_output",
    ]


def test_minimal_reusing_same_graph_keeps_repl_state_after_edit():
    replies = iter(
        [
            '```repl\nx = 41\ndone("first")\n```',
            '```repl\ndone(str(x + 1))\n```',
        ]
    )
    flow = Flow(StubLLM(lambda _messages: next(replies)))
    graph = Graph(query="first")

    assert flow.run(graph) == "first"
    assert (graph.graph_id, "root") in flow.repls

    graph.nodes.pop()
    graph.commit(UserQuery(content="continue"))

    assert flow.run(graph) == "42"
    assert graph.result() == "42"


def test_minimal_close_repls_can_target_one_graph_id():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    first = Graph(query="first")
    second = Graph(query="second")

    flow.run(first)
    flow.run(second)

    assert (first.graph_id, "root") in flow.repls
    assert (second.graph_id, "root") in flow.repls

    flow.close_repls(graph_id=first.graph_id)

    assert (first.graph_id, "root") not in flow.repls
    assert (second.graph_id, "root") in flow.repls

    flow.close_repls()

    assert flow.repls == {}


def test_minimal_step_until_node_count_then_done():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    async def collect():
        first = await flow.step(graph, until="node", n=2)
        rest = await flow.step(until="done")
        return first, rest

    first, rest = asyncio.run(collect())

    assert [event.node_type for event in first if event.type == "append_node"] == [
        "user_query",
        "llm_output",
    ]
    assert rest[-1].type == "append_node"
    assert rest[-1].node_type == "done_output"
    assert graph.result() == "ok"


def test_minimal_step_after_start_advances_existing_graph():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = flow.start(Graph(query="q"))

    async def collect():
        return await flow.step(graph)

    events = asyncio.run(collect())

    assert [node.type for node in graph.nodes[:2]] == ["user_query", "llm_output"]
    assert any(
        event.type == "append_node" and event.node_type == "llm_output"
        for event in events
    )


def test_minimal_step_accepts_callable_boundary():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    async def collect():
        return await flow.step(
            graph,
            until=lambda event, _graph: (
                event.type == "append_node"
                and event.node_type == "exec_action"
            ),
        )

    events = asyncio.run(collect())

    assert [event.node_type for event in events if event.type == "append_node"] == [
        "user_query",
        "llm_output",
        "exec_action",
    ]


def test_minimal_step_supports_named_error_boundary():
    flow = Flow(StubLLM(lambda _messages: "no repl here"))
    graph = Graph(query="q")

    async def collect():
        return await flow.step(graph, until="error")

    events = asyncio.run(collect())

    assert [event.node_type for event in events if event.type == "append_node"] == [
        "user_query",
        "llm_output",
        "exec_action",
        "error_output",
    ]
    assert graph.nodes[-1].output.startswith("MissingReplError:")


def test_minimal_uses_default_prompt_builder_and_inputs_namespace():
    seen = {}

    def reply(messages):
        seen["system"] = messages[0]["content"]
        return '```repl\nprint(INPUTS["x"])\ndone("ok")\n```'

    flow = Flow(StubLLM(reply))
    graph = Graph(query="q")

    assert flow.run(graph, inputs={"x": "available"}) == "ok"
    assert "Recursive Coding Agent" in seen["system"]
    assert "Available in the REPL" in seen["system"]
    assert "Status" in seen["system"]
    assert "available" in graph.nodes[-1].output
    assert "[done] ok" in graph.nodes[-1].output


def test_minimal_runtime_register_tools_are_prompted_and_executable():
    seen = {}

    @tool("Double a number.")
    def double(x: int) -> int:
        return x * 2

    def reply(messages):
        seen["system"] = messages[0]["content"]
        return '```repl\nprint(double(3))\ndone("ok")\n```'

    runtime = LocalRuntime()
    runtime.register_tools([double])
    flow = Flow(StubLLM(reply), runtime=runtime)
    graph = Graph(query="q")

    assert flow.run(graph) == "ok"
    assert "`double" in seen["system"]
    assert "6" in graph.nodes[-1].output


def test_minimal_local_runtime_uses_working_directory(tmp_path):
    flow = Flow(
        StubLLM(
            lambda _messages: (
                "```repl\n"
                "from pathlib import Path\n"
                'Path("note.txt").write_text("hello")\n'
                'done(Path("note.txt").read_text())\n'
                "```"
            )
        ),
        runtime=LocalRuntime(working_directory=tmp_path),
    )
    graph = Graph(query="q")

    assert flow.run(graph) == "hello"
    assert (tmp_path / "note.txt").read_text() == "hello"


def test_minimal_repl_client_uses_minimal_repl_server():
    repl = ReplClient(
        PopenConnection(
            [sys.executable, "-u", "-m", "rflow.minimal.runtime.repl_server"],
            label="test minimal remote REPL",
            repl_timeout=5,
        )
    )

    def done(answer):
        repl.done_result = str(answer)
        raise DoneSignal()

    async def run():
        try:
            repl.seed({"done": done}, {"x": "remote"})
            return await repl.run('print(INPUTS["x"])\ndone("ok")')
        finally:
            repl.close()

    output = asyncio.run(run())

    assert "remote" in output
    assert repl.done_result == "ok"
    assert not repl.errored


def test_minimal_remote_protocol_models_round_trip():
    run = parse_request(RunRequest(id="r1", code="print(1)").model_dump())
    response = parse_client_message(ReplResponse(id="r1", output="1").model_dump())
    proxy = parse_client_message(
        ProxyCall(id="p1", proxy="done", args=["ok"]).model_dump()
    )

    assert isinstance(run, RunRequest)
    assert run.code == "print(1)"
    assert isinstance(response, ReplResponse)
    assert response.output == "1"
    assert isinstance(proxy, ProxyCall)
    assert proxy.proxy == "done"


def test_minimal_repl_server_supports_ping_and_capabilities():
    repl = ReplClient(
        PopenConnection(
            [sys.executable, "-u", "-m", "rflow.minimal.runtime.repl_server"],
            label="test minimal remote REPL",
            repl_timeout=5,
        )
    )
    try:
        ping = repl.call(PingRequest(id="ping-test"))
        capabilities = repl.call(CapabilitiesRequest(id="cap-test"))
    finally:
        repl.close()

    assert ping.ok
    assert capabilities.capabilities is not None
    assert capabilities.capabilities.fork == "none"


def test_minimal_subprocess_runtime_executes_agent_code():
    flow = Flow(
        StubLLM(lambda _messages: '```repl\nprint("subproc")\ndone("ok")\n```'),
        runtime=SubprocessRuntime(),
    )
    graph = Graph(query="q")

    assert flow.run(graph) == "ok"
    assert "subproc" in graph.nodes[-1].output


def test_minimal_subprocess_runtime_uses_working_directory(tmp_path):
    flow = Flow(
        StubLLM(
            lambda _messages: (
                "```repl\n"
                "from pathlib import Path\n"
                'Path("note.txt").write_text("hello")\n'
                'done(Path("note.txt").read_text())\n'
                "```"
            )
        ),
        runtime=SubprocessRuntime(working_directory=tmp_path),
    )
    graph = Graph(query="q")

    assert flow.run(graph) == "hello"
    assert (tmp_path / "note.txt").read_text() == "hello"


def test_minimal_docker_argv_uses_minimal_repl_server(tmp_path):
    argv = build_docker_argv(
        "rlmflow:minimal",
        mounts={str(tmp_path): "/workspace"},
        workdir="/workspace",
        network="none",
    )

    assert "rflow.minimal.runtime.repl_server" in argv
    assert "rflow.runtime.repl_server" not in argv
    assert argv[:4] == ["docker", "run", "-i", "--rm"]


def test_minimal_package_does_not_import_main_rflow_modules():
    root = Path("rflow/minimal")
    disallowed = (
        "from rflow.runtime",
        "import rflow.runtime",
        "from rflow.tools",
        "import rflow.tools",
        "from rflow.prompts",
        "import rflow.prompts",
        "from rflow.integrations",
        "import rflow.integrations",
    )

    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text()
        if any(pattern in text for pattern in disallowed):
            offenders.append(str(path))

    assert offenders == []


def test_minimal_graph_agent_tree_is_high_level():
    def reply(messages):
        task = first_user(messages)
        if task == "child task":
            return '```repl\ndone("child result")\n```'
        return (
            "```repl\n"
            'results = await launch_subagents([{"name": "auth", "query": "child task"}])\n'
            'done("root saw " + results[0])\n'
            "```"
        )

    flow = Flow(StubLLM(reply), max_depth=1)
    graph = Graph(query="root query")
    flow.run(graph)

    tree = render_tree(graph)
    assert tree.startswith("root query")
    assert "root: done root saw child result" in tree
    assert "root.auth: done child result" in tree
    assert "provider response" not in tree
    assert "tool call:" not in tree


def test_minimal_live_tree_renderer_consumes_events(capsys):
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="root query")

    renderer = LiveTreeRenderer(clear=False)

    async def collect():
        async for event in flow.run_stream(graph):
            renderer.handle(event, graph)

    asyncio.run(collect())
    output = capsys.readouterr().out
    assert "root:" in output
    assert "done ok" in output


def test_minimal_graph_saves_run_layout(tmp_path):
    def reply(messages):
        task = first_user(messages)
        if task == "child task":
            return '```repl\ndone("child result")\n```'
        return (
            "```repl\n"
            'results = await launch_subagents([{"name": "auth", "query": "child task"}])\n'
            'done("root saw " + results[0])\n'
            "```"
        )

    flow = Flow(StubLLM(reply), max_depth=1)
    graph = Graph(query="root query")
    flow.run(graph)
    run_dir = graph.save(tmp_path / "run", metadata={"example": "test"})

    manifest = json.loads((run_dir / "graph.json").read_text())
    assert manifest["root_agent_id"] == "root"
    assert manifest["metadata"]["example"] == "test"
    assert "root.auth" in manifest["agents"]

    root_dir = run_dir / "agents" / "root"
    child_dir = root_dir / "auth"
    assert json.loads((root_dir / "agent.json").read_text())["query"] == "root query"
    assert json.loads((child_dir / "agent.json").read_text())["query"] == "child task"
    assert '"type": "done_output"' in (child_dir / "session.jsonl").read_text()
    assert json.loads((child_dir / "latest.json").read_text())["result"] == "child result"

    child_events = [
        json.loads(line)["type"]
        for line in (child_dir / "session.jsonl").read_text().splitlines()
    ]
    assert child_events == ["user_query", "llm_output", "exec_action", "done_output"]

    loaded = Graph.load(run_dir)
    assert loaded.result() == "root saw child result"
    assert loaded["root.auth"].result() == "child result"


def test_minimal_graph_id_is_auto_created_and_persisted(tmp_path):
    graph = Graph(query="loaded")
    graph.commit(UserQuery(content="loaded"))
    graph_id = graph.graph_id

    loaded = Graph.load(graph.save(tmp_path / "run"))

    assert graph_id.startswith("g_")
    assert loaded.graph_id == graph_id
    assert loaded.query == "loaded"


# --- Reversibility: checkpoint / revert / rebuild / fork / merge / discard ----


def _seed_exec_graph(*code_blocks):
    """A graph shaped like a real trajectory: (llm_output, exec_action, exec_output)*."""
    graph = Graph(query="q")
    graph.commit(UserQuery(content="q"))
    for code in code_blocks:
        graph.commit(LLMOutput(content="turn", code=code))
        graph.commit(ExecAction(code=code))
        graph.commit(ExecOutput(content="", output=""))
    return graph


async def _worker_step(flow, graph, code):
    """Simulate one exec turn on a branch: record the action, then run it."""
    graph.commit(ExecAction(code=code))
    await flow.exec_turn(graph, code)


def test_minimal_checkpoint_revert_restores_nodes():
    graph = _seed_exec_graph("x = 1")
    checkpoint = graph.checkpoint()
    assert isinstance(checkpoint, GraphCheckpoint)

    graph.commit(LLMOutput(content="more", code="y = 2"))
    graph.commit(ExecAction(code="y = 2"))
    graph.commit(ExecOutput(content="", output=""))
    assert len(graph.nodes) == 7

    graph.revert(checkpoint)

    assert len(graph.nodes) == 4
    assert graph.checkpoint().digest == checkpoint.digest


def test_minimal_revert_refuses_after_history_rewrite():
    graph = _seed_exec_graph("x = 1")
    checkpoint = graph.checkpoint()

    # Rewrite a node BELOW the checkpoint -> digest no longer matches.
    graph.replace_node(graph.nodes[1].id, LLMOutput(content="changed", code="x = 999"))

    refused = False
    try:
        graph.revert(checkpoint)
    except ValueError:
        refused = True
    assert refused


def test_minimal_rebuild_repl_reconstructs_variables_without_appending():
    async def run():
        flow = Flow(StubLLM(lambda _messages: "unused"))
        graph = _seed_exec_graph("x = 21 * 2", "y = x + 1")
        repl = await flow.rebuild_repl(graph)
        return repl.namespace.get("x"), repl.namespace.get("y"), len(graph.nodes)

    x, y, node_count = asyncio.run(run())
    assert (x, y) == (42, 43)
    assert node_count == 7  # replay must not append result nodes


def test_minimal_fork_is_isolated_from_parent():
    async def run():
        flow = Flow(StubLLM(lambda _messages: "unused"))
        parent = _seed_exec_graph("x = 1")
        await flow.rebuild_repl(parent)

        child = await flow.fork(parent)
        await _worker_step(flow, child, "y = x + 1")

        return (
            parent.graph_id,
            child.graph_id,
            len(parent.nodes),
            flow.repl_for(parent).namespace,
            flow.repl_for(child).namespace,
        )

    parent_id, child_id, parent_nodes, parent_ns, child_ns = asyncio.run(run())
    assert child_id != parent_id
    assert parent_nodes == 4  # parent trajectory untouched
    assert "y" not in parent_ns  # parent REPL untouched
    assert child_ns.get("x") == 1  # inherited via fork replay
    assert child_ns.get("y") == 2  # child's own work


def test_minimal_merge_folds_disjoint_children():
    async def run():
        flow = Flow(StubLLM(lambda _messages: "unused"))
        parent = _seed_exec_graph("report = {}")
        await flow.rebuild_repl(parent)  # parent has a live REPL -> delta-run rung

        child_a = await flow.fork(parent)
        child_b = await flow.fork(parent)
        await _worker_step(flow, child_a, "stats_a = 12")
        await _worker_step(flow, child_b, "stats_b = 34")

        await flow.merge(parent, child_a)
        await flow.merge(parent, child_b)
        return parent, flow.repl_for(parent).namespace

    parent, namespace = asyncio.run(run())
    assert namespace.get("report") == {}
    assert namespace.get("stats_a") == 12
    assert namespace.get("stats_b") == 34

    summaries = [
        node.content
        for node in parent.nodes
        if node.type == "exec_output" and "merged branch" in (node.content or "")
    ]
    assert len(summaries) == 2  # exactly one summary node per merge


def test_minimal_merge_adopts_first_child_repl():
    async def run():
        flow = Flow(StubLLM(lambda _messages: "unused"))
        parent = _seed_exec_graph("report = {}")  # NOTE: no live parent REPL

        child = await flow.fork(parent)
        await _worker_step(flow, child, "stats_a = 12")

        await flow.merge(parent, child)  # rung 1: adopt child's REPL, zero re-exec
        return flow.repl_for(parent).namespace

    namespace = asyncio.run(run())
    assert namespace.get("report") == {}
    assert namespace.get("stats_a") == 12


def test_minimal_merge_conflict_is_last_write_wins():
    async def run():
        flow = Flow(StubLLM(lambda _messages: "unused"))
        parent = _seed_exec_graph("v = 0")
        await flow.rebuild_repl(parent)

        child_a = await flow.fork(parent)
        child_b = await flow.fork(parent)
        await _worker_step(flow, child_a, "v = 1")
        await _worker_step(flow, child_b, "v = 2")

        await flow.merge(parent, child_a)
        await flow.merge(parent, child_b)  # merged last -> wins
        return flow.repl_for(parent).namespace.get("v")

    assert asyncio.run(run()) == 2


def test_minimal_discard_closes_branch_repls():
    async def run():
        flow = Flow(StubLLM(lambda _messages: "unused"))
        parent = _seed_exec_graph("x = 1")
        await flow.rebuild_repl(parent)

        child = await flow.fork(parent)
        child_key = repl_key(child)
        present_before = child_key in flow.repls

        flow.discard(child)
        return present_before, child_key in flow.repls

    present_before, present_after = asyncio.run(run())
    assert present_before
    assert not present_after


def _counting_replies(*replies):
    """StubLLM callable returning ``replies`` in order, repeating the last."""
    state = {"n": 0}

    def reply(_messages):
        index = min(state["n"], len(replies) - 1)
        state["n"] += 1
        return replies[index]

    return reply


def test_minimal_truncate_output_caps_exec_observation_not_done_result():
    reply = _counting_replies(
        "```repl\nprint('X' * 100)\n```",
        "```repl\ndone('final')\n```",
    )
    flow = Flow(StubLLM(reply), max_output_length=10)
    graph = Graph(query="q")

    assert flow.run(graph) == "final"
    exec_output = next(
        node for node in graph.nodes if isinstance(node, ExecOutput)
    )
    assert exec_output.output.startswith("X" * 10)
    assert "truncated 90 chars" in exec_output.output
    assert len(exec_output.output) < 100


def test_minimal_max_query_chars_refuses_overlong_child_query():
    long_query = "q" * 20
    reply = (
        "```repl\n"
        f'results = await launch_subagents([{{"name": "w", "query": "{long_query}"}}])\n'
        "done(results[0])\n"
        "```"
    )
    flow = Flow(StubLLM(lambda _messages: reply), max_depth=1, max_query_chars=5)

    assert "refused: query too long" in flow.run(Graph(query="parent"))


def test_minimal_rflow_env_vars_visible_to_agent_and_restored():
    reply = (
        "```repl\n"
        "import os\n"
        'done(os.environ["RFLOW_AGENT_ID"] + "|" + os.environ["RFLOW_IS_ROOT"] '
        '+ "|" + os.environ["RFLOW_MAX_DEPTH"])\n'
        "```"
    )
    flow = Flow(StubLLM(lambda _messages: reply), max_depth=3)

    assert flow.run(Graph(query="q")) == "root|1|3"
    assert "RFLOW_AGENT_ID" not in os.environ


def test_minimal_llm_query_batched_forwards_sampling_kwargs():
    class KwargsLLM:
        def __init__(self):
            self.calls = []
            self.last_usage = None

        def chat(self, messages, **kwargs):
            self.calls.append(kwargs)
            return "ok"

    llm = KwargsLLM()
    flow = Flow(llm)

    async def collect():
        return await flow.llm_query_batched(
            ["a"], temperature=0.5, top_p=0.9, max_tokens=10, stop=["x"]
        )

    assert asyncio.run(collect()) == ["ok"]
    assert llm.calls == [
        {"temperature": 0.5, "top_p": 0.9, "max_tokens": 10, "stop": ["x"]}
    ]


def test_minimal_max_budget_stops_run_when_token_cap_reached():
    flow = Flow(
        UsageLLM('```repl\nprint("loop")\n```', input_tokens=3, output_tokens=3),
        max_budget=5,
        max_iters=10,
    )

    assert flow.run(Graph(query="q")) == "[budget exceeded]"


def test_minimal_terminate_requests_agent_stop():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("never")\n```'))
    flow.terminate(["root"])

    assert flow.run(Graph(query="q")) == "[terminated]"


def test_minimal_child_max_iters_bounds_children_independently():
    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                'results = await launch_subagents([{"name": "w", "query": "go"}])\n'
                "done(results[0])\n"
                "```"
            )
        return '```repl\nprint("working")\n```'

    flow = Flow(StubLLM(reply), max_depth=1, max_iters=10, child_max_iters=1)

    assert flow.run(Graph(query="parent")) == "[max_iters exceeded]"


def test_minimal_llm_request_timeout_raises_on_slow_client():
    class SlowLLM:
        async def chat(self, _messages):
            await asyncio.sleep(1)
            return "x"

    flow = Flow(SlowLLM(), llm_request_timeout=0.01)

    raised = False
    try:
        flow.run(Graph(query="q"))
    except TimeoutError:
        raised = True
    assert raised


def test_minimal_add_tool_injects_into_live_repl_and_prompt():
    @tool("Return a fixed greeting.")
    def greet() -> str:
        return "hi from tool"

    flow = Flow(StubLLM(lambda _messages: "```repl\npass\n```"))
    graph = Graph(query="q")
    repl = flow.repl_for(graph)
    assert "greet" not in repl.namespace

    flow.add_tool(greet)

    assert repl.namespace["greet"] is greet
    assert flow.tools["greet"] is greet
    # Per-turn prompt reflects the newly registered tool.
    assert "greet()" in flow.build_system_prompt(graph)
    # ...and it is immediately callable inside the running REPL.
    out = asyncio.run(repl.run("print(greet())"))
    assert "hi from tool" in out


def test_minimal_add_tool_reaches_agents_spawned_later():
    @tool("Return the answer.")
    def answer() -> int:
        return 42

    flow = Flow(StubLLM(lambda _messages: "```repl\ndone(str(answer()))\n```"))
    flow.add_tool(answer)

    assert flow.run(Graph(query="q")) == "42"


def test_minimal_remove_tool_drops_from_live_repl_and_flow():
    @tool("Return a fixed greeting.")
    def greet() -> str:
        return "hi"

    flow = Flow(StubLLM(lambda _messages: "```repl\npass\n```"), tools=[greet])
    graph = Graph(query="q")
    repl = flow.repl_for(graph)
    assert "greet" in repl.namespace

    removed = flow.remove_tool("greet")

    assert removed is greet
    assert "greet" not in flow.tools
    assert "greet" not in repl.namespace


def test_minimal_add_remove_tool_reject_reserved_names():
    flow = Flow(StubLLM(lambda _messages: "```repl\npass\n```"))
    for name in ("done", "launch_subagents", "INPUTS"):
        try:
            flow.add_tool(lambda: None, name=name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"add_tool should reject reserved name {name!r}")
    try:
        flow.remove_tool("done")
    except ValueError:
        pass
    else:
        raise AssertionError("remove_tool should reject reserved names")


def test_minimal_repl_client_inject_and_remove_tool():
    repl = ReplClient(
        PopenConnection(
            [sys.executable, "-u", "-m", "rflow.minimal.runtime.repl_server"],
            label="test minimal remote REPL",
            repl_timeout=5,
        )
    )

    def done(answer):
        repl.done_result = str(answer)
        raise DoneSignal()

    @tool("Echo the argument back.", proxy=True)
    def echo(value):
        return value

    async def run():
        try:
            repl.seed({"done": done}, {})
            repl.inject_tool("echo", echo)
            injected = await repl.run('print(echo("hey"))')
            repl.remove_tool("echo")
            removed = await repl.run(
                'print("gone" if "echo" not in dir() else echo("x"))'
            )
            return injected, removed
        finally:
            repl.close()

    injected, removed = asyncio.run(run())

    assert "hey" in injected
    assert "gone" in removed


def test_minimal_tool_metadata_marks_async_and_renders_await():
    from rflow.minimal.tools import format_tool_line, get_tool_metadata

    @tool("Do async work.")
    async def worker(x):
        return x

    @tool("Do sync work.")
    def helper(x):
        return x

    assert get_tool_metadata(worker).is_async is True
    assert get_tool_metadata(helper).is_async is False
    assert format_tool_line(worker).startswith("- `await worker(")
    assert format_tool_line(helper).startswith("- `helper(")


def test_minimal_repl_client_async_proxy_tool_is_awaitable():
    repl = ReplClient(
        PopenConnection(
            [sys.executable, "-u", "-m", "rflow.minimal.runtime.repl_server"],
            label="test minimal remote REPL",
            repl_timeout=5,
        )
    )

    def done(answer):
        repl.done_result = str(answer)
        raise DoneSignal()

    @tool("Double asynchronously.", proxy=True)
    async def adouble(value):
        return value * 2

    async def run():
        try:
            repl.seed({"done": done}, {})
            repl.inject_tool("adouble", adouble)
            return await repl.run("v = await adouble(21)\nprint(v)")
        finally:
            repl.close()

    out = asyncio.run(run())

    assert "42" in out


def test_minimal_subprocess_runtime_supports_awaited_launch_subagents():
    def reply(messages):
        if first_user(messages) == "parent":
            return (
                "```repl\n"
                'r = await launch_subagents([{"name": "c", "query": "child"}])\n'
                "done(r[0])\n"
                "```"
            )
        return '```repl\ndone("child-done")\n```'

    flow = Flow(StubLLM(reply), runtime=SubprocessRuntime(), max_depth=1)

    assert flow.run(Graph(query="parent")) == "child-done"
