import asyncio
import json
import sys
from pathlib import Path

from rflow.minimal import (
    AddChild,
    AppendNode,
    AsyncPool,
    Graph,
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
    tool,
)
from rflow.minimal.docker import build_docker_argv
from rflow.minimal.protocol import (
    CapabilitiesRequest,
    PingRequest,
    ProxyCall,
    ReplResponse,
    RunRequest,
    parse_client_message,
    parse_request,
)
from rflow.minimal.repl import DoneSignal
from rflow.minimal.rendering import LiveTreeRenderer, render_tree
from rflow.minimal.remote import ReplClient
from rflow.minimal.stdio import PopenServerConnection


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


def test_minimal_repl_client_uses_minimal_remote_server():
    repl = ReplClient(
        PopenServerConnection(
            [sys.executable, "-u", "-m", "rflow.minimal.remote_server"],
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


def test_minimal_remote_server_supports_ping_and_capabilities():
    repl = ReplClient(
        PopenServerConnection(
            [sys.executable, "-u", "-m", "rflow.minimal.remote_server"],
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


def test_minimal_docker_argv_uses_minimal_remote_server(tmp_path):
    argv = build_docker_argv(
        "rlmflow:minimal",
        mounts={str(tmp_path): "/workspace"},
        workdir="/workspace",
        network="none",
    )

    assert "rflow.minimal.remote_server" in argv
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
