import asyncio
import json
import os

from rflow import (
    ExecOutput,
    Flow,
    Graph,
    LLMOutput,
    LLMUsage,
    UserQuery,
)

from helpers import (
    StubLLM,
    UsageLLM,
    assert_llm_outputs_are_followed_by_exec_actions,
    counting_replies,
    first_user,
)


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



def test_minimal_run_stream_emits_graph_events_only():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))

    async def collect():
        return [event async for event in flow.run_streaming("q")]

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
        first = [event async for event in flow.run_streaming(graph, until="next", n=2)]
        rest = [event async for event in flow.run_streaming(until="done")]
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
        return [event async for event in flow.run_streaming(graph, until="next")]

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
        return [
            event
            async for event in flow.run_streaming(
                graph,
                until=lambda event, current: (
                    event.type == "append_node"
                    and event.node_type == "exec_action"
                ),
            )
        ]

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
        return [event async for event in flow.run_streaming(graph, until="error")]

    events = asyncio.run(collect())

    assert [event.node_type for event in events if event.type == "append_node"] == [
        "user_query",
        "llm_output",
        "exec_action",
        "error_output",
    ]
    assert graph.nodes[-1].output.startswith("MissingReplError:")



def test_minimal_truncate_output_caps_exec_observation_not_done_result():
    reply = counting_replies(
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

