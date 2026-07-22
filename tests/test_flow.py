import asyncio
import json
import os

from rlmflow import (
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
    seed_exec_graph,
)


def test_adopt_reparents_fork_and_moves_its_repl():
    """``flow.adopt`` attaches a prepared fork under a parent: it shares the
    parent's ``graph_id``, gets a nested ``agent_id``, keeps its replayed REPL
    state under the new key, and is registered as a child."""
    flow = Flow(StubLLM(lambda _m: '```repl\ndone("x")\n```'), max_depth=1)
    parent = Graph(query="parent")

    async def prepare_and_adopt():
        base = seed_exec_graph("x = 41")
        fork = await flow.fork(base)  # replay rebuilds the fork's REPL (x == 41)
        assert flow.get_var(fork, "x") == 41
        return flow.adopt(parent, fork, name="b0")

    child = asyncio.run(prepare_and_adopt())

    assert child.graph_id == parent.graph_id
    assert child.agent_id == "root.b0"
    assert child.parent_agent_id == parent.agent_id
    assert child.depth == parent.depth + 1
    assert "root.b0" in parent.children
    assert parent["root.b0"] is child
    assert all(node.agent_id == "root.b0" for node in child.nodes)
    # the fork's live REPL moved with it — rewound state survives the reparent
    assert flow.get_var(child, "x") == 41


def test_launch_subagents_warm_start_from_prepared_graph():
    """A ``launch_subagents`` spec may carry a prepared ``graph`` (a fork) instead
    of a ``query``: it's adopted, its ``query`` is appended as the next turn, and
    it self-drives to ``done`` on the same supervise/await loop as a cold child."""
    flow = Flow(StubLLM(lambda _m: '```repl\ndone("child done")\n```'), max_depth=1)
    parent = Graph(query="parent")

    async def drive():
        base = seed_exec_graph("y = 7")
        fork = await flow.fork(base)
        launch = flow.launch_subagents(parent, flow.repl_for(parent))
        return await launch(
            [{"graph": fork, "name": "b0", "query": "recover now"}]
        )

    results = asyncio.run(drive())

    assert results == ["child done"]
    child = parent["root.b0"]
    assert child.result() == "child done"
    # warm start: kept its prior trajectory AND got the kickoff query appended
    assert flow.get_var(child, "y") == 7
    assert any(
        node.type == "user_query" and "recover now" in (node.content or "")
        for node in child.nodes
    )
    assert parent.nodes[-2].type == "supervising_output"
    assert parent.nodes[-1].type == "resume_action"


def test_launch_subgraphs_runs_prepared_forks_as_children():
    """``flow.launch_subgraphs`` is the warm, graph-first wrapper over
    ``launch_subagents``: hand it prepared forks (+ optional per-child kickoff
    queries and names) and they adopt, self-drive, and are awaited together."""
    flow = Flow(StubLLM(lambda _m: '```repl\ndone("done")\n```'), max_depth=1)
    parent = Graph(query="parent")

    async def drive():
        forks = [await flow.fork(seed_exec_graph(f"v = {i}")) for i in range(2)]
        return await flow.launch_subgraphs(
            parent,
            forks,
            queries=["go a", "go b"],
            names=["b0", "b1"],
        )

    results = asyncio.run(drive())

    assert results == ["done", "done"]
    # both prepared graphs attached under the parent with the given names, kept
    # their rewound state, and got their kickoff queries appended
    assert flow.get_var(parent["root.b0"], "v") == 0
    assert flow.get_var(parent["root.b1"], "v") == 1
    assert any(
        node.type == "user_query" and "go b" in (node.content or "")
        for node in parent["root.b1"].nodes
    )


def test_minimal_flow_launch_subagents_is_public():
    def reply(messages):
        task = first_user(messages)
        return f'```repl\ndone("{task} done")\n```'

    flow = Flow(StubLLM(reply), max_depth=1)
    graph = Graph(query="parent")
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

    assert flow.run(graph=graph) == "ok"
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
            graph=graph,
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

    assert flow.run(graph=graph) == "p:c"
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

    assert flow.run(graph=graph) == "fast child"
    assert graph["root.fast"].model == "fast"


def _spawn_then_branch_on_system(marker: str, child_prompt: str):
    # One client whose behavior branches on the system prompt it receives: the
    # root spawns a child under ``child_prompt``; the child recognizes the injected
    # profile text and finishes. Proves the profile actually reached the child.
    def reply(messages):
        if marker in messages[0]["content"]:
            return '```repl\ndone("used-profile")\n```'
        return (
            "```repl\n"
            'r = await launch_subagents([{"name": "c", "query": "child", '
            f'"prompt_profile": "{child_prompt}"}}])\n'
            "done(r[0])\n"
            "```"
        )

    return StubLLM(reply)


def test_child_prompt_profile_reaches_child_llm():
    flow = Flow(
        _spawn_then_branch_on_system("CODER-9000", "coder"),
        prompts={"coder": "You are CODER-9000."},
        max_depth=1,
    )
    graph = Graph(query="parent")

    assert flow.run(graph=graph) == "used-profile"
    assert graph["root.c"].prompt_profile == "coder"


def test_prompt_router_overrides_spec_selection():
    # The spec asks for "coder" but the router forces "reviewer" on any child.
    flow = Flow(
        _spawn_then_branch_on_system("REVIEWER-7", "coder"),
        prompts={"coder": "You are CODER-9000.", "reviewer": "You are REVIEWER-7."},
        prompt_router=lambda _flow, graph: "reviewer" if graph.depth > 0 else "default",
        max_depth=1,
    )
    graph = Graph(query="parent")

    # Child finished only because the REVIEWER prompt (router's choice) reached it,
    # even though the spawn spec named "coder" (still recorded on the graph).
    assert flow.run(graph=graph) == "used-profile"
    assert graph["root.c"].prompt_profile == "coder"


def test_prompt_router_graph_honors_stamped_prompt():
    flow = Flow(
        StubLLM(lambda _m: '```repl\ndone("ok")\n```'),
        prompts={"coder": "You are CODER-9000."},
        prompt_router="graph",
    )
    graph = Graph(query="q", prompt_profile="coder")
    assert flow.prompt_name_for(graph) == "coder"
    assert flow.build_system_prompt(graph) == "You are CODER-9000."


def test_prompt_router_rejects_unknown_mode():
    import pytest

    with pytest.raises(TypeError, match="prompt_router"):
        Flow(StubLLM(lambda _m: ""), prompt_router="auto")  # type: ignore[arg-type]


def test_default_reserved_in_prompts_registry():
    import pytest

    with pytest.raises(ValueError, match="reserved"):
        Flow(StubLLM(lambda _m: ""), prompts={"default": "nope"})


def test_unknown_profile_falls_back_to_recorded_prompt():
    # A graph pointed at a profile the flow doesn't have resolves to the last
    # system prompt actually recorded on it (self-contained continuation), instead
    # of raising or silently using the flow default.
    flow = Flow(StubLLM(lambda _m: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q", prompt_profile="ghost")
    graph.system_prompts["sys_x"] = "RECORDED SYSTEM PROMPT"
    graph.commit(LLMOutput(content="hi", metadata={"system_prompt": "sys_x"}))

    assert flow.build_system_prompt(graph) == "RECORDED SYSTEM PROMPT"


def test_unknown_profile_with_no_history_raises():
    flow = Flow(StubLLM(lambda _m: ""))
    graph = Graph(query="q", prompt_profile="ghost")

    import pytest

    with pytest.raises(ValueError, match="unknown prompt"):
        flow.build_system_prompt(graph)



def test_minimal_usage_accounting_is_stored_in_llm_output_metadata():
    flow = Flow(UsageLLM('```repl\ndone("ok")\n```', input_tokens=3, output_tokens=2))
    graph = Graph(query="q")

    assert flow.run(graph=graph) == "ok"
    llm_output = next(node for node in graph.nodes if isinstance(node, LLMOutput))
    assert llm_output.metadata["model"] == "default"
    assert llm_output.metadata["usage"] == {"input_tokens": 3, "output_tokens": 2}
    # The node references its system prompt by id; the text lives in the graph's
    # content-addressed table and resolves back via system_prompt_for.
    assert llm_output.metadata["system_prompt"].startswith("sys_")
    assert graph.system_prompt_for(llm_output).startswith("You are a Recursive")
    assert graph.usage() == LLMUsage(input_tokens=3, output_tokens=2)
    assert graph.tokens() == (3, 2)
    assert graph.total_tokens() == 5



def test_minimal_run_stream_emits_graph_events_only():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))

    async def collect():
        return [event async for event in flow.run_streaming(query="q")]

    events = asyncio.run(collect())
    assert events[0].type == "graph_created"
    assert events[-1].type == "append_node"
    assert events[-1].node_type == "done_output"
    assert {event.type for event in events} == {
        "graph_created",
        "append_node",
    }
    assert [event.node_type for event in events if event.type == "append_node"] == [
        "llm_output",
        "exec_action",
        "done_output",
    ]



def test_minimal_events_carry_graph_id():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    async def collect():
        return [event async for event in flow.run_streaming(graph=graph)]

    events = asyncio.run(collect())
    appended = [event for event in events if event.type == "append_node"]
    assert appended
    assert all(event.graph_id == graph.graph_id for event in appended)



def test_minimal_run_streaming_emits_events():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))

    async def collect():
        return [event async for event in flow.run_streaming(query="q")]

    seen = asyncio.run(collect())
    graph = seen[0].graph

    assert graph.result() == "ok"
    assert seen[0].type == "graph_created"
    assert seen[-1].type == "append_node"
    assert seen[-1].node_type == "done_output"
    assert [event.node_type for event in seen if event.type == "append_node"] == [
        "llm_output",
        "exec_action",
        "done_output",
    ]



def test_minimal_run_streaming_mutates_passed_graph_in_place():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    async def collect():
        return [event async for event in flow.run_streaming(graph=graph)]

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

    assert flow.run(graph=graph) == "first"
    assert (graph.graph_id, "root") in flow.repls

    graph.nodes.pop()
    graph.commit(UserQuery(content="continue"))

    assert flow.run(graph=graph) == "42"
    assert graph.result() == "42"



def test_minimal_close_repls_can_target_one_graph_id():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    first = Graph(query="first")
    second = Graph(query="second")

    flow.run(graph=first)
    flow.run(graph=second)

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
        first = [event async for event in flow.run_streaming(graph=graph, until="next", n=2)]
        rest = [event async for event in flow.run_streaming(graph=graph, until="done")]
        return first, rest

    first, rest = asyncio.run(collect())

    assert [event.node_type for event in first if event.type == "append_node"] == [
        "llm_output",
        "exec_action",
    ]
    assert rest[-1].type == "append_node"
    assert rest[-1].node_type == "done_output"
    assert graph.result() == "ok"



def test_minimal_step_after_start_advances_existing_graph():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")

    async def collect():
        return [event async for event in flow.run_streaming(graph=graph, until="next")]

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
                graph=graph,
                until=lambda event, current: (
                    event.type == "append_node"
                    and event.node_type == "exec_action"
                ),
            )
        ]

    events = asyncio.run(collect())

    assert [event.node_type for event in events if event.type == "append_node"] == [
        "llm_output",
        "exec_action",
    ]



def test_minimal_step_supports_named_error_boundary():
    flow = Flow(StubLLM(lambda _messages: "no repl here"))
    graph = Graph(query="q")

    async def collect():
        return [event async for event in flow.run_streaming(graph=graph, until="error")]

    events = asyncio.run(collect())

    assert [event.node_type for event in events if event.type == "append_node"] == [
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

    assert flow.run(graph=graph) == "final"
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

    assert "refused: query too long" in flow.run(graph=Graph(query="parent"))



def test_rlmflow_env_vars_visible_to_agent():
    reply = (
        "```repl\n"
        'done(ENV["RLMFLOW_AGENT_ID"] + "|" + ENV["RLMFLOW_IS_ROOT"] '
        '+ "|" + ENV["RLMFLOW_MAX_DEPTH"])\n'
        "```"
    )
    flow = Flow(StubLLM(lambda _messages: reply), max_depth=3)
    graph = Graph(query="q")

    assert flow.run(graph=graph) == "root|1|3"
    assert "RLMFLOW_AGENT_ID" not in os.environ
    assert flow.get_env_var(graph, "RLMFLOW_AGENT_ID") == "root"
    assert flow.get_env_var(graph, "RLMFLOW_MAX_DEPTH") == "3"



def test_minimal_max_budget_stops_run_when_token_cap_reached():
    flow = Flow(
        UsageLLM('```repl\nprint("loop")\n```', input_tokens=3, output_tokens=3),
        max_budget=5,
        max_iters=10,
    )

    assert flow.run(graph=Graph(query="q")) == "[budget exceeded]"



def test_minimal_terminate_requests_agent_stop():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("never")\n```'))
    flow.terminate(["root"])

    assert flow.run(graph=Graph(query="q")) == "[terminated]"



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

    assert flow.run(graph=Graph(query="parent")) == "[max_iters exceeded]"



def test_minimal_llm_request_timeout_raises_on_slow_client():
    class SlowLLM:
        async def chat(self, _messages):
            await asyncio.sleep(1)
            return "x"

    flow = Flow(SlowLLM(), llm_request_timeout=0.01)

    raised = False
    try:
        flow.run(graph=Graph(query="q"))
    except TimeoutError:
        raised = True
    assert raised


def test_llm_request_timeout_pushes_down_to_blocking_client():
    # asyncio.wait_for cannot cancel a blocking client running on a pool thread,
    # so the request timeout must be pushed down to the request itself.
    seen = {}

    class BlockingLLM:
        def chat(self, _messages, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            return '```repl\ndone("ok")\n```'

    flow = Flow(BlockingLLM(), llm_request_timeout=5)
    assert flow.run(graph=Graph(query="q")) == "ok"
    assert seen["timeout"] == 5


def test_llm_request_timeout_bounds_hung_blocking_client():
    # A blocking SDK that honors `timeout` and raises must abort the run quickly,
    # not block for its default (the bug: it hung ~600s despite wait_for).
    import time

    class HungSDK:
        def chat(self, _messages, **kwargs):
            if kwargs.get("timeout") is not None:
                raise TimeoutError(f"exceeded {kwargs['timeout']}s")
            time.sleep(60)
            return 'done("x")'

    flow = Flow(HungSDK(), llm_request_timeout=0.01)
    raised = False
    try:
        flow.run(graph=Graph(query="q"))
    except TimeoutError:
        raised = True
    assert raised


def test_llm_request_timeout_skips_client_without_timeout_kwarg():
    # A lean blocking client that does not accept `timeout` must not be handed
    # one (no TypeError), even with a request timeout configured.
    class LeanBlocking:
        def chat(self, _messages):
            return '```repl\ndone("ok")\n```'

    flow = Flow(LeanBlocking(), llm_request_timeout=5)
    assert flow.run(graph=Graph(query="q")) == "ok"

