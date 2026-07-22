import asyncio

import pytest

from rlmflow import (
    Flow,
    Graph,
    LLMOutput,
    enable_graph_ops,
    get_tool_metadata,
    inject_tools,
    register_halt,
    tool,
)

from helpers import StubLLM

DONE = StubLLM(lambda _messages: '```repl\ndone("ok")\n```')


def _controller_with_child(flow, query="parent"):
    """A controller graph that has spawned one child agent ``root.a``."""
    parent = Graph(query=query)
    repl = flow.repl_for(parent)
    launch = flow.launch_subagents(parent, repl)
    asyncio.run(launch([{"name": "a", "query": "child task"}]))
    return parent, repl


def test_enable_graph_ops_flow_wide_advertises_and_injects():
    flow = Flow(DONE)
    enable_graph_ops(flow)

    graph = Graph(query="q")
    tools = flow.build_tools(graph, flow.repl_for(graph))

    for name in ("fork", "rewind", "run", "step", "merge", "discard"):
        assert name in tools
        assert get_tool_metadata(tools[name]) is not None
    # async ops advertise the await prefix; discard is sync.
    assert get_tool_metadata(tools["fork"]).is_async is True
    assert get_tool_metadata(tools["discard"]).is_async is False


def test_ambient_authority_over_descendants():
    flow = Flow(DONE, max_depth=1)
    enable_graph_ops(flow)
    parent, repl = _controller_with_child(flow)

    tools = flow.build_tools(parent, repl)
    forked = asyncio.run(tools["fork"]("root.a"))

    assert isinstance(forked, Graph)
    assert forked.graph_id != parent.graph_id


def test_rewind_forks_dropping_the_last_n_decisions():
    flow = Flow(DONE)
    enable_graph_ops(flow)
    controller = Graph(query="c")
    tools = flow.build_tools(controller, flow.repl_for(controller))

    # A trajectory with three decision turns.
    target = Graph(query="sub")
    for i in range(3):
        target.commit(LLMOutput(content=f"turn {i}", code=f"print({i})"))

    branch = asyncio.run(tools["rewind"](target, n=2))

    assert branch.graph_id != target.graph_id
    # Original untouched; branch dropped the last two decision turns.
    assert sum(isinstance(node, LLMOutput) for node in target.nodes) == 3
    assert sum(isinstance(node, LLMOutput) for node in branch.nodes) == 1


def test_rewind_where_counts_only_matching_turns():
    flow = Flow(DONE)
    target = Graph(query="sub")
    # setup + two real moves + a trailing no-op decision.
    target.commit(LLMOutput(content="setup", code="setup()"))
    target.commit(LLMOutput(content="move a", code="move('a')"))
    target.commit(LLMOutput(content="move b", code="move('b')"))
    target.commit(LLMOutput(content="stop", code="done('x')"))

    def is_move(node):
        return "move(" in (node.code or "")

    # n=1 over moves must undo the last real move, not the trailing done turn.
    branch = asyncio.run(flow.rewind(target, n=1, where=is_move))

    codes = [n.code for n in branch.nodes if isinstance(n, LLMOutput)]
    assert codes == ["setup()", "move('a')"]


def test_rewind_rejects_out_of_range_n():
    flow = Flow(DONE)
    enable_graph_ops(flow)
    controller = Graph(query="c")
    tools = flow.build_tools(controller, flow.repl_for(controller))

    target = Graph(query="sub")
    target.commit(LLMOutput(content="only turn", code="print(1)"))

    with pytest.raises(ValueError):
        asyncio.run(tools["rewind"](target, n=0))
    with pytest.raises(ValueError):
        asyncio.run(tools["rewind"](target, n=5))


def test_authority_refusal_for_unrelated_graph():
    flow = Flow(DONE)
    enable_graph_ops(flow)
    parent = Graph(query="p")

    tools = flow.build_tools(parent, flow.repl_for(parent))
    with pytest.raises(KeyError):
        asyncio.run(tools["fork"]("not-a-descendant"))


def test_targeted_scope_only_controller_gets_tools():
    flow = Flow(DONE)
    controller = Graph(query="controller")
    other = Graph(query="other")
    enable_graph_ops(flow, controller)

    ctrl_tools = flow.build_tools(controller, flow.repl_for(controller))
    other_tools = flow.build_tools(other, flow.repl_for(other))

    assert "fork" in ctrl_tools
    assert "fork" not in other_tools


def test_granted_roots_are_forkable_and_bound_by_name():
    flow = Flow(DONE)
    controller = Graph(query="controller")
    worker = Graph(query="worker")
    enable_graph_ops(flow, controller, roots={"worker": worker})

    tools = flow.build_tools(controller, flow.repl_for(controller))

    assert tools["worker"] is worker
    forked = asyncio.run(tools["fork"]("worker"))
    assert forked.graph_id != worker.graph_id


def test_roots_without_controller_is_rejected():
    flow = Flow(DONE)
    with pytest.raises(ValueError):
        enable_graph_ops(flow, roots={"worker": Graph(query="w")})


def test_run_tool_drives_a_graph_to_done():
    flow = Flow(DONE)
    enable_graph_ops(flow)
    controller = Graph(query="controller")
    tools = flow.build_tools(controller, flow.repl_for(controller))

    target = Graph(query="sub")
    ran = asyncio.run(tools["run"](target))

    assert ran is target
    assert target.result() == "ok"


def test_step_advances_one_event():
    flow = Flow(DONE)
    enable_graph_ops(flow)
    controller = Graph(query="c")
    tools = flow.build_tools(controller, flow.repl_for(controller))

    target = Graph(query="sub")
    before = len(target.nodes)

    stepped = asyncio.run(tools["step"](target))

    assert stepped is target
    assert len(target.nodes) == before + 1
    assert not target.finished


def test_register_halt_name_governs_a_run():
    flow = Flow(DONE)

    def solved(_event, _graph):
        return True

    register_halt(flow, "solved", solved)
    # Registration just names the predicate on the flow.
    assert flow.halts["solved"] is solved

    # run_streaming resolves the name itself — no resolve_halt needed at call sites.
    seen: list[int] = []
    register_halt(flow, "watch", lambda _e, _g: bool(seen.append(1)))

    target = Graph(query="q")

    async def drive():
        async for _ in flow.run_streaming(graph=target, until="watch"):
            pass

    asyncio.run(drive())
    assert seen  # the named predicate was consulted while driving the run


def test_inject_tools_static_list():
    flow = Flow(DONE)

    @tool("Add two numbers.")
    def add(a, b):
        return a + b

    inject_tools(flow, [add])

    tools = flow.build_tools(Graph(query="q"))
    assert tools["add"] is add


def test_inject_tools_backfills_open_repls():
    flow = Flow(DONE)
    controller = Graph(query="controller")
    # Materialize the REPL first, so backfill (not build_tools) is exercised.
    repl = flow.repl_for(controller)
    # Register a live run so open_repls can pair the graph with its REPL.
    from rlmflow.flow import Run
    from rlmflow.tasks import TaskQueue

    flow.runs[controller.graph_id] = Run(graph=controller, tasks=TaskQueue())

    @tool("Add.")
    def add(a, b):
        return a + b

    inject_tools(flow, [add])
    assert repl.get_var("add") is add
