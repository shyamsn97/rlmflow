import asyncio
import json

from rflow import (
    AddChild,
    AppendNode,
    ExecAction,
    ExecOutput,
    Flow,
    Graph,
    GraphCheckpoint,
    GraphCreated,
    LLMOutput,
    RemoveChild,
    RemoveNode,
    ReplaceNode,
    UserQuery,
    apply_graph_action,
)
from rflow.utils import repl_key

from helpers import (
    StubLLM,
    first_user,
    seed_exec_graph,
    worker_step,
)


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



def test_minimal_checkpoint_revert_restores_nodes():
    graph = seed_exec_graph("x = 1")
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
    graph = seed_exec_graph("x = 1")
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
        graph = seed_exec_graph("x = 21 * 2", "y = x + 1")
        repl = await flow.rebuild_repl(graph)
        return repl.namespace.get("x"), repl.namespace.get("y"), len(graph.nodes)

    x, y, node_count = asyncio.run(run())
    assert (x, y) == (42, 43)
    assert node_count == 7  # replay must not append result nodes



def test_minimal_fork_is_isolated_from_parent():
    async def run():
        flow = Flow(StubLLM(lambda _messages: "unused"))
        parent = seed_exec_graph("x = 1")
        await flow.rebuild_repl(parent)

        child = await flow.fork(parent)
        await worker_step(flow, child, "y = x + 1")

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
        parent = seed_exec_graph("report = {}")
        await flow.rebuild_repl(parent)  # parent has a live REPL -> delta-run rung

        child_a = await flow.fork(parent)
        child_b = await flow.fork(parent)
        await worker_step(flow, child_a, "stats_a = 12")
        await worker_step(flow, child_b, "stats_b = 34")

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
        parent = seed_exec_graph("report = {}")  # NOTE: no live parent REPL

        child = await flow.fork(parent)
        await worker_step(flow, child, "stats_a = 12")

        await flow.merge(parent, child)  # rung 1: adopt child's REPL, zero re-exec
        return flow.repl_for(parent).namespace

    namespace = asyncio.run(run())
    assert namespace.get("report") == {}
    assert namespace.get("stats_a") == 12



def test_minimal_merge_conflict_is_last_write_wins():
    async def run():
        flow = Flow(StubLLM(lambda _messages: "unused"))
        parent = seed_exec_graph("v = 0")
        await flow.rebuild_repl(parent)

        child_a = await flow.fork(parent)
        child_b = await flow.fork(parent)
        await worker_step(flow, child_a, "v = 1")
        await worker_step(flow, child_b, "v = 2")

        await flow.merge(parent, child_a)
        await flow.merge(parent, child_b)  # merged last -> wins
        return flow.repl_for(parent).namespace.get("v")

    assert asyncio.run(run()) == 2



def test_minimal_discard_closes_branch_repls():
    async def run():
        flow = Flow(StubLLM(lambda _messages: "unused"))
        parent = seed_exec_graph("x = 1")
        await flow.rebuild_repl(parent)

        child = await flow.fork(parent)
        child_key = repl_key(child)
        present_before = child_key in flow.repls

        flow.discard(child)
        return present_before, child_key in flow.repls

    present_before, present_after = asyncio.run(run())
    assert present_before
    assert not present_after

