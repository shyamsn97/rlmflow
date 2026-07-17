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
    InsertNode,
    LLMOutput,
    RemoveChild,
    RemoveNode,
    ReplaceNode,
    SupervisingOutput,
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
        GraphCreated(type="graph_created", graph=Graph()),
    )
    graph = apply_graph_action(
        graph,
        AppendNode(
            type="append_node",
            agent_id="root",
            node=UserQuery(content="q"),
        ),
    )
    first = graph.nodes[0]
    graph = apply_graph_action(
        graph,
        ReplaceNode(
            type="replace_node",
            agent_id="root",
            node=LLMOutput(content="replacement"),
            replaced_node=first,
        ),
    )
    assert [node.type for node in graph.nodes] == ["llm_output"]
    assert graph.nodes[0].seq == 0

    graph = apply_graph_action(
        graph,
        RemoveNode(
            type="remove_node",
            agent_id="root",
            node=graph.nodes[0],
        ),
    )
    assert graph.nodes == []



def test_minimal_event_node_view_is_consistent_across_types():
    root = UserQuery(content="q")
    root.id = "n-root"
    new = LLMOutput(content="new")
    new.id = "n-new"
    old = UserQuery(content="old")
    old.id = "n-old"

    append = AppendNode(type="append_node", agent_id="root", node=root)
    assert append.node is root
    assert append.node_id == "n-root"
    assert append.node_type == "user_query"

    insert = InsertNode(type="insert_node", agent_id="root", node=root, index=0)
    assert insert.node is root
    assert insert.node_id == "n-root"

    replace = ReplaceNode(
        type="replace_node", agent_id="root", node=new, replaced_node=old
    )
    # `.node` is the NEW node; the replaced one is tracked separately.
    assert replace.node is new
    assert replace.node_id == "n-new"
    assert replace.node_type == "llm_output"
    assert replace.replaced_node is old
    assert replace.replaced_node_id == "n-old"

    remove = RemoveNode(type="remove_node", agent_id="root", node=old)
    assert remove.node is old
    assert remove.node_id == "n-old"

    graph = Graph(query="q")
    created = GraphCreated(type="graph_created", graph=graph)
    assert created.node is graph.nodes[0]
    assert created.node_type == "user_query"

    child = Graph(agent_id="root.child", query="c", parent_agent_id="root", depth=1)
    add = AddChild(type="add_child", parent_agent_id="root", child=child)
    assert add.node is child.nodes[0]

    rm_child = RemoveChild(type="remove_child", parent_agent_id="root", child=child)
    assert rm_child.child_agent_id == "root.child"
    assert rm_child.node is child.nodes[0]
    assert rm_child.node_id == child.nodes[0].id
    assert rm_child.node_type == "user_query"


def test_minimal_inject_action_builds_events_with_node_view():
    graph = Graph(query="q")
    first = graph.nodes[0]

    appended = graph.inject_action(UserQuery(content="two"))
    assert isinstance(appended, AppendNode)
    assert appended.node.content == "two"
    assert appended.node_type == "user_query"

    inserted = graph.inject_action(UserQuery(content="mid"), at=first, mode="before")
    assert isinstance(inserted, InsertNode)
    assert inserted.node.content == "mid"

    replaced = graph.inject_action(
        LLMOutput(content="repl"), at=first, mode="replace"
    )
    assert isinstance(replaced, ReplaceNode)
    assert replaced.node.content == "repl"
    assert replaced.replaced_node is first
    assert replaced.replaced_node_id == first.id

    removed = graph.rewind_action(first.id)
    assert isinstance(removed, RemoveNode)
    assert removed.node is first
    assert removed.node_id == first.id
    assert removed.subtree is True


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
            child=child,
        ),
    )
    assert "root.child" not in graph



def test_minimal_graph_operation_helpers_edit_transcripts():
    graph = Graph(query="q")
    injected = graph.inject("please verify")

    assert isinstance(injected, AppendNode)
    assert [node.content for node in graph.nodes] == ["q", "please verify"]

    first_id = graph.nodes[0].id
    graph.replace(first_id, UserQuery(content="replacement"))

    assert [node.content for node in graph.nodes] == ["replacement", "please verify"]
    assert [node.seq for node in graph.nodes] == [0, 1]

    graph.rewind(graph.nodes[1].id)

    assert [node.content for node in graph.nodes] == ["replacement"]



def test_minimal_inject_modes_and_helpers_build_off_one_primitive():
    graph = Graph(query="q")  # seeds UserQuery "q"

    # append (mode="after", at=None) -> AppendNode at the end
    appended = graph.append("after")
    assert isinstance(appended, AppendNode)

    # prepend (mode="before", at=None) -> InsertNode at the start
    prepended = graph.prepend(UserQuery(content="before"))
    assert isinstance(prepended, InsertNode)

    # anchored insert (mode="after" a specific node) -> InsertNode mid-stream
    graph.inject(UserQuery(content="between"), at=graph.nodes[1], mode="after")

    assert [node.content for node in graph.nodes] == [
        "before",
        "q",
        "between",
        "after",
    ]
    assert [node.seq for node in graph.nodes] == [0, 1, 2, 3]



def test_minimal_replace_node_truncate_descendants_reroutes_branch():
    """Replacing a delegating node with truncate drops downstream turns and the
    child agents that node spawned, re-routing the branch in one call."""
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")
    graph.commit(LLMOutput(content="delegate", code="launch..."))
    supervising = graph.commit(
        SupervisingOutput(output="spawned", waiting_on=["root.worker"])
    )
    graph.commit(ExecOutput(output="downstream turn", content="downstream turn"))
    child = Graph(
        agent_id="root.worker",
        graph_id=graph.graph_id,
        depth=1,
        parent_agent_id="root",
    )
    flow.apply_action(
        graph,
        AddChild(type="add_child", parent_agent_id="root", child=child),
    )
    assert "root.worker" in graph.children

    graph.replace(
        supervising,
        ExecOutput(output="new route", content="new route"),
        truncate="descendants",
    )

    assert [node.type for node in graph.nodes] == [
        "user_query",
        "llm_output",
        "exec_output",
    ]
    assert graph.nodes[-1].content == "new route"
    assert [node.seq for node in graph.nodes] == [0, 1, 2]
    assert "root.worker" not in graph.children  # orphaned child pruned



def test_minimal_replace_node_default_keeps_descendants_and_children():
    graph = Graph(query="q")
    graph.commit(LLMOutput(content="a"))
    graph.commit(ExecOutput(output="b", content="b"))

    graph.replace(graph.nodes[1], LLMOutput(content="edited", code="x = 1"))

    assert [node.type for node in graph.nodes] == [
        "user_query",
        "llm_output",
        "exec_output",
    ]
    assert graph.nodes[1].content == "edited"



def test_minimal_graph_fork_creates_independent_graph_branch():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    graph = Graph(query="q")
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



def test_minimal_graph_fork_keep_anchor_retains_the_cut_node():
    graph = Graph(query="q")
    graph.inject("branch point")
    graph.inject("after point")

    # Default drops the anchor (branch ends before it); keep_anchor retains it.
    dropped = graph.fork(from_node_id=graph.nodes[1].id)
    kept = graph.fork(from_node_id=graph.nodes[1].id, keep_anchor=True)

    assert [node.content for node in dropped.nodes] == ["q"]
    assert [node.content for node in kept.nodes] == ["q", "branch point"]
    # Parent is untouched by either fork.
    assert [node.content for node in graph.nodes] == ["q", "branch point", "after point"]



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
    flow.run(graph=graph)
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
    graph.replace(graph.nodes[1].id, LLMOutput(content="changed", code="x = 999"))

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

