"""Phase 0 graph enablers: serialization, predicates, flat views, copy, step counters.

These exercise the pure data-model surface added in Phase 0 — independent of the
engine, but also checked against a real (delegated) run so the round-trip and views
hold on a non-trivial subtree.
"""

from __future__ import annotations

from rflow import (
    DoneOutput,
    Edge,
    ErrorOutput,
    ExecAction,
    Graph,
    LLMAction,
    LLMOutput,
    SupervisingOutput,
    UserQuery,
    is_action,
    is_done,
    is_errored,
    is_observation,
    is_resumed,
    is_supervising,
    parse_node_obj,
)
from rflow.graph import EdgesView, NodesView
from tests.helpers import ScriptedLLM, run_to_completion


def _tight_parent_child(messages):
    task = next((m["content"] for m in messages if m["role"] == "user"), "")
    if "depth 1" in task:
        return '```repl\ndone("c")\n```'
    return (
        "```repl\n"
        'results = await launch_subagents([{"name": "child", "query": "child task"}])\n'
        'done("p:" + results[0])\n'
        "```"
    )


def _delegated_graph() -> Graph:
    from rflow import Flow

    flow = Flow(ScriptedLLM(_tight_parent_child), max_depth=1, max_iters=5)
    return run_to_completion(flow, "parent")


# ── serialization round-trip ──────────────────────────────────────────


def test_to_dict_includes_system_prompt():
    g = Graph(agent_id="root", query="q", system_prompt="SYS")
    assert g.to_dict()["system_prompt"] == "SYS"


def test_repl_inputs_excludes_query():
    g = Graph(agent_id="root", query="q", inputs={"doc": "text"})
    # The query is delivered as the first user message, not mirrored into INPUTS.
    assert g.repl_inputs() == {"doc": "text"}


def test_from_dict_round_trips_a_delegated_run():
    g = _delegated_graph()
    restored = Graph.from_dict(g.to_dict())

    assert isinstance(restored, Graph)
    assert restored.agent_id == g.agent_id
    assert restored.system_prompt == g.system_prompt
    assert list(restored.children) == list(g.children)
    assert [n.type for n in restored.nodes] == [n.type for n in g.nodes]
    assert restored["root.child"].result() == g["root.child"].result()
    assert restored.result() == g.result() == "p:c"
    # full structural equality of the serialized form
    assert restored.to_dict() == g.to_dict()


def test_from_dict_rebuilds_concrete_node_types():
    g = _delegated_graph()
    restored = Graph.from_dict(g.to_dict())
    by_type = {n.type: type(n) for n in restored.nodes}
    assert by_type["user_query"] is UserQuery
    assert by_type["llm_output"] is LLMOutput
    assert by_type["supervising_output"] is SupervisingOutput
    assert by_type["done_output"] is DoneOutput


def test_from_dict_accepts_legacy_states_alias_and_ignores_unknown_fields():
    data = {
        "agent_id": "root",
        "query": "q",
        "states": [UserQuery(content="hi").to_dict()],  # legacy key
        "totally_unknown": 123,
    }
    g = Graph.from_dict(data)
    assert len(g.nodes) == 1
    assert isinstance(g.nodes[0], UserQuery)


def test_parse_node_obj_discriminates_on_type():
    node = parse_node_obj({"type": "done_output", "result": "answer"})
    assert isinstance(node, DoneOutput)
    assert node.result == "answer"
    assert is_done(node)


# ── predicates ────────────────────────────────────────────────────────


def test_predicates_classify_nodes():
    assert is_observation(UserQuery())
    assert not is_action(UserQuery())
    assert is_action(LLMAction())
    assert is_action(ExecAction())
    assert is_done(DoneOutput(result="x"))
    assert is_errored(ErrorOutput(error="boom"))
    assert is_supervising(SupervisingOutput(waiting_on=["root.child"]))


def test_is_resumed_only_true_for_resumed_observations():
    assert not is_resumed(DoneOutput(result="x"))
    assert is_resumed(DoneOutput(result="x", resumed_from=["root.child"]))


# ── flat views ────────────────────────────────────────────────────────


def test_all_nodes_flattens_subtree_and_filters():
    g = _delegated_graph()
    view = g.all_nodes
    assert isinstance(view, NodesView)

    # parent (7) + child (5) nodes
    assert len(view) == 12
    assert all(a.type == "llm_action" for a in view.llm_actions())
    assert view.results() and all(is_done(r) for r in view.results())
    assert {n.agent_id for n in view.where(lambda n: n.agent_id == "root.child")} == {
        "root.child"
    }
    assert view.where(type="supervising_output")  # only the parent has one


def test_all_nodes_find_locates_by_id():
    g = _delegated_graph()
    target = g.nodes[0]
    assert g.all_nodes.find(target.id) is target
    assert g.all_nodes.find("does-not-exist") is None
    assert target.id in g.all_nodes


def test_edges_derives_flow_and_spawn_edges():
    g = _delegated_graph()
    edges = g.edges
    assert isinstance(edges, EdgesView)

    spawns = edges.spawns()
    assert len(spawns) == 1
    spawn = spawns[0]
    assert isinstance(spawn, Edge) and spawn.kind == "spawns"
    # spawn goes from the parent's running action node to the child's first node
    assert spawn.to == g["root.child"].nodes[0].id
    assert spawn.from_ in {n.id for n in g.nodes}

    flows = edges.flows_to()
    assert all(e.kind == "flows_to" for e in flows)
    # within an agent, consecutive nodes are linked: (#nodes - 1) per agent
    assert len(flows) == (len(g.nodes) - 1) + (len(g["root.child"].nodes) - 1)


# ── copy + step counters ──────────────────────────────────────────────


def test_copy_is_deep_and_independent():
    g = _delegated_graph()
    clone = g.copy()  # deep by default
    assert clone is not g
    assert clone.to_dict() == g.to_dict()

    clone.nodes.append(UserQuery(content="mutated"))
    clone.children["root.child"].nodes.clear()
    assert len(g.nodes) != len(clone.nodes)
    assert g["root.child"].nodes  # original child untouched


def test_snapshot_is_deep_and_independent():
    g = _delegated_graph()
    snapshot = g.snapshot()
    assert snapshot is not g
    assert snapshot.to_dict() == g.to_dict()

    snapshot.nodes.append(UserQuery(content="mutated"))
    snapshot.children["root.child"].nodes.clear()
    assert len(g.nodes) != len(snapshot.nodes)
    assert g["root.child"].nodes


def test_shallow_copy_shares_node_list():
    g = _delegated_graph()
    shallow = g.copy(deep=False)
    assert shallow is not g
    assert shallow.nodes is g.nodes  # dataclasses.replace copies fields by reference


def test_global_step_counters_on_empty_and_populated_graphs():
    empty = Graph(agent_id="root")
    assert empty.max_global_step() is None
    assert empty.next_global_step() == 0

    g = _delegated_graph()
    top = g.max_global_step()
    assert top is not None
    assert g.next_global_step() == top + 1


# ── finished + scheduling invariants (ported from legacy graph_surgery) ─


def _supervising_root_with_child(child_finished: bool) -> Graph:
    """A root paused at ``await`` on one child, optionally still running."""
    child_nodes = [UserQuery(content="c")]
    if child_finished:
        child_nodes.append(DoneOutput(result="c-done"))
    child = Graph(agent_id="root.child", depth=1, nodes=child_nodes)
    root = Graph(
        agent_id="root",
        nodes=[
            UserQuery(content="p"),
            LLMAction(),
            LLMOutput(reply="r"),
            ExecAction(code="..."),
            SupervisingOutput(waiting_on=["root.child"]),
        ],
        children={"root.child": child},
    )
    return root


def test_finished_requires_all_descendants_finished():
    # Terminal parent but an unfinished child → the whole subtree isn't done.
    child = Graph(agent_id="root.child", depth=1, nodes=[UserQuery(content="c")])
    root = Graph(
        agent_id="root",
        nodes=[UserQuery(content="p"), DoneOutput(result="x")],
        children={"root.child": child},
    )
    assert child.finished is False
    assert root.finished is False
    child.nodes.append(DoneOutput(result="c"))
    assert root.finished is True


def test_runnable_is_child_not_paused_supervisor():
    root = _supervising_root_with_child(child_finished=False)
    # the paused supervisor must not be runnable while its child still runs.
    assert root.get_runnable_nodes() == ["root.child"]


def test_runnable_is_supervisor_once_child_finished():
    root = _supervising_root_with_child(child_finished=True)
    assert root.get_runnable_nodes() == ["root"]


def test_runnable_empty_when_waited_child_missing():
    # supervisor waits on an agent id that isn't in the tree → nothing runnable
    # (the engine can't resume it, and there's no descendant to advance).
    root = Graph(
        agent_id="root",
        nodes=[
            UserQuery(content="p"),
            SupervisingOutput(waiting_on=["root.ghost"]),
        ],
    )
    assert root.get_runnable_nodes() == []
