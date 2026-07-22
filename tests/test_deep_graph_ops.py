"""Deep-tree graph surgery coverage (max_depth > 2).

See ``docs/internal/deep_tree_graph_ops_test_plan.md``.
"""

from __future__ import annotations

import asyncio

import pytest

from rlmflow import (
    ExecOutput,
    Graph,
    LLMOutput,
    SupervisingOutput,
    enable_graph_ops,
    replay,
)
from rlmflow.utils.helpers import repl_key

from helpers import (
    DEEP_TREE_AGENT_IDS,
    DEPTH1_AGENT_IDS,
    build_deep_tree,
    build_depth1_tree,
    first_supervising,
    run_deep_tree,
    run_depth1_tree,
    worker_step,
)


# ---------------------------------------------------------------------------
# A. Fixture / structural invariants
# ---------------------------------------------------------------------------


def test_deep_tree_fixture_shape():
    flow, root = run_deep_tree()

    assert [a.agent_id for a in root.walk()] == list(DEEP_TREE_AGENT_IDS)
    assert root.depth == 0
    assert root["root.m0"].depth == 1
    assert root["root.m0.l0"].depth == 2
    assert root["root.m0"].parent_agent_id == "root"
    assert root["root.m0.l1"].parent_agent_id == "root.m0"
    assert set(root.children) == {"root.m0", "root.m1"}
    assert set(root["root.m0"].children) == {"root.m0.l0", "root.m0.l1"}
    assert set(root["root.m1"].children) == {"root.m1.l0"}

    for agent_id in ("root", "root.m0", "root.m1"):
        sup = first_supervising(root[agent_id])
        assert isinstance(sup, SupervisingOutput)
        assert sup.waiting_on
        for child_id in sup.waiting_on:
            assert child_id in root[agent_id].children

    assert root.result() == "leaf+leaf|leaf"
    assert flow.runs == {}


def test_deep_tree_repls_are_keyed_independently():
    flow, root = run_deep_tree()

    assert flow.get_var(root, "root_var") == "root"
    assert flow.get_var(root["root.m0"], "mid_var") == "m0"
    assert flow.get_var(root["root.m1"], "mid_var") == "m1"
    assert flow.get_var(root["root.m0.l0"], "leaf_var") == "root.m0.l0"
    assert flow.get_var(root["root.m0.l1"], "leaf_var") == "root.m0.l1"
    assert flow.get_var(root["root.m1.l0"], "leaf_var") == "root.m1.l0"

    keys = {repl_key(a) for a in root.walk() if repl_key(a) in flow.repls}
    assert ("root" in {k[1] for k in keys}) or repl_key(root) in flow.repls
    assert len({k[1] for k in flow.repls if k[0] == root.graph_id}) >= 3


# ---------------------------------------------------------------------------
# B. Fork on deep trees
# ---------------------------------------------------------------------------


def test_fork_deep_tree_remaps_all_graph_ids():
    async def run():
        flow, root = await build_deep_tree()
        parent_ids = [a.agent_id for a in root.walk()]
        parent_gid = root.graph_id
        branch = await flow.fork(root)
        return root, branch, parent_ids, parent_gid

    root, branch, parent_ids, parent_gid = asyncio.run(run())

    assert branch.graph_id != parent_gid
    assert all(a.graph_id == branch.graph_id for a in branch.walk())
    assert [a.agent_id for a in branch.walk()] == parent_ids
    assert [a.agent_id for a in root.walk()] == parent_ids
    assert root.graph_id == parent_gid
    assert all(a.graph_id == parent_gid for a in root.walk())


def test_fork_deep_tree_replay_restores_targeted_repl_only():
    """Policy: ``Flow.fork(mode="replay")`` rebuilds only the targeted agent's REPL.

    Descendant REPLs on the branch stay cold until ``rebuild_repl`` / first use.
    ``launch_subagents`` short-circuits on replay so the node stream is not polluted.
    """

    async def run():
        flow, root = await build_deep_tree()
        before_types = [n.type for n in root.nodes]
        branch = await flow.fork(root, mode="replay")
        root_ns = dict(flow.repl_for(branch).namespace)
        # Fresh mid REPL (not rebuilt) — no mid_var from the parent's history.
        mid_ns = dict(flow.repl_for(branch["root.m0"]).namespace)
        return before_types, [n.type for n in branch.nodes], root_ns, mid_ns

    before_types, after_types, root_ns, mid_ns = asyncio.run(run())

    assert after_types == before_types  # no extra supervising/resume from replay
    assert root_ns.get("root_var") == "root"
    assert "mid_var" not in mid_ns


def test_fork_mid_agent_by_agent_id():
    async def run():
        flow, root = await build_deep_tree()
        mid = root["root.m0"]
        # Extra decision turn so a mid-targeted rewind/fork cut is visible.
        mid.commit(LLMOutput(content="extra mid", code="print('extra')"))
        cut = mid.nodes[-1]
        root_len = len(root.nodes)
        m1_len = len(root["root.m1"].nodes)
        branch = await flow.fork(
            root, agent_id="root.m0", from_node_id=cut.id, mode="lazy"
        )
        return root, branch, root_len, m1_len, len(mid.nodes)

    root, branch, root_len, m1_len, mid_len = asyncio.run(run())

    assert len(root.nodes) == root_len
    assert len(root["root.m0"].nodes) == mid_len  # parent untouched
    assert len(branch.nodes) == root_len
    assert len(branch["root.m1"].nodes) == m1_len
    # Branch mid dropped the cut turn (lazy mode then appends a fork note).
    assert all(
        getattr(n, "content", None) != "extra mid" for n in branch["root.m0"].nodes
    )
    assert any(
        "forked" in (getattr(n, "content", None) or "").lower()
        for n in branch["root.m0"].nodes
    )


def test_fork_before_supervising_output_prunes_orphaned_children():
    """Policy: rewind/fork-cut past a launch prunes children (RemoveNode + prune)."""
    _, root = run_deep_tree()
    sup = first_supervising(root)
    branch = root.fork(from_node_id=sup.id)

    assert list(branch.children) == []
    assert [a.agent_id for a in branch.walk()] == ["root"]
    # Parent tree untouched.
    assert set(root.children) == {"root.m0", "root.m1"}
    assert "root.m0.l0" in root


def test_fork_keep_children_false_clears_only_targeted_agent_children():
    _, root = run_deep_tree()
    branch = root.fork(agent_id="root.m0", keep_children=False)

    assert branch["root.m0"].children == {}
    assert "root.m0.l0" not in branch
    # Sibling mid (and its leaf) still present on the deepcopy.
    assert "root.m1" in branch
    assert "root.m1.l0" in branch
    assert set(root["root.m0"].children) == {"root.m0.l0", "root.m0.l1"}


# ---------------------------------------------------------------------------
# C. Rewind
# ---------------------------------------------------------------------------


def test_rewind_mid_drops_last_n_mid_turns_not_root():
    async def run():
        flow, root = await build_deep_tree()
        mid = root["root.m0"]
        mid.commit(LLMOutput(content="m-turn-a", code="a()"))
        mid.commit(LLMOutput(content="m-turn-b", code="b()"))
        root_llm = sum(isinstance(n, LLMOutput) for n in root.nodes)
        mid_llm = sum(isinstance(n, LLMOutput) for n in mid.nodes)
        branch = await flow.rewind(root, agent_id="root.m0", n=1, mode="lazy")
        return root, branch, root_llm, mid_llm

    root, branch, root_llm, mid_llm = asyncio.run(run())

    assert sum(isinstance(n, LLMOutput) for n in root.nodes) == root_llm
    assert sum(isinstance(n, LLMOutput) for n in root["root.m0"].nodes) == mid_llm
    assert sum(isinstance(n, LLMOutput) for n in branch.nodes) == root_llm
    assert sum(isinstance(n, LLMOutput) for n in branch["root.m0"].nodes) == mid_llm - 1
    codes = [n.code for n in branch["root.m0"].nodes if isinstance(n, LLMOutput)]
    assert "b()" not in codes
    assert "a()" in codes


def test_rewind_past_delegation_prunes_orphaned_children():
    async def run():
        flow, root = await build_deep_tree()
        # Root has one decision turn (the launch). Rewinding it drops past the
        # SupervisingOutput / ResumeAction and must prune m0/m1 (+ grandchildren).
        branch = await flow.rewind(root, n=1, mode="lazy")
        return root, branch

    root, branch = asyncio.run(run())

    assert set(root.children) == {"root.m0", "root.m1"}
    assert list(branch.children) == []
    assert [a.agent_id for a in branch.walk()] == ["root"]


def test_tool_rewind_ambient_deep_descendant():
    async def run():
        flow, root = await build_deep_tree()
        enable_graph_ops(flow)
        tools = flow.build_tools(root, flow.repl_for(root))
        leaf = root["root.m0.l0"]
        leaf.commit(LLMOutput(content="leaf-extra", code="noop()"))
        before = sum(isinstance(n, LLMOutput) for n in leaf.nodes)
        branch = await tools["rewind"]("root.m0.l0", n=1, mode="lazy")
        return before, branch, root

    before, branch, root = asyncio.run(run())

    assert branch.graph_id != root.graph_id
    assert sum(isinstance(n, LLMOutput) for n in branch["root.m0.l0"].nodes) == before - 1
    assert sum(isinstance(n, LLMOutput) for n in root["root.m0.l0"].nodes) == before


# ---------------------------------------------------------------------------
# D. Merge
# ---------------------------------------------------------------------------


def test_merge_flat_delta_into_parent_with_unrelated_deep_children():
    async def run():
        flow, parent = await build_deep_tree()
        children_before = set(parent.children)
        branch = await flow.fork(parent, mode="lazy")
        await worker_step(flow, branch, "merged_flag = 1")
        await flow.merge(parent, branch)
        return parent, children_before, flow.repl_for(parent).namespace

    parent, children_before, ns = asyncio.run(run())

    assert set(parent.children) == children_before
    assert "root.m0.l0" in parent
    assert ns.get("merged_flag") == 1


def test_merge_does_not_import_child_subtree():
    """Policy: merge is agent-local — child agents on the branch are not folded."""

    async def run():
        flow, parent = await build_deep_tree()
        before = set(parent["root.m0"].children)
        branch = await flow.fork(parent, agent_id="root.m0", mode="lazy")
        ghost = Graph(
            agent_id="root.m0.ghost",
            graph_id=branch.graph_id,
            query="ghost",
            depth=2,
            parent_agent_id="root.m0",
        )
        branch["root.m0"].children["root.m0.ghost"] = ghost
        await worker_step(flow, branch["root.m0"], "mid_extra = 7")
        await flow.merge(parent, branch, agent_id="root.m0")
        return parent, before, flow.get_var(parent["root.m0"], "mid_extra")

    parent, before, mid_extra = asyncio.run(run())

    assert set(parent["root.m0"].children) == before
    assert "root.m0.ghost" not in parent["root.m0"].children
    assert mid_extra == 7


def test_merge_with_agent_id_on_mid():
    async def run():
        flow, parent = await build_deep_tree()
        leaf_nodes = len(parent["root.m0.l0"].nodes)
        root_var = flow.get_var(parent, "root_var")
        branch = await flow.fork(parent, agent_id="root.m0", mode="lazy")
        await worker_step(flow, branch["root.m0"], "mid_merged = 42")
        await flow.merge(parent, branch, agent_id="root.m0")
        return (
            flow.get_var(parent["root.m0"], "mid_merged"),
            flow.get_var(parent, "root_var"),
            root_var,
            len(parent["root.m0.l0"].nodes),
            leaf_nodes,
        )

    mid_merged, root_var_after, root_var_before, leaf_after, leaf_before = asyncio.run(
        run()
    )

    assert mid_merged == 42
    assert root_var_after == root_var_before == "root"
    assert leaf_after == leaf_before


# ---------------------------------------------------------------------------
# E. Discard / adopt / checkpoint
# ---------------------------------------------------------------------------


def test_discard_deep_branch_closes_all_descendant_repls():
    async def run():
        flow, root = await build_deep_tree()
        branch = await flow.fork(root, mode="lazy")
        # Materialize mid + leaf REPLs under the branch graph_id.
        await flow.rebuild_repl(branch, agent_id="root.m0")
        await flow.rebuild_repl(branch, agent_id="root.m0.l0")
        keys_before = [k for k in flow.repls if k[0] == branch.graph_id]
        flow.discard(branch)
        keys_after = [k for k in flow.repls if k[0] == branch.graph_id]
        return keys_before, keys_after

    keys_before, keys_after = asyncio.run(run())

    assert len(keys_before) >= 2
    assert keys_after == []


def test_adopt_rejects_multi_agent_child():
    """Policy: adopt is single-agent only (docstring contract, now enforced)."""

    async def run():
        flow, root = await build_deep_tree()
        deep_fork = await flow.fork(root, mode="lazy")
        host = Graph(query="host")
        return flow.adopt(host, deep_fork, name="x")

    with pytest.raises(ValueError, match="single-agent"):
        asyncio.run(run())


def test_checkpoint_revert_mid_leaves_siblings():
    _, root = run_deep_tree()
    mid = root["root.m0"]
    leaf = root["root.m1.l0"]
    cp = root.checkpoint(agent_id="root.m0")
    mid_pos = cp.position
    leaf_before = len(leaf.nodes)

    mid.commit(LLMOutput(content="mid-after-cp", code="x()"))
    leaf.commit(LLMOutput(content="leaf-grew", code="y()"))

    root.revert(cp)

    assert len(mid.nodes) == mid_pos
    assert all(
        getattr(n, "content", None) != "mid-after-cp" for n in mid.nodes
    )
    assert len(leaf.nodes) == leaf_before + 1
    assert leaf.nodes[-1].content == "leaf-grew"
    assert root.checkpoint(agent_id="root.m0").digest == cp.digest


# ---------------------------------------------------------------------------
# F. Replace / timeline / persistence
# ---------------------------------------------------------------------------


def test_replace_truncate_descendants_prunes_grandchildren():
    _, root = run_deep_tree()
    sup = first_supervising(root)

    root.replace(
        sup,
        ExecOutput(output="reroute", content="reroute"),
        truncate="descendants",
    )

    assert "root.m0" not in root.children
    assert "root.m1" not in root.children
    assert "root.m0.l0" not in root
    assert [a.agent_id for a in root.walk()] == ["root"]
    assert root.nodes[-1].content == "reroute"


def test_timeline_on_deep_tree_after_rewind():
    _, root = run_deep_tree()
    sup = first_supervising(root)
    branch = root.fork(from_node_id=sup.id)

    timeline = replay(branch)
    assert timeline
    # Every snapshot's agent set must be consistent with surviving waiting_on edges
    # (no ghost children after the prune).
    for snap in timeline:
        for agent in snap.walk():
            for node in agent.nodes:
                if isinstance(node, SupervisingOutput):
                    for child_id in node.waiting_on:
                        assert child_id in agent.children
        # After cutting before the root launch, no snapshot should carry mids.
        assert "root.m0" not in snap


def test_save_load_roundtrip_after_deep_fork(tmp_path):
    async def run():
        flow, root = await build_deep_tree()
        branch = await flow.fork(root, mode="lazy")
        path = branch.save(tmp_path / "deep-fork")
        loaded = Graph.load(path)
        return branch, loaded

    branch, loaded = asyncio.run(run())

    assert [a.agent_id for a in loaded.walk()] == [a.agent_id for a in branch.walk()]
    assert {a.agent_id: len(a.nodes) for a in loaded.walk()} == {
        a.agent_id: len(a.nodes) for a in branch.walk()
    }
    assert loaded["root.m0.l0"].result() == branch["root.m0.l0"].result()


# ---------------------------------------------------------------------------
# G. Composed smoke
# ---------------------------------------------------------------------------


def test_deep_best_of_n_style_loop():
    """fork mid → work → merge → discard (Shepherd/autoresearch-shaped smoke)."""

    async def run():
        flow, parent = await build_deep_tree()
        branch = await flow.fork(parent, agent_id="root.m0", mode="lazy")
        await worker_step(flow, branch["root.m0"], "best = 11")
        await flow.merge(parent, branch, agent_id="root.m0", summary="picked branch")
        flow.discard(branch)
        return (
            flow.get_var(parent["root.m0"], "best"),
            any(
                "picked branch" in (n.content or "")
                for n in parent["root.m0"].nodes
                if n.type == "exec_output"
            ),
            any(k[0] == branch.graph_id for k in flow.repls),
            "root.m0.l0" in parent,
        )

    best, has_summary, branch_repl_left, leaf_ok = asyncio.run(run())

    assert best == 11
    assert has_summary
    assert not branch_repl_left
    assert leaf_ok


def test_launch_subagents_replay_short_circuit_keeps_finished_children():
    """Direct unit check for the rebuild/fork replay path in builtins."""

    async def run():
        flow, root = await build_deep_tree()
        before = [n.type for n in root.nodes]
        children_before = set(root.children)
        launch = flow.launch_subagents(root, flow.repl_for(root))
        results = await launch(
            [{"name": "m0", "query": "mid0"}, {"name": "m1", "query": "mid1"}]
        )
        return before, [n.type for n in root.nodes], children_before, set(root.children), results

    before, after, children_before, children_after, results = asyncio.run(run())

    assert after == before
    assert children_after == children_before
    assert results == ["leaf+leaf", "leaf"]


# ---------------------------------------------------------------------------
# Leftovers: close every matrix cell
# ---------------------------------------------------------------------------


def test_fork_mid_replay_restores_mid_repl_leaves_cold():
    """``Flow.fork(agent_id=mid, mode="replay")`` rebuilds mid only; leaves stay cold."""

    async def run():
        flow, root = await build_deep_tree()
        mid_types = [n.type for n in root["root.m0"].nodes]
        branch = await flow.fork(root, agent_id="root.m0", mode="replay")
        mid_ns = dict(flow.repl_for(branch["root.m0"]).namespace)
        leaf_ns = dict(flow.repl_for(branch["root.m0.l0"]).namespace)
        return mid_types, [n.type for n in branch["root.m0"].nodes], mid_ns, leaf_ns

    mid_types, after_types, mid_ns, leaf_ns = asyncio.run(run())

    assert after_types == mid_types
    assert mid_ns.get("mid_var") == "m0"
    assert "leaf_var" not in leaf_ns


def test_replace_truncate_mid_via_agent_id_prunes_leaves_only():
    """``replace(..., agent_id=mid, truncate="descendants")`` drops that mid's leaves."""
    _, root = run_deep_tree()
    mid = root["root.m0"]
    sup = first_supervising(mid)

    root.replace(
        sup,
        ExecOutput(output="mid reroute", content="mid reroute"),
        agent_id="root.m0",
        truncate="descendants",
    )

    assert root["root.m0"].children == {}
    assert "root.m0.l0" not in root
    assert "root.m0.l1" not in root
    # Sibling mid subtree untouched.
    assert "root.m1" in root.children
    assert "root.m1.l0" in root
    assert root["root.m0"].nodes[-1].content == "mid reroute"


def test_rebuild_repl_mid_via_agent_id():
    async def run():
        flow, root = await build_deep_tree()
        flow.close_repl(root["root.m0"])
        assert repl_key(root["root.m0"]) not in flow.repls
        repl = await flow.rebuild_repl(root, agent_id="root.m0")
        return repl.namespace.get("mid_var"), flow.get_var(root["root.m0"], "mid_var")

    mid_var, via_get = asyncio.run(run())
    assert mid_var == "m0"
    assert via_get == "m0"


def test_tool_fork_run_merge_discard_on_deep_descendants():
    """Ambient graph-ops tools: fork / run / merge / discard over deep agent ids."""

    async def run():
        flow, root = await build_deep_tree()
        enable_graph_ops(flow)
        tools = flow.build_tools(root, flow.repl_for(root))

        # fork mid subtree (resolve by agent_id → deepcopy of that agent)
        forked = await tools["fork"]("root.m0", mode="lazy")
        assert forked.agent_id == "root.m0"
        assert forked.graph_id != root.graph_id
        assert "root.m0.l0" in forked

        await worker_step(flow, forked, "tool_merged = 3")
        await tools["merge"]("root.m0", forked, summary="tool merge")
        assert flow.get_var(root["root.m0"], "tool_merged") == 3

        # run: continue a finished deep leaf on a fresh branch
        leaf_branch = await tools["fork"]("root.m0.l0", mode="lazy")
        leaf_branch.append_query("again")
        ran = await tools["run"](leaf_branch)
        assert ran is leaf_branch
        assert leaf_branch.result() == "leaf"
        assert any(
            n.type == "user_query" and n.content == "again" for n in leaf_branch.nodes
        )

        tools["discard"](forked, leaf_branch)
        assert not any(
            k[0] in {forked.graph_id, leaf_branch.graph_id} for k in flow.repls
        )
        return True

    assert asyncio.run(run())


# ---------------------------------------------------------------------------
# Depth-1 column (root → root.a)
# ---------------------------------------------------------------------------


def test_depth1_fixture_shape():
    _, root = run_depth1_tree()
    assert [a.agent_id for a in root.walk()] == list(DEPTH1_AGENT_IDS)
    assert root["root.a"].depth == 1
    assert root["root.a"].result() == "child"


def test_depth1_graph_fork_structure():
    _, root = run_depth1_tree()
    kept = root.fork(keep_children=True)
    dropped = root.fork(from_node_id=first_supervising(root).id, keep_children=True)

    assert "root.a" in kept
    assert all(a.graph_id == kept.graph_id for a in kept.walk())
    assert list(dropped.children) == []  # prune after cut before/at supervising


def test_depth1_flow_fork_replay_and_lazy():
    async def run():
        flow, root = await build_depth1_tree()
        replayed = await flow.fork(root, mode="replay")
        lazy = await flow.fork(root, mode="lazy")
        return (
            flow.repl_for(replayed).namespace.get("root_var"),
            "root_var" in flow.repl_for(lazy).namespace,
            "child_var" in flow.repl_for(replayed["root.a"]).namespace,
            lazy.nodes[-1].type,
        )

    root_var, lazy_has_root, child_warm, lazy_note_type = asyncio.run(run())
    assert root_var == "root"
    assert not lazy_has_root  # lazy: cold until touched
    assert not child_warm  # replay rebuilds targeted (root) only
    assert lazy_note_type == "exec_output"


def test_depth1_rewind_merge_discard():
    async def run():
        flow, root = await build_depth1_tree()
        child = root["root.a"]
        child.commit(LLMOutput(content="extra", code="extra()"))
        branch = await flow.rewind(root, agent_id="root.a", n=1, mode="lazy")
        assert sum(isinstance(n, LLMOutput) for n in branch["root.a"].nodes) == sum(
            isinstance(n, LLMOutput) for n in root["root.a"].nodes
        ) - 1

        work = await flow.fork(root, mode="lazy")
        await worker_step(flow, work, "d1 = 1")
        await flow.merge(root, work)
        assert flow.get_var(root, "d1") == 1

        flow.discard(work, branch)
        return not any(k[0] in {work.graph_id, branch.graph_id} for k in flow.repls)

    assert asyncio.run(run())


def test_depth1_checkpoint_rebuild_timeline_save(tmp_path):
    async def run():
        flow, root = await build_depth1_tree()
        child = root["root.a"]
        cp = root.checkpoint(agent_id="root.a")
        child.commit(LLMOutput(content="after", code="z()"))
        root.revert(cp)
        assert all(getattr(n, "content", None) != "after" for n in child.nodes)

        flow.close_repl(child)
        await flow.rebuild_repl(root, agent_id="root.a")
        assert flow.get_var(child, "child_var") == "a"

        snaps = replay(root)
        assert snaps and "root.a" in snaps[-1]

        path = root.save(tmp_path / "d1")
        loaded = Graph.load(path)
        return [a.agent_id for a in loaded.walk()], loaded["root.a"].result()

    ids, result = asyncio.run(run())
    assert ids == list(DEPTH1_AGENT_IDS)
    assert result == "child"
