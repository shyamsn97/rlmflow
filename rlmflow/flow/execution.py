"""Keep each run's REPL in step with its nodes: run a node's code, rebuild a REPL from the graph.

A REPL is up to date at ``repl.last_ran``, the last node it ran and that
node's version. Before a node runs, its REPL catches up on the nodes before
it. One that is behind, or gone, is rebuilt by running those nodes again with
their saved calls, so nothing runs twice.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from rlmflow.graph import calls
from rlmflow.graph.config import MAX_OUTPUT
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import (
    DEAD_KEY,
    DONE,
    READY,
    RUNNING,
    TERMINAL_KEY,
    WORKSPACE_KEY,
    Node,
)
from rlmflow.graph.queries import last_workspace
from rlmflow.repl import Repl, running
from rlmflow.runtime.repl import ReplRun, ReplStatus
from rlmflow.runtime.runtime import RLMFLOW_REPLAY
from rlmflow.tools.declare import RESERVED_TOOLS
from rlmflow.tools.namespace import build_namespace
from rlmflow.utils import truncate_output

if TYPE_CHECKING:
    from rlmflow.flow.flow import Flow

DEAD_REPL_NOTE = """\
[this REPL died while running this code. It will be rebuilt from the recorded nodes \
before the next run, so variables set earlier come back; anything this code \
did is gone.]"""


async def run_node(flow: Flow, g: Graph, run: Node, node: Node) -> ReplRun | None:
    """Run one of ``run``'s nodes in its REPL: bind its inputs, then run its code.

    The REPL first catches up on the nodes before it. A node that is edited or
    removed while it runs is left as the edit made it, and ``None`` is
    returned. Saved calls a rerun didn't make again are undone.
    """
    version = node.version
    rerun = bool(node.calls)
    log = calls.CallLog(node)
    node.state, node.started_at = RUNNING, time.time()
    try:
        async with flow.runtime.lock(run):
            await catch_up(flow, g, run, node.transcript()[:-1])
            forget(flow, run)
            ran = await execute(Repl(flow, run, node, calls=log))
    except asyncio.CancelledError:
        if node.state == RUNNING:
            node.state = READY
        raise
    except BaseException as exc:
        node.output, node.error = f"{type(exc).__name__}: {exc}", True
        node.state, node.finished_at = DONE, time.time()
        raise
    if node.id not in g or node.version != version:
        flow.runtime.close_repl(run)
        return None
    dead = ran.status is ReplStatus.DEAD
    if rerun and not dead:
        calls.undo_calls(g, log.unclaimed())
    node.output = truncate_output(ran.output, MAX_OUTPUT)
    node.error = ran.failed
    if dead:
        node.output = f"{node.output}\n{DEAD_REPL_NOTE}" if node.output else DEAD_REPL_NOTE
        node.data[DEAD_KEY] = True
    if ran.submitted:
        node.result = ran.answer
        # A session's code runs mid-turn; the node that ends its turn is the terminal one.
        if node.ends_turn:
            node.data[TERMINAL_KEY] = True
    # Non-code replies are finalized in Flow._step; code nodes finish here because
    # session code may run before the node that actually ends the session turn.
    if node.runs_code:
        workspace = await flow.workspaces.snapshot(run)
        if workspace is not None:
            node.data[WORKSPACE_KEY] = workspace
    node.state, node.finished_at = DONE, time.time()
    if not dead and (repl := flow.runtime.get(run)) is not None:
        repl.last_ran = _marker(node)
    flow.land(node)
    return ran


async def prepare(flow: Flow, g: Graph, run: Node) -> None:
    """Bring the run's live REPL up to its last node."""
    async with flow.runtime.lock(run):
        await catch_up(flow, g, run, run.nodes())


def forget(flow: Flow, run: Node) -> None:
    """Mark the run's REPL as behind, so the next run rebuilds it."""
    repl = flow.runtime.get(run)
    if repl is not None:
        repl.last_ran = None


async def execute(handle: Repl) -> ReplRun:
    """Bind the node's inputs, seed the run's tools, and run the node's code."""
    flow, g, run, node = handle.flow, handle.graph, handle.root, handle.node
    repl = handle._process()
    clashes = sorted(node.input.keys() & (RESERVED_TOOLS | flow.tools_for(g).keys()))
    if clashes:
        raise ValueError(f"inputs collide with REPL tools: {', '.join(clashes)}")
    with running(handle):
        for name, value in node.input.items():
            repl.inject(name, value)
        if not node.runs_code:
            return ReplRun()
        repl.seed(build_namespace(handle))
        return await flow.runtime.execute(run, node.code, g)


async def catch_up(flow: Flow, g: Graph, run: Node, nodes: list[Node]) -> None:
    """Bring the run's REPL up to the end of ``nodes``, rebuilding it if needed.

    Nodes that ran before run again with their saved calls, so nothing runs
    twice; nodes attached since the REPL last ran just bind their inputs.
    """
    runtime = flow.runtime
    start = _resume_index(flow, run, nodes)
    rebuilding = start is None
    if rebuilding:
        runtime.close_repl(run)
        start = 0
    repl = Repl(flow, run)._process()
    last = None
    for earlier in nodes[start:]:
        if earlier.state != DONE or earlier.dead:
            continue
        replaying = earlier.runs_code and earlier.started_at is not None
        if replaying:
            repl.update_env({RLMFLOW_REPLAY: "1"})
        handle = Repl(flow, run, earlier, replaying=replaying, calls=calls.CallLog(earlier))
        try:
            await execute(handle)
        finally:
            if replaying:
                repl.update_env({RLMFLOW_REPLAY: "0"})
        last = earlier
    if rebuilding:
        # Files last: rerun nodes may have written to them.
        snapshot = last_workspace(nodes)
        if snapshot is not None:
            await flow.workspaces.checkout(run, snapshot)
    if last is not None and (live := runtime.get(run)) is not None:
        live.last_ran = _marker(last)


def _resume_index(flow: Flow, run: Node, nodes: list[Node]) -> int | None:
    """Where the live REPL picks up in ``nodes``; ``None`` when it must be rebuilt."""
    repl = flow.runtime.get(run)
    marker = repl.last_ran if repl is not None else None
    if marker is None:
        return None
    return next((i + 1 for i in range(len(nodes) - 1, -1, -1) if _marker(nodes[i]) == marker), None)


def _marker(node: Node) -> tuple[str, int]:
    return node.id, node.version


__all__ = ["forget", "prepare", "run_node"]
