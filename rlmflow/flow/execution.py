"""Keep each agent's REPL in step with its nodes: run a node's code, rebuild a REPL from the graph.

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
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import (
    DEAD_KEY,
    DONE,
    READY,
    RUNNING,
    TERMINAL_KEY,
    WORKSPACE_KEY,
    Node,
    Step,
    running_step,
)
from rlmflow.graph.queries import last_workspace
from rlmflow.runtime.repl import ReplRun, ReplStatus
from rlmflow.runtime.runtime import RLMFLOW_REPLAY
from rlmflow.tools import RESERVED_TOOLS, build_namespace
from rlmflow.utils import truncate_output

if TYPE_CHECKING:
    from rlmflow.flow.flow import Flow

DEAD_REPL_NOTE = """\
[this REPL died while running this code. It will be rebuilt from the recorded nodes \
before the next run, so variables set earlier come back; anything this code \
did is gone.]"""


async def run_node(flow: Flow, g: Graph, agent: Node, node: Node) -> ReplRun | None:
    """Run one of ``agent``'s nodes in its REPL: bind its inputs, then run its code.

    The REPL first catches up on the nodes before it. A node that is edited or
    removed while it runs is left as the edit made it, and ``None`` is
    returned. Saved calls a rerun didn't make again are undone.
    """
    version = node.version
    rerun = bool(node.calls)
    log = calls.CallLog(node)
    node.state, node.started_at = RUNNING, time.time()
    try:
        async with flow.runtime.lock(agent):
            await catch_up(flow, g, agent, node.transcript()[:-1])
            forget(flow, agent)
            run = await execute(flow, Step(g, agent, node, calls=log, flow=flow))
    except asyncio.CancelledError:
        if node.state == RUNNING:
            node.state = READY
        raise
    except BaseException as exc:
        node.output, node.error = f"{type(exc).__name__}: {exc}", True
        node.state, node.finished_at = DONE, time.time()
        raise
    if node.id not in g or node.version != version:
        flow.runtime.close_repl(agent)
        return None
    dead = run.status is ReplStatus.DEAD
    if rerun and not dead:
        calls.undo_calls(g, log.unclaimed())
    node.output = truncate_output(run.output, agent.config.max_output_length)
    node.error = run.failed
    if dead:
        node.output = f"{node.output}\n{DEAD_REPL_NOTE}" if node.output else DEAD_REPL_NOTE
        node.data[DEAD_KEY] = True
    if run.submitted:
        node.result = run.answer
        # A session's code runs mid-turn; the node that ends its turn is the terminal one.
        if node.ends_turn:
            node.data[TERMINAL_KEY] = True
    if node.runs_code:
        workspace = await flow.workspaces.snapshot(agent)
        if workspace is not None:
            node.data[WORKSPACE_KEY] = workspace
    node.state, node.finished_at = DONE, time.time()
    if not dead and (repl := flow.runtime.get(agent)) is not None:
        repl.last_ran = _marker(node)
    flow.land(node)
    return run


async def prepare(flow: Flow, g: Graph, agent: Node) -> None:
    """Bring the agent's live REPL up to its last node."""
    async with flow.runtime.lock(agent):
        await catch_up(flow, g, agent, agent.nodes())


def forget(flow: Flow, agent: Node) -> None:
    """Mark the agent's REPL as behind, so the next run rebuilds it."""
    repl = flow.runtime.get(agent)
    if repl is not None:
        repl.last_ran = None


async def execute(flow: Flow, step: Step) -> ReplRun:
    """Bind the step's node's inputs, seed the step's tools, and run its code."""
    g, agent, node = step.graph, step.agent, step.node
    repl = flow.runtime.repl_for(agent, g)
    clashes = sorted(node.input.keys() & (RESERVED_TOOLS | flow.tools_for(g).keys()))
    if clashes:
        raise ValueError(f"inputs collide with REPL tools: {', '.join(clashes)}")
    with running_step(step):
        for name, value in node.input.items():
            repl.inject(name, value)
        if not node.runs_code:
            return ReplRun()
        repl.seed(build_namespace(flow, step))
        return await flow.runtime.execute(agent, node.code, g)


async def catch_up(flow: Flow, g: Graph, agent: Node, nodes: list[Node]) -> None:
    """Bring the agent's REPL up to the end of ``nodes``, rebuilding it if needed.

    Nodes that ran before run again with their saved calls, so nothing runs
    twice; nodes attached since the REPL last ran just bind their inputs.
    """
    runtime = flow.runtime
    start = _resume_index(flow, agent, nodes)
    rebuilding = start is None
    if rebuilding:
        runtime.close_repl(agent)
        start = 0
    repl = runtime.repl_for(agent, g)
    last = None
    for earlier in nodes[start:]:
        if earlier.state != DONE or earlier.dead:
            continue
        replaying = earlier.runs_code and earlier.started_at is not None
        if replaying:
            repl.update_env({RLMFLOW_REPLAY: "1"})
        step = Step(g, agent, earlier, replaying=replaying, calls=calls.CallLog(earlier), flow=flow)
        try:
            await execute(flow, step)
        finally:
            if replaying:
                repl.update_env({RLMFLOW_REPLAY: "0"})
        last = earlier
    if rebuilding:
        # Files last: rerun nodes may have written to them.
        snapshot = last_workspace(nodes)
        if snapshot is not None:
            await flow.workspaces.checkout(agent, snapshot)
    if last is not None and (live := runtime.get(agent)) is not None:
        live.last_ran = _marker(last)


def _resume_index(flow: Flow, agent: Node, nodes: list[Node]) -> int | None:
    """Where the live REPL picks up in ``nodes``; ``None`` when it must be rebuilt."""
    repl = flow.runtime.get(agent)
    marker = repl.last_ran if repl is not None else None
    if marker is None:
        return None
    return next((i + 1 for i in range(len(nodes) - 1, -1, -1) if _marker(nodes[i]) == marker), None)


def _marker(node: Node) -> tuple[str, int]:
    return node.id, node.version


__all__ = ["DEAD_REPL_NOTE", "catch_up", "execute", "forget", "prepare", "run_node"]
