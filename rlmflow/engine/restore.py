"""Rebuild REPL namespaces from recorded ``ExecAction``s."""

from __future__ import annotations

from typing import Any

from rlmflow.engine import handoff
from rlmflow.graph.nodes import AgentStart, ExecAction
from rlmflow.runtime.env import RLMFLOW_REPLAY


def _replay_root(flow: Any, agent: AgentStart) -> AgentStart:
    """Return the highest missing worker owner required by ``agent``."""
    root = agent
    while root.config.reuse_worker and root.parent is not None:
        parent = root.parent.parent_agent
        if parent is None or flow.runtime.get(parent) is not None:
            break
        root = parent
    return root


def _required_agents(root: AgentStart) -> tuple[list[AgentStart], set[int]]:
    """Agents in worker sessions that contain at least one unfinished tenant."""
    agents = [node for node in root.walk() if isinstance(node, AgentStart)]

    def owner(agent: AgentStart) -> AgentStart:
        while agent.config.reuse_worker and agent.parent is not None:
            parent = agent.parent.parent_agent
            if parent is None:
                break
            agent = parent
        return agent

    required_owners = {id(owner(agent)) for agent in agents if not agent.terminal}
    required = {id(agent) for agent in agents if id(owner(agent)) in required_owners}
    return agents, required


async def replay(flow: Any, root: AgentStart) -> None:
    """Rebuild the namespaces of a graph this flow did not run.

    Appends nothing: ``launch_subagent`` resolves children already attached
    to each action and reads finished results from their terminal nodes. A block
    that failed the first time fails the same way here, leaving the same partial
    bindings, which is why output and errors are discarded.

    Terminal agents are normally skipped, but a terminal worker owner is replayed
    when an unfinished ``reuse_worker`` descendant still needs its worker session.
    ``execution_guard`` is a live admission policy, so replay intentionally calls
    the underlying runtime after preparing the same namespace and environment.
    """
    root = _replay_root(flow, root)
    agents, required = _required_agents(root)
    for agent in sorted(agents, key=lambda item: item.config.depth):
        if id(agent) in required:
            flow.runtime.repl_for(agent)

    actions = [
        node
        for node in root.walk()
        if (
            isinstance(node, ExecAction)
            and node is not node.parent_agent.frontier
            and id(node.parent_agent) in required
        )
    ]
    actions.sort(
        key=lambda node: (
            node.repl_execution_order is None,
            node.repl_execution_order or 0,
            node.created_at,
        )
    )
    for node in actions:
        repl = flow.wrapped_runtime.prepare(node)
        repl.update_env({RLMFLOW_REPLAY: "1"})
        try:
            await flow.runtime.execute(node, node.code)
        finally:
            repl.update_env({RLMFLOW_REPLAY: "0"})


async def ensure_replayed(flow: Any, agent: AgentStart) -> None:
    """Restore an unfinished namespace immediately before its first execution."""
    if id(agent) in flow._restored_agents:
        return
    async with flow._restore_lock:
        if id(agent) in flow._restored_agents:
            return
        if flow.runtime.get(agent) is not None:
            flow._restored_agents.add(id(agent))
            return

        root = _replay_root(flow, agent)
        await replay(flow, root)
        flow._restored_agents.update(
            id(node) for node in root.walk() if isinstance(node, AgentStart)
        )


async def resume(flow: Any, roots: list[AgentStart]) -> None:
    """Make graphs ready to drive: live namespaces, fresh scheduling, mailboxes."""
    for root in roots:
        if flow.runtime.get(root) is None:  # a graph this Flow has not run
            if flow.restore == "replay":
                await replay(flow, root)
        else:
            flow._restored_agents.add(id(root))
            if flow.restore == "replay":
                for node in root.walk():
                    if (
                        isinstance(node, AgentStart)
                        and not node.terminal
                        and flow.runtime.get(node) is None
                    ):
                        await replay(flow, node)
        for node in root.walk():
            if isinstance(node, AgentStart):
                node.dependency_scheduled = False
        handoff.rebuild_mailboxes(root)


__all__ = ["ensure_replayed", "replay", "resume"]
