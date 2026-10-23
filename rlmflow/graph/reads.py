"""Pure agent queries over a :class:`~rlmflow.graph.graph.Graph`."""

from __future__ import annotations

from typing import Any

from rlmflow.graph.graph import DEPENDS_ON, FROM, NEXT, Graph
from rlmflow.graph.nodes import (
    ASK_KEY,
    AgentStart,
    DoneOutput,
    LLMOutput,
    LLMUsage,
    Node,
    TransitionDecision,
    UserQuery,
)


def prev(g: Graph, node: Node) -> Node | None:
    """The node before ``node`` in history, or ``None`` at a run's first node."""
    edges = g.edges(kind=NEXT, target=node.id)
    return g.node(edges[0].source) if edges else None


def history(g: Graph, at: Node) -> list[Node]:
    """Nodes leading to ``at``; for an agent, its whole history to the frontier.

    A fork's walk continues through its base agent's prefix.
    """
    node: Node | None = g._last(at.id) if isinstance(at, AgentStart) else at
    nodes: list[Node] = []
    while node is not None:
        nodes.append(node)
        node = prev(g, node)
    nodes.reverse()
    return nodes


def project(g: Graph, at: Node, keep: int | None = None) -> list[dict[str, str]]:
    """The chat history leading to ``at``, as provider messages."""
    return project_nodes(history(g, at), keep)


def project_nodes(nodes: list[Node], keep: int | None = None) -> list[dict[str, str]]:
    """Provider messages for a history list.

    ``keep`` counts flattened messages. Invisible nodes consume no capacity,
    and a multi-message node is trimmed from its oldest messages when only part
    of it fits.
    """
    if keep is not None and keep <= 0:
        return []
    groups: list[list[dict[str, str]]] = []
    remaining = keep
    for node in reversed(nodes):
        messages = node._history_message_dicts()
        if remaining is not None:
            messages = messages[-remaining:]
        if not messages:
            continue
        groups.append(messages)
        if remaining is not None:
            remaining -= len(messages)
            if remaining == 0:
                break
    return [message for messages in reversed(groups) for message in messages]


def frontier(g: Graph, agent: AgentStart) -> Node:
    """The agent's latest node. An empty fork stands at its base node."""
    last = g._last(agent.id)
    if last is agent:
        return prev(g, agent) or agent
    return last


def terminal(g: Graph, agent: AgentStart) -> bool:
    """Finished, with no queued message that would reopen it."""
    return isinstance(frontier(g, agent), DoneOutput) and not g._inbox.get(agent.id)


def result(g: Graph, agent: AgentStart) -> Any:
    """The agent's submitted value, or ``None`` until it is terminal."""
    node = frontier(g, agent)
    return node.result if isinstance(node, DoneOutput) else None


def inbox(g: Graph, agent: AgentStart) -> list[UserQuery]:
    """Messages sent to ``agent`` that are not yet part of its history."""
    return [g.node(node_id) for node_id in g._inbox.get(agent.id, ())]


def producers(g: Graph, agent: AgentStart) -> list[AgentStart]:
    """Agents this agent must wait for, through active ``depends_on`` edges."""
    return [g.node(edge.target) for edge in g.edges(kind=DEPENDS_ON, source=agent.id)]


def inputs(g: Graph, agent: AgentStart) -> dict[str, Any]:
    """Producer results bound to names by ``depends_on`` edges with ``as``."""
    return {
        edge.data["as"]: result(g, g.node(edge.target))
        for edge in g.edges(kind=DEPENDS_ON, source=agent.id)
        if edge.data.get("as")
    }


def ready(g: Graph) -> list[AgentStart]:
    """Agents with work to do whose producers have all finished."""
    return [
        agent
        for agent in g.nodes(AgentStart)
        if not terminal(g, agent) and all(terminal(g, producer) for producer in producers(g, agent))
    ]


def launcher(g: Graph, agent: AgentStart) -> AgentStart | None:
    """The agent that owns this agent's ``from`` target; ``None`` when top-level."""
    edges = g.edges(kind=FROM, source=agent.id)
    if not edges:
        return None
    return g.node(g.node(edges[0].target).agent_id)


def children(g: Graph, agent: AgentStart) -> list[AgentStart]:
    """Agents launched by ``agent``, in ``seq`` order."""
    launched = [
        g.node(edge.source)
        for node in g.nodes(agent=agent.id)
        for edge in g.edges(kind=FROM, target=node.id)
        if isinstance(g.node(edge.source), AgentStart)
    ]
    return sorted(launched, key=lambda child: child.seq)


def lineage(g: Graph, agent: AgentStart) -> list[AgentStart]:
    """The launcher chain from the top-level agent down to ``agent``."""
    chain: list[AgentStart] = []
    current: AgentStart | None = agent
    while current is not None:
        chain.append(current)
        current = launcher(g, current)
    chain.reverse()
    return chain


def path(g: Graph, agent: AgentStart) -> str:
    return ".".join(item.config.name for item in lineage(g, agent))


def depth(g: Graph, agent: AgentStart) -> int:
    return len(lineage(g, agent)) - 1


def tree(g: Graph, agent: AgentStart) -> list[AgentStart]:
    """``agent`` and every agent it launched, transitively, depth first."""
    found: list[AgentStart] = []
    stack = [agent]
    while stack:
        current = stack.pop()
        found.append(current)
        stack.extend(reversed(children(g, current)))
    return found


def usage(g: Graph, agents: list[AgentStart] | None = None) -> LLMUsage:
    """Model usage spent by ``agents``, or anywhere in the graph."""
    if agents is None:
        nodes = g.nodes()
    else:
        nodes = [node for agent in agents for node in g.nodes(agent=agent.id)]
    spent = (node.usage for node in nodes if isinstance(node, (LLMOutput, TransitionDecision)))
    return sum(spent, LLMUsage())


def ask_terms(g: Graph, at: Node) -> dict[str, Any] | None:
    """The terms of the latest ask in ``at``'s history, if one is open there."""
    for item in reversed(history(g, at)):
        if isinstance(item, UserQuery) and ASK_KEY in item.metadata:
            return item.metadata[ASK_KEY]
        if isinstance(item, DoneOutput):
            return None
    return None


__all__ = [
    "ask_terms",
    "children",
    "depth",
    "frontier",
    "history",
    "inbox",
    "inputs",
    "launcher",
    "lineage",
    "path",
    "prev",
    "producers",
    "project",
    "project_nodes",
    "ready",
    "result",
    "terminal",
    "tree",
    "usage",
]
