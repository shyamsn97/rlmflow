"""The history writes. Each function checks its own preconditions against the
committed graph, then logs its nodes and edges in one step."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from functools import wraps
from typing import Any, TypeVar

from rlmflow.graph import reads
from rlmflow.graph.config import AgentConfig, input_parts, validate_agent_name
from rlmflow.graph.graph import DEPENDS_ON, FROM, NEXT, Edge, Graph, InvariantError, StaleFrontier
from rlmflow.graph.nodes import (
    ActionNode,
    AgentStart,
    AppendChild,
    DoneOutput,
    ErrorOutput,
    ExecOutput,
    Node,
    UserQuery,
    build_agent,
)

T = TypeVar("T")

#: Frontiers where a queued message may enter history without cutting a turn.
_DELIVERABLE = (AgentStart, UserQuery, ExecOutput, ErrorOutput, DoneOutput)


def _require(g: Graph, *nodes: Node | None) -> None:
    for node in nodes:
        if node is not None and node.id not in g._nodes:
            raise InvariantError(f"{node.type} {node.id!r} is not in the graph")


def _require_agent(g: Graph, agent: Node) -> None:
    _require(g, agent)
    if not isinstance(agent, AgentStart):
        raise InvariantError(f"expected an AgentStart, not {agent.type} {agent.id!r}")


def locked(write: Callable[..., T]) -> Callable[..., T]:
    """Check and write under the graph's lock, so no other write lands between."""

    @wraps(write)
    def write_under_lock(g: Graph, *args: Any, **kwargs: Any) -> T:
        with g._lock:
            return write(g, *args, **kwargs)

    return write_under_lock


@locked
def append(g: Graph, agent: AgentStart, node: Node, *, after: Node | None = None) -> Node:
    """Land ``node`` as the agent's next node.

    ``after`` is a compare-and-set guard: the write raises ``StaleFrontier``
    unless ``after`` is still the agent's frontier.
    """
    _require_agent(g, agent)
    if isinstance(node, AgentStart):
        raise InvariantError("an AgentStart is added by spawn or fork, not append")
    if node.id in g._nodes:
        raise InvariantError(f"duplicate node id {node.id!r}")
    source = g._last(agent.id)
    if after is not None and after is not source:
        if not (source is agent and reads.prev(g, agent) is after):
            raise StaleFrontier(f"{after.type} {after.id!r} is not the frontier of {agent.id!r}")
        source = agent  # an empty fork extends from its own AgentStart
    node.agent_id = agent.id
    g._append_unchecked(node, Edge(kind=NEXT, source=source.id, target=node.id))
    return node


@locked
def spawn(
    g: Graph,
    task: str = "",
    *,
    by: Node | None = None,
    name: str | None = None,
    config: AgentConfig | None = None,
    depends_on: Iterable[AgentStart] = (),
    inputs: Mapping[str, Any] | None = None,
    **args: Any,
) -> AgentStart:
    """Add an agent; ``by`` is the node that launched it, ``None`` for top-level.

    ``inputs`` binds producer results to names. A value may be an ``AgentStart``
    or ``Input(agent, description=...)``.
    """
    _require(g, by)
    configured = config or AgentConfig()
    if name is not None:
        validate_agent_name(name)
        configured = replace(configured, name=name)
    start = build_agent(task, args, config=configured)
    items: list[Node | Edge] = [start]
    if by is not None:
        items.append(Edge(kind=FROM, source=start.id, target=by.id))
    for producer in depends_on:
        _require_agent(g, producer)
        items.append(Edge(kind=DEPENDS_ON, source=start.id, target=producer.id))
    for input_name, value in (inputs or {}).items():
        wrapped = input_parts(value)
        producer, description = wrapped if wrapped is not None else (value, None)
        _require_agent(g, producer)
        data = {"as": input_name, **({"description": description} if description else {})}
        items.append(Edge(kind=DEPENDS_ON, source=start.id, target=producer.id, data=data))
    g._append_unchecked(*items)
    return start


@locked
def send(
    g: Graph,
    agent: AgentStart,
    content: str | UserQuery,
    *,
    sender: Node | Sequence[Node] | None = None,
    **metadata: Any,
) -> UserQuery:
    """Queue a durable message in the agent's inbox without touching its history."""
    _require_agent(g, agent)
    query = content if isinstance(content, UserQuery) else UserQuery(content=content)
    query.agent_id = agent.id
    query.metadata.update(metadata)
    senders = [] if sender is None else [sender] if isinstance(sender, Node) else list(sender)
    _require(g, *senders)
    if query.id in g._nodes:
        raise InvariantError(f"duplicate node id {query.id!r}")
    g._append_unchecked(
        query, *(Edge(kind=FROM, source=query.id, target=item.id) for item in senders)
    )
    return query


@locked
def deliver(g: Graph, agent: AgentStart) -> list[UserQuery]:
    """Move queued messages into history, in ``seq`` order, at a turn boundary.

    A fork that owns nothing yet takes messages at any base node, so a fork made
    at an ``LLMOutput`` hears its message before it would rerun that code.
    """
    queued = reads.inbox(g, agent)
    source = g._last(agent.id)
    if not queued or not (source is agent or isinstance(reads.frontier(g, agent), _DELIVERABLE)):
        return []
    edges = []
    for query in queued:
        edges.append(Edge(kind=NEXT, source=source.id, target=query.id))
        source = query
    g._append_unchecked(*edges)
    return queued


@locked
def fork(
    g: Graph,
    at: Node,
    *,
    name: str | None = None,
    by: Node | None = None,
    **config: Any,
) -> AgentStart:
    """Branch a new agent whose history is everything up to ``at``.

    Nothing is copied: the fork shares the prefix through one ``next`` edge.
    ``at`` may be any landed node except an action; fork before an action to
    run a fresh one.
    """
    if isinstance(at, ActionNode):
        raise ValueError(f"cannot fork at {at.type} {at.id!r}; fork at the node before it")
    _require(g, at, by)
    if not isinstance(at, AgentStart) and not g.edges(kind=NEXT, target=at.id):
        raise InvariantError(f"cannot fork at undelivered {at.type} {at.id!r}")
    owner = g.node(at.agent_id)
    if name is None:
        name = f"{owner.config.name}_fork{sum(1 for a in g.nodes(AgentStart) if reads.prev(g, a))}"
    validate_agent_name(name)
    start = AgentStart(
        content=owner.content,
        args=deepcopy(owner.args),
        config=replace(deepcopy(owner.config), name=name, **config),
        system_prompts=dict(owner.system_prompts),
        _messages=(),
    )
    items: list[Node | Edge] = [start, Edge(kind=NEXT, source=at.id, target=start.id)]
    if by is not None:
        items.append(Edge(kind=FROM, source=start.id, target=by.id))
    g._append_unchecked(*items)
    return start


@locked
def attach(
    g: Graph,
    parent: AgentStart,
    *agents: AgentStart,
    names: Sequence[str | None] | None = None,
) -> AppendChild:
    """Give ``parent`` one launch action whose receipts resolve to ``agents``.

    Attached agents may be finished or still running. Each must not already
    have a launcher.
    """
    if not agents:
        raise ValueError("attach needs at least one agent")
    requested = list(names) if names is not None else [None] * len(agents)
    if len(requested) != len(agents):
        raise ValueError("names must match agents")
    _require_agent(g, parent)
    ancestors = {parent.id}
    launcher = reads.launcher(g, parent)
    while launcher is not None:
        ancestors.add(launcher.id)
        launcher = reads.launcher(g, launcher)
    used = {child.config.name for child in reads.children(g, parent)}
    final: list[str] = []
    for agent, name in zip(agents, requested, strict=True):
        _require_agent(g, agent)
        if agent.id in ancestors:
            raise InvariantError(f"agent {agent.id!r} cannot launch itself")
        if g.edges(kind=FROM, source=agent.id):
            raise ValueError(f"agent {agent.id!r} already has a launcher")
        chosen = name or (agent.config.name if agent.config.name != "root" else f"child{len(used)}")
        validate_agent_name(chosen)
        if chosen in used:
            raise ValueError(f"duplicate child name {chosen!r}")
        used.add(chosen)
        final.append(chosen)
    action = AppendChild.launching(agents)
    action.agent_id = parent.id
    g._append_unchecked(
        action,
        Edge(kind=NEXT, source=g._last(parent.id).id, target=action.id),
        *(Edge(kind=FROM, source=agent.id, target=action.id) for agent in agents),
    )
    for agent, name in zip(agents, final, strict=True):
        if agent.config.name != name:
            agent.config = replace(agent.config, name=name)
    return action


__all__ = ["append", "attach", "deliver", "fork", "send", "spawn"]
