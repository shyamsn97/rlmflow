"""Agent-facing messaging tools: send, ask, connect, disconnect, and waits.

Each tool is one ``ops`` call (or ``g.add_edge``/``g.remove_edge``) whose receipt, the id
it wrote, is saved on the running ``ExecAction`` by call index. A replayed call
returns its receipt and writes nothing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rlmflow.graph import ops, reads
from rlmflow.graph.config import NATIVE_BACKEND, ContextPolicy
from rlmflow.graph.graph import NEXT, STRUCTURAL, Edge, Graph
from rlmflow.graph.nodes import (
    ASK_KEY,
    AgentStart,
    DoneOutput,
    Node,
    UserQuery,
    current_step,
)
from rlmflow.runtime.repl_client import current_rpc_call_id, current_rpc_name
from rlmflow.tools.agents_view import resolve_agent
from rlmflow.tools.namespace import tool

#: ``UserQuery.metadata`` key naming the tool call that sent the message.
CALL_KEY = "call"


def _wake(flow: Any) -> None:
    if flow.queue is not None:
        flow.queue.wake()


def _call_key() -> str:
    return f"{current_rpc_name()}:{current_rpc_call_id()}"


def _sent(g: Graph, action: Node, key: str) -> UserQuery | None:
    """The message this call already wrote, if a rerun reaches it again."""
    for edge in g.edges(target=action.id):
        source = g.node(edge.source)
        if isinstance(source, UserQuery) and source.metadata.get(CALL_KEY) == key:
            return source
    return None


def _check_followup(flow: Any, target: AgentStart) -> None:
    if target.config.context_policy == ContextPolicy.EPHEMERAL:
        raise ValueError(f"agent {target.config.name!r} is ephemeral and takes no follow-ups")
    if target.config.backend != NATIVE_BACKEND and not getattr(
        flow.backend_for(target), "resumable", False
    ):
        raise ValueError(f"backend {target.config.backend!r} cannot continue a session")


@contextmanager
def waiting(flow: Any, g: Graph, waiter: AgentStart, target: AgentStart) -> Iterator[None]:
    """Register ``waiter -> target`` for the length of a wait, refusing cycles.

    A wait closes a cycle when ``target`` already waits, directly or through
    ``depends_on`` inputs, open asks, or handle waits, on ``waiter``.
    """
    seen: set[str] = set()
    stack = [target.id]
    while stack:
        current = stack.pop()
        if current == waiter.id:
            raise RuntimeError(
                f"waiting on {target.config.name!r} would deadlock: it waits on this agent"
            )
        if current in seen:
            continue
        seen.add(current)
        stack.extend(flow._waits.get(current, ()))
        stack.extend(producer.id for producer in reads.producers(g, g.node(current)))
    targets = flow._waits.setdefault(waiter.id, [])
    targets.append(target.id)
    try:
        yield
    finally:
        targets.remove(target.id)
        if not targets:
            flow._waits.pop(waiter.id, None)


def _answered(g: Graph, target: AgentStart, query: UserQuery) -> bool:
    """Whether ``target`` finished a turn after taking ``query`` into history."""
    delivered = g.edges(kind=NEXT, target=query.id)
    frontier = reads.frontier(g, target)
    return bool(delivered) and isinstance(frontier, DoneOutput) and frontier.seq > delivered[0].seq


def send_tool(flow: Any, node: Node):
    step = current_step()
    g = step.graph

    @tool(
        "Queue a message for another agent without waiting. It is delivered at "
        "that agent's next turn boundary and reopens a finished agent. Returns "
        "the message id.",
        proxy=True,
        is_async=False,
        restore="result",
    )
    def send_agent(target: Any, message: str) -> str:
        if not isinstance(message, str):
            raise TypeError("send_agent message must be a string")
        agent = resolve_agent(g, target)
        if agent.config.context_policy == ContextPolicy.EPHEMERAL and reads.terminal(g, agent):
            raise ValueError(f"agent {agent.config.name!r} is ephemeral and takes no follow-ups")
        key = _call_key()
        query = _sent(g, node, key) or ops.send(g, agent, message, sender=node, **{CALL_KEY: key})
        _wake(flow)
        return query.id

    return send_agent


def ask_tool(flow: Any, node: Node):
    step = current_step()
    g = step.graph

    @tool(
        "Send a message to another agent and wait for the result of its next "
        "finished turn. `max_iters=` and `schema=` apply only to that turn.",
        proxy=True,
        restore="result",
    )
    async def ask_agent(
        target: Any,
        message: str,
        *,
        max_iters: int | None = None,
        schema: Any = None,
    ) -> Any:
        if not isinstance(message, str):
            raise TypeError("ask_agent message must be a string")
        agent = resolve_agent(g, target)
        if agent.id == step.agent.id:
            raise ValueError("an agent cannot ask itself")
        _check_followup(flow, agent)
        if flow.queue is None:
            raise RuntimeError("ask_agent requires an active stream")
        terms = {
            key: value
            for key, value in {
                "max_iters": max_iters,
                "schema": flow.output_codec.normalize(schema) if schema is not None else None,
            }.items()
            if value is not None
        }
        with waiting(flow, g, step.agent, agent):
            key = _call_key()
            query = _sent(g, node, key) or ops.send(
                g, agent, message, sender=node, **{CALL_KEY: key, ASK_KEY: terms}
            )
            _wake(flow)
            await flow.queue.join(lambda: _answered(g, agent, query))
        return reads.result(g, agent)

    return ask_agent


def connect_tool(flow: Any, node: Node):
    step = current_step()
    g = step.graph

    @tool(
        "Add an application edge `kind` from you to another agent, with JSON "
        "`**data`. Returns the edge id.",
        proxy=True,
        is_async=False,
        restore="result",
    )
    def connect_agent(target: Any, kind: str, **data: Any) -> str:
        if not isinstance(kind, str) or not kind or kind in STRUCTURAL:
            raise ValueError(f"edge kind must be a non-empty application kind, not {kind!r}")
        edge = Edge(kind=kind, source=step.agent.id, target=resolve_agent(g, target).id, data=data)
        return g.add_edge(edge).id

    return connect_agent


def disconnect_tool(flow: Any, node: Node):
    g = current_step().graph

    @tool("Remove an application edge by id.", proxy=True, is_async=False, restore="result")
    def disconnect_agent(edge_id: str) -> None:
        g.remove_edge(edge_id)

    return disconnect_agent


def peer_tools(flow: Any, node: Node) -> dict[str, Any]:
    """Tools that message other agents and edit the edges between them."""
    return {
        "send_agent": send_tool(flow, node),
        "ask_agent": ask_tool(flow, node),
        "connect_agent": connect_tool(flow, node),
        "disconnect_agent": disconnect_tool(flow, node),
    }


__all__ = [
    "peer_tools",
    "waiting",
]
