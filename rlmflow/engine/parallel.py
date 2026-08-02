"""Drive several Node trajectories through one Flow queue at once."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from rlmflow.graph.nodes import AgentStart, Node
from rlmflow.utils import node_from_input

if TYPE_CHECKING:
    from rlmflow.flow import Flow, StepUntil


async def parallel_stream(
    flow: Flow,
    *roots: AgentStart | str,
    until: StepUntil = "done",
    close_repls: bool = False,
) -> AsyncIterator[Node]:
    """Drive several roots through one queue, yielding nodes as they are created."""
    agents = [node_from_input(root) for root in roots]
    if not agents:
        return
    stream = flow.run_streaming(
        agents[0],
        *agents[1:],
        until=until,
        close_repls=close_repls,
    )
    try:
        async for node in stream:
            yield node
    finally:
        await stream.aclose()


async def parallel_run(
    flow: Flow,
    *roots: AgentStart | str,
    until: StepUntil = "done",
    close_repls: bool = False,
) -> list[AgentStart]:
    """Drive several graphs to completion, returning them in argument order.

    String queries are coerced to graphs first so the caller gets back the live
    graphs (and their ``result()``) without holding them beforehand.
    """
    coerced = [node_from_input(root) for root in roots]
    async for _event in parallel_stream(
        flow, *coerced, until=until, close_repls=close_repls
    ):
        pass
    return coerced


__all__ = ["parallel_run", "parallel_stream"]
