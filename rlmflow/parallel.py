"""Drive several Node trajectories on one Flow at once.

Free functions, not ``Flow`` methods: each graph is just a run the flow already
knows how to stream (``run_streaming``); this only fans those runs out and merges
their Nodes. Keeping it off ``Flow`` keeps the class focused on single-graph
policy and makes the fan-out easy to read in one place.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from rlmflow.graph import AgentStart, Node
from rlmflow.utils import node_from_input

if TYPE_CHECKING:
    from rlmflow.flow import Flow, StepUntil


async def parallel_stream(
    flow: Flow,
    *roots: AgentStart | str,
    until: StepUntil = "done",
    close_repls: bool = False,
) -> AsyncIterator[Node]:
    """Drive several trajectories, merging their published Nodes.

    Each root gets its own stream boundary while sharing the Flow's Runtime.
    Concurrent streams must have distinct trajectory identities.
    """
    streams = [
        aiter(
            flow.run_streaming(
                node_from_input(root),
                until=until,
                close_repls=close_repls,
            )
        )
        for root in roots
    ]
    # One in-flight "next Node" task per live stream; whichever resolves first
    # is yielded, then re-armed. No sentinels, no queue, no bound.
    pending = {asyncio.ensure_future(anext(it)): it for it in streams}
    try:
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                it = pending.pop(task)
                try:
                    yield task.result()
                except StopAsyncIteration:
                    continue
                pending[asyncio.ensure_future(anext(it))] = it
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for it in streams:
            await it.aclose()


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
