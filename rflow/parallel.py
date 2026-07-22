"""Drive several graphs on one :class:`~rflow.flow.Flow` at once.

Free functions, not ``Flow`` methods: each graph is just a run the flow already
knows how to stream (``run_streaming``); this only fans those runs out and merges
their events. Keeping it off ``Flow`` keeps the class focused on single-graph
policy and makes the fan-out easy to read in one place.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from rflow.graph import Graph
from rflow.graph.events import Event, StepUntil
from rflow.utils import graph_from_input

if TYPE_CHECKING:
    from rflow.flow import Flow


async def parallel_stream(
    flow: Flow,
    *graphs: Graph | str,
    until: StepUntil = "done",
    n: int | None = None,
    close_repls: bool = False,
) -> AsyncIterator[Event]:
    """Drive several graphs on one flow, merging their events into one stream.

    Each graph gets its own run (keyed by ``graph_id``) with its own scheduler
    and ``until`` boundary; every event carries ``graph_id`` so the merged stream
    is self-describing. Graphs must be distinct — two entries for the same
    ``graph_id`` would drive one run twice (``run_streaming`` rejects the second).
    """
    streams = [
        aiter(
            flow.run_streaming(
                graph=graph_from_input(g),
                until=until,
                n=n,
                close_repls=close_repls,
            )
        )
        for g in graphs
    ]
    # One in-flight "next event" task per live stream; whichever resolves first
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
    *graphs: Graph | str,
    until: StepUntil = "done",
    n: int | None = None,
    close_repls: bool = False,
) -> list[Graph]:
    """Drive several graphs to completion, returning them in argument order.

    String queries are coerced to graphs first so the caller gets back the live
    graphs (and their ``result()``) without holding them beforehand.
    """
    coerced = [graph_from_input(g) for g in graphs]
    async for _event in parallel_stream(
        flow, *coerced, until=until, n=n, close_repls=close_repls
    ):
        pass
    return coerced


__all__ = ["parallel_run", "parallel_stream"]
