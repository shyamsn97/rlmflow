"""Run and stream several minimal Flows together.

A ``FlowGroup`` is a thin handle over N independent flows. Iterate it to fan-in
their graph events as they arrive (each event carries ``graph_id`` so the stream
is self-describing), or ``await group.run()`` to drive them all to completion and
get back the resulting graphs keyed by label.

Concurrency is governed by a :class:`~rflow.pool.Pool` (default: an
unbounded :class:`AsyncPool`). Pass ``pool=AsyncPool(4)`` to cap how many flows
run at once, or ``pool=SequentialPool()`` to run them one after another — a flow
holds its slot for as long as it is streaming.

    group = group_flows(
        cols=(cols_flow, cols_graph),
        root=(root_flow, root_graph),
    )

    async for event in group:
        print(event.graph_id, event.node_type)

    graphs = await group.run()
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from rflow.flow import Flow
from rflow.graph import Graph
from rflow.graph.events import Event
from rflow.pool import AsyncPool, Pool

FlowEntry = tuple[Flow, Graph | str]


class FlowGroup:
    """Concurrently run a set of labelled ``(flow, graph)`` entries.

    Each entry must use a distinct ``Flow`` instance; a flow can only drive one
    run at a time.
    """

    def __init__(
        self, entries: dict[str, FlowEntry], *, pool: Pool | None = None
    ) -> None:
        self._pool = pool or AsyncPool()
        self._flows: dict[str, Flow] = {}
        self._graphs: dict[str, Graph] = {}
        for label, (flow, graph_or_query) in entries.items():
            graph = (
                graph_or_query
                if isinstance(graph_or_query, Graph)
                else Graph(query=graph_or_query)
            )
            self._flows[label] = flow
            self._graphs[label] = graph

    @property
    def graphs(self) -> dict[str, Graph]:
        """The graph for each label (mutated in place as the flows run)."""
        return dict(self._graphs)

    async def _run_under_pool(self, flow: Flow, graph: Graph) -> AsyncIterator[Event]:
        # Hold a pool slot for the flow's whole lifetime. When the pool is
        # bounded, a flow that hasn't acquired a slot never starts (its first
        # ``anext`` blocks here), so nothing runs beyond the concurrency limit.
        async with self._pool.limit():
            async for event in flow.run_streaming(graph):
                yield event

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        streams = {
            label: aiter(self._run_under_pool(self._flows[label], self._graphs[label]))
            for label in self._graphs
        }
        # One in-flight "get the next event" task per live stream; whichever
        # resolves first is yielded, then re-armed. No sentinels, no queue.
        pending = {asyncio.ensure_future(anext(it)): it for it in streams.values()}
        try:
            while pending:
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    stream = pending.pop(task)
                    try:
                        yield task.result()
                    except StopAsyncIteration:
                        continue
                    pending[asyncio.ensure_future(anext(stream))] = stream
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for stream in streams.values():
                await stream.aclose()

    async def run(self) -> dict[str, Graph]:
        """Drive every flow to completion and return the graphs keyed by label."""
        async for _event in self:
            pass
        return self.graphs


def group_flows(*, pool: Pool | None = None, **entries: FlowEntry) -> FlowGroup:
    """Group ``label=(flow, graph_or_query)`` entries into a :class:`FlowGroup`.

    ``pool`` bounds how many flows run concurrently (default: unbounded).
    """
    return FlowGroup(entries, pool=pool)
