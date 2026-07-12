"""Unit tests for the graph-free per-agent TaskQueue primitive."""

import asyncio

from rflow.tasks import TaskQueue


async def collect(queue):
    return [item async for item in queue.stream()]


def test_taskqueue_streams_items_by_agent_id():
    async def main():
        tq = TaskQueue()
        tq.emit("a", "x")
        tq.emit("b", "y")
        assert await collect(tq) == [("a", "x"), ("b", "y")]

    asyncio.run(main())


def test_taskqueue_adds_followup_after_running_task_finishes():
    log = []

    async def work(tq, value):
        log.append(value)
        tq.emit("a", value)

    async def main():
        tq = TaskQueue()
        tq.add("a", lambda: work(tq, "first"))
        tq.add("a", lambda: work(tq, "second"))
        assert await collect(tq) == [("a", "first"), ("a", "second")]
        assert log == ["first", "second"]

    asyncio.run(main())


def test_taskqueue_stop_drops_queued_followup():
    async def first(tq):
        tq.emit("a", "first")

    async def main():
        tq = TaskQueue()
        tq.add("a", lambda: first(tq))
        tq.add("a", lambda: first(tq))
        tq.stop("a")
        assert await collect(tq) == [("a", "first")]

    asyncio.run(main())


def test_taskqueue_surfaces_task_exception():
    async def boom():
        raise ValueError("boom")

    async def main():
        tq = TaskQueue()
        tq.add("a", boom)
        await collect(tq)
        exc = tq.exception()
        assert isinstance(exc, ValueError)

    asyncio.run(main())


def test_taskqueue_aclose_cancels_running_tasks():
    async def work():
        await asyncio.sleep(10)

    async def main():
        tq = TaskQueue()
        tq.add("a", work)
        await tq.aclose()
        assert tq.tasks["a"].done()

    asyncio.run(main())
