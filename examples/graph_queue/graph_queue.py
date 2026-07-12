"""Minimal graph queue: one stream per agent.

Each agent owns one queue. When the stream yields an item for that agent, the
agent may add another task to its queue, or may add nothing.

Run:  python examples/graph_queue/graph_queue.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


class TaskQueue:
    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Queue] = {}
        self.notify = asyncio.Event()

    def queue(self, agent_id: str) -> asyncio.Queue:
        if agent_id not in self.queues:
            self.queues[agent_id] = asyncio.Queue()
        return self.queues[agent_id]

    async def put(self, agent_id: str, item: str) -> None:
        await self.queue(agent_id).put(item)
        self.notify.set()

    async def stream(self):
        """Pull items from every per-id queue as they arrive."""
        while True:
            drained = False
            for agent_id, queue in list(self.queues.items()):
                while not queue.empty():
                    yield agent_id, queue.get_nowait()
                    drained = True
            if self.queues and all(queue.empty() for queue in self.queues.values()):
                return
            if drained:
                continue
            self.notify.clear()
            if any(not queue.empty() for queue in self.queues.values()):
                continue
            await self.notify.wait()


@dataclass
class Agent:
    id: str
    queue: TaskQueue
    should_add_tasks: bool
    remaining: int = 3

    async def add_task(self, item: str) -> None:
        print(f"  [{self.id}] add {item}")
        await self.queue.put(self.id, item)

    async def maybe_add_task(self, item: str) -> None:
        if not self.should_add_tasks:
            print(f"  [{self.id}] add nothing")
            return
        if self.remaining <= 0:
            print(f"  [{self.id}] add nothing")
            return
        self.remaining -= 1
        await self.add_task(f"task{self.remaining}")


async def main() -> None:
    tq = TaskQueue()
    agents = {
        "a": Agent("a", tq, should_add_tasks=True),
        "b": Agent("b", tq, should_add_tasks=False),
    }

    await agents["a"].add_task("start")
    await agents["b"].add_task("start")

    print("-- stream --")
    async for agent_id, item in tq.stream():
        print(f"got ({agent_id}, {item})")
        await agents[agent_id].maybe_add_task(item)
    print("ok")


if __name__ == "__main__":
    asyncio.run(main())
