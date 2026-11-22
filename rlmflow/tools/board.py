"""A shared task board whose state lives on the nodes that changed it.

``Flow(tools=[TaskBoard()])`` gives agents a DeLM-style board: tasks they post,
claim and finish, and findings they publish for everyone. There is no board
agent and no store. Every board call is a ``cache=True`` host tool, so what it
returned is saved on the node that made it, and a write's value is the event
itself, with a write number::

    {"n": 7, "op": "claim", "id": "sim", "by": "peer2", "ok": True}

The board is those events replayed in write order (``TaskBoard.state(g)``). So
it follows the graph: a saved run loads with its board, a rerun that stops
making a call takes its event back, and a fork has the board as it was at the
cut. A read returns ``upto``, the last write it saw, which is what the figure
draws reads from. A run has one board: every ``TaskBoard`` reads the same events.

Claims are atomic within one process: the first claim of an open task wins.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

from rlmflow.graph.nodes import current_step
from rlmflow.tools import tool

#: The board tools that change it; their saved values are events.
WRITES = ("board_post", "board_claim", "board_finish", "board_publish")
#: The board tools that only read it; their saved values carry ``upto``.
READS = ("board_read",)


def events(g: Any) -> list[dict[str, Any]]:
    """Every board write saved on ``g``'s nodes, in write order."""
    return [event for _, event in sorted(_saved(g).items())]


def _saved(g: Any) -> dict[int, dict[str, Any]]:
    found = {}
    for node in g.nodes():
        for entry in node.calls:
            value = entry.get("value")
            if entry["tool"] in WRITES and isinstance(value, dict) and "n" in value:
                found[value["n"]] = value
    return found


def fold(log: list[dict[str, Any]]) -> dict[str, Any]:
    """The board a list of events adds up to: ``tasks``, ``findings`` and ``upto``."""
    tasks: dict[str, dict[str, Any]] = {}
    findings: list[dict[str, Any]] = []
    for event in log:
        if not event.get("ok"):
            continue
        op, task_id = event["op"], event.get("id")
        if op in ("claim", "finish") and task_id not in tasks:
            continue
        if op == "post":
            tasks[task_id] = {
                "task": event["task"],
                "posted_by": event["by"],
                "status": "open",
                "owner": None,
                "result": None,
            }
        elif op == "claim":
            tasks[task_id].update(status="claimed", owner=event["by"])
        elif op == "finish":
            tasks[task_id].update(status="done", result=event["result"])
        elif op == "publish":
            findings.append(
                {"n": event["n"], "by": event["by"], "topic": event["topic"], "text": event["text"]}
            )
    return {"tasks": tasks, "findings": findings, "upto": log[-1]["n"] if log else 0}


class TaskBoard:
    """The board's tools, for ``Flow(tools=[...])``; the board itself is in the graph."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        #: Events returned but not yet saved on their node, by graph id.
        self._pending: dict[str, list[dict[str, Any]]] = {}

    def __iter__(self) -> Iterator[Callable[..., Any]]:
        yield from (
            self.board_post,
            self.board_claim,
            self.board_finish,
            self.board_publish,
            self.board_read,
        )

    @staticmethod
    def state(g: Any) -> dict[str, Any]:
        """``g``'s board: ``tasks``, ``findings``, and ``upto``, its last write."""
        return fold(events(g))

    def _events(self, g: Any) -> list[dict[str, Any]]:
        """Saved events plus the ones still on their way to their node."""
        saved = _saved(g)
        pending = self._pending.setdefault(g.id, [])
        pending[:] = [event for event in pending if event["n"] not in saved]
        for event in pending:
            saved[event["n"]] = event
        return [event for _, event in sorted(saved.items())]

    def _write(self, op: str, allowed: Callable[[dict, str], bool], **fields: Any) -> dict:
        """Append one event; ``allowed(board, who)`` says whether it takes effect."""
        step = _step()
        g, who = step.graph, step.agent.name
        with self._lock:
            log = self._events(g)
            event = {
                "n": (log[-1]["n"] if log else 0) + 1,
                "op": op,
                "by": who,
                "ok": bool(allowed(fold(log), who)),
                **fields,
            }
            self._pending[g.id].append(event)
        return event

    @tool(cache=True)
    def board_post(self, id: str, task: str) -> dict:
        """Post a task to the shared board under a new `id`; `ok` is False if the id is taken."""
        return self._write("post", lambda b, who: id not in b["tasks"], id=id, task=task)

    @tool(cache=True)
    def board_claim(self, id: str) -> dict:
        """Claim an open task for yourself; `ok` is False if someone else claimed it first."""
        return self._write(
            "claim", lambda b, who: b["tasks"].get(id, {}).get("status") == "open", id=id
        )

    @tool(cache=True)
    def board_finish(self, id: str, result: str) -> dict:
        """Mark a task you claimed done, with a short result others can use."""
        return self._write(
            "finish",
            lambda b, who: b["tasks"].get(id, {}).get("owner") == who,
            id=id,
            result=result,
        )

    @tool(cache=True)
    def board_publish(self, topic: str, text: str) -> dict:
        """Publish a finding to the board's shared context, such as an interface you defined.

        Findings are append-only: to correct one, publish again under the same topic.
        """
        return self._write("publish", lambda b, who: True, topic=topic, text=text)

    @tool(cache=True)
    def board_read(self) -> dict:
        """The whole board: `tasks` (status, owner, result) and `findings`, as of write `upto`."""
        g = _step().graph
        with self._lock:
            return fold(self._events(g))


def _step() -> Any:
    step = current_step()
    if step is None:
        raise RuntimeError("board tools run from an agent's code")
    return step


__all__ = ["READS", "WRITES", "TaskBoard", "events", "fold"]
