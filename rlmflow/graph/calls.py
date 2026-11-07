"""What a cell's tool calls returned, saved on the cell so running it again repeats nothing.

``cell.data["calls"]`` is a list; the n-th entry for a tool is that tool's n-th
call. Each entry names its tool and holds what the call returned::

    {"tool": "launch_agent", "agent": "agent_9f..."}
    {"tool": "await", "agent": "agent_9f...", "done": "node_41..."}
    {"tool": "llm_query", "value": "Paris"}

When the cell runs again, the n-th call to a tool gets the n-th entry for that
tool instead of running. An entry with nothing saved (the call failed, was cut
off, or returned something too large or not JSON) runs the call again.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from rlmflow.graph.nodes import Node

#: The built-in calls whose entries name an agent.
LAUNCH = "launch_agent"
AWAIT = "await"
SEND = "agent.send"

#: Results larger than this, as JSON, aren't saved; the call runs again instead.
MAX_SAVED_BYTES = 64 * 1024

_lock = threading.Lock()


def entries(cell: Node) -> list[dict[str, Any]]:
    """The cell's saved calls, in call order."""
    return cell.data.setdefault("calls", [])


def saved(cell: Node, tool: str, index: int) -> dict[str, Any] | None:
    """The entry for the ``index``-th call to ``tool``, if one was saved."""
    matching = [entry for entry in cell.data.get("calls") or () if entry["tool"] == tool]
    if index < len(matching) and len(matching[index]) > 1:
        return matching[index]
    return None


def record(cell: Node, tool: str, index: int, **result: Any) -> None:
    """Save what the ``index``-th call to ``tool`` returned.

    ``result`` is ``value=...``, or ``agent=...`` (plus ``done=...`` for an await).
    A value too large or not JSON leaves the call unsaved.
    """
    if "value" in result:
        try:
            text = json.dumps(result["value"], allow_nan=False)
        except (TypeError, ValueError):
            return
        if len(text) > MAX_SAVED_BYTES:
            return
        result["value"] = json.loads(text)
    with _lock:
        slots = entries(cell)
        matching = [entry for entry in slots if entry["tool"] == tool]
        while len(matching) <= index:
            slot = {"tool": tool}
            slots.append(slot)
            matching.append(slot)
        entry = matching[index]
        entry.clear()
        entry.update(tool=tool, **result)


def launched(cell: Node) -> list[str]:
    """Ids of the agents this cell launched, in call order."""
    return [
        entry["agent"]
        for entry in cell.data.get("calls") or ()
        if entry["tool"] == LAUNCH and "agent" in entry
    ]


def rebind(cell: Node, old: str, new: str) -> None:
    """Point every entry that names agent ``old`` at ``new``, forgetting what an await got."""
    for entry in cell.data.get("calls") or ():
        if entry.get("agent") == old:
            entry["agent"] = new
            entry.pop("done", None)


__all__ = [
    "AWAIT",
    "LAUNCH",
    "MAX_SAVED_BYTES",
    "SEND",
    "entries",
    "launched",
    "rebind",
    "record",
    "saved",
]
