"""What a cell's tool calls returned, saved on the cell so running it again repeats nothing.

``cell.data["calls"]`` is a list in call order. Each entry is keyed by the call
itself, a hash of the tool name and the canonical JSON of its arguments, and
holds what the call returned::

    {"tool": "launch_agent", "key": "a1f3...", "agent": "agent_9f..."}
    {"tool": "await", "key": "c07e...", "agent": "agent_9f...", "done": "node_41..."}
    {"tool": "llm_query", "key": "5b2d...", "value": "Paris"}

When the cell runs again, a call claims the first unclaimed entry with its tool
and key, so repeated identical calls get their results in order. A call with no
matching entry runs and is saved. After a rerun, ``finish`` returns the entries
nothing claimed: calls the new code no longer makes, whose effects the caller
undoes.

Built-in tools are always saved; a host tool only with ``@tool(cache=True)``.
"""

from __future__ import annotations

import hashlib
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
#: For each cell running now, the ids of the entries its calls claimed.
_claimed: dict[str, set[int]] = {}


def key(tool: str, *parts: Any) -> str:
    """The key of a call to ``tool`` with ``parts``; ``TypeError`` unless they're JSON."""
    try:
        text = json.dumps([tool, *parts], sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{tool} arguments must be JSON to save the call") from exc
    return hashlib.sha256(text.encode()).hexdigest()


def entries(cell: Node) -> list[dict[str, Any]]:
    """The cell's saved calls, in call order."""
    return cell.data.setdefault("calls", [])


def begin(cell: Node) -> None:
    """Start a run of ``cell``: no entry is claimed yet."""
    with _lock:
        _claimed[cell.id] = set()


def claim(cell: Node, tool: str, call_key: str) -> dict[str, Any] | None:
    """The first unclaimed entry saved for this call, now claimed; ``None`` on a miss."""
    with _lock:
        claimed = _claimed.setdefault(cell.id, set())
        for entry in cell.data.get("calls") or ():
            if id(entry) not in claimed and entry["tool"] == tool and entry.get("key") == call_key:
                claimed.add(id(entry))
                return entry
    return None


def record(cell: Node, tool: str, call_key: str, **result: Any) -> None:
    """Save what a call returned: ``value=...``, or ``agent=...`` (plus ``done=``/``message=``).

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
    entry = {"tool": tool, "key": call_key, **result}
    with _lock:
        entries(cell).append(entry)
        _claimed.setdefault(cell.id, set()).add(id(entry))


def finish(cell: Node) -> list[dict[str, Any]]:
    """End the run of ``cell``: drop and return the entries no call claimed."""
    with _lock:
        claimed = _claimed.pop(cell.id, set())
        saved = cell.data.get("calls") or []
        unclaimed = [entry for entry in saved if id(entry) not in claimed]
        if unclaimed:
            cell.data["calls"] = [entry for entry in saved if id(entry) in claimed]
    return unclaimed


def forget(cell: Node) -> None:
    """End the run of ``cell`` without dropping anything (it was cut off or replayed)."""
    with _lock:
        _claimed.pop(cell.id, None)


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
            if entry["tool"] == AWAIT:
                entry["key"] = key(AWAIT, new)


__all__ = [
    "AWAIT",
    "LAUNCH",
    "MAX_SAVED_BYTES",
    "SEND",
    "begin",
    "claim",
    "entries",
    "finish",
    "forget",
    "key",
    "launched",
    "rebind",
    "record",
]
