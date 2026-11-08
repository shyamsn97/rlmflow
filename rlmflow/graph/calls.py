"""What a node's tool calls returned, saved on the node so running it again repeats nothing.

``node.calls`` is a list in call order. Each entry is keyed by the call
itself, a hash of the tool name and the canonical JSON of its arguments, and
holds what the call returned::

    {"tool": "launch_agent", "key": "a1f3...", "agent": "agent_9f..."}
    {"tool": "await", "key": "c07e...", "agent": "agent_9f...", "done": "node_41..."}
    {"tool": "llm_query", "key": "5b2d...", "value": "Paris"}

Each run of a node gets a ``CallLog``. When the node runs again, a call claims
the first unclaimed entry with its tool and key, so repeated identical calls get
their results in order. A call with no matching entry runs and is saved. After
a rerun, ``unclaimed`` returns the entries nothing claimed: calls the new code
no longer makes, whose effects the caller undoes.

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


def key(tool: str, *parts: Any) -> str:
    """The key of a call to ``tool`` with ``parts``; ``TypeError`` unless they're JSON."""
    try:
        text = json.dumps([tool, *parts], sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{tool} arguments must be JSON to save the call") from exc
    return hashlib.sha256(text.encode()).hexdigest()


class CallLog:
    """One run of a node: its saved calls, and which of them this run has used.

    Tools may call from worker threads, so claims and records take a lock.
    """

    def __init__(self, node: Node) -> None:
        self.node = node
        self._claimed: set[int] = set()
        self._lock = threading.Lock()

    def claim(self, tool: str, call_key: str) -> dict[str, Any] | None:
        """The first unclaimed entry saved for this call, now claimed; ``None`` on a miss."""
        with self._lock:
            for entry in self.node.calls:
                if (
                    id(entry) not in self._claimed
                    and entry["tool"] == tool
                    and entry.get("key") == call_key
                ):
                    self._claimed.add(id(entry))
                    return entry
        return None

    def record(self, tool: str, call_key: str, **result: Any) -> None:
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
        with self._lock:
            self.node.calls.append(entry)
            self._claimed.add(id(entry))

    def unclaimed(self) -> list[dict[str, Any]]:
        """Drop the entries no call of this run claimed from the node, and return them."""
        with self._lock:
            dropped = [entry for entry in self.node.calls if id(entry) not in self._claimed]
            if dropped:
                self.node.calls = [e for e in self.node.calls if id(e) in self._claimed]
        return dropped


def record(node: Node, tool: str, call_key: str, **result: Any) -> None:
    """Save a call on ``node`` outside a run, as a controller preparing a graph would."""
    CallLog(node).record(tool, call_key, **result)


def launched(node: Node) -> list[str]:
    """Ids of the agents this node launched, in call order."""
    return [entry["agent"] for entry in node.calls if entry["tool"] == LAUNCH and "agent" in entry]


def rebind(node: Node, old: str, new: str) -> None:
    """Point every entry that names agent ``old`` at ``new``, forgetting what an await got."""
    for entry in node.calls:
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
    "CallLog",
    "key",
    "launched",
    "rebind",
    "record",
]
