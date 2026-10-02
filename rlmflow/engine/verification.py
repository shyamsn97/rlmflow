"""Launch verifiers: one fingerprint for host and worker, and the host-side check.

Standard library only: the REPL worker imports this to fingerprint verifiers
defined in agent code, and a replayed launch must hash to the same value on
whichever side registered it.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import queue
import threading
from typing import Any


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def fingerprint(verify: Any) -> str:
    """Stable identity of a verifier callable's code."""
    code = getattr(verify, "__code__", None)
    return _digest(
        {
            "qualname": getattr(verify, "__qualname__", repr(verify)),
            "code": code.co_code.hex() if code is not None else repr(verify),
            "constants": repr(code.co_consts) if code is not None else "",
        }
    )


def schema_hash(schema: object) -> str | None:
    return _digest(schema) if schema is not None else None


def check(flow: Any, agent: Any, candidate: object) -> str | None:
    """Route a finish candidate to the verifier owned by its launching agent."""
    verifier_id = agent.config.verifier_id
    if verifier_id is None:
        return None
    verify = flow._verifiers.get(verifier_id)
    if verify is not None:
        result = verify(candidate)
        if inspect.isawaitable(result):
            result = _run(result)
    else:
        root = agent.root
        launch = root._index.launch_actions_by_agent_id.get(agent.id) if root is not None else None
        owner = launch.parent_agent if launch is not None else None
        repl = flow.runtime.get(owner) if owner is not None else None
        worker_verify = getattr(repl, "verify", None)
        if worker_verify is None:
            raise RuntimeError(f"verifier {verifier_id!r} is unavailable")
        result = worker_verify(verifier_id, candidate)
    if result is not None and not isinstance(result, str):
        raise TypeError("verify must return str or None")
    return result


def _run(awaitable: Any) -> Any:
    """Run an async verifier behind the synchronous ``finish`` API."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            results.put((True, asyncio.run(awaitable)))
        except BaseException as exc:  # noqa: BLE001 - cross-thread verifier result
            results.put((False, exc))

    thread = threading.Thread(target=run, daemon=True, name="rlmflow-verifier")
    thread.start()
    ok, value = results.get()
    thread.join()
    if not ok:
        raise value
    return value


__all__ = ["check", "fingerprint", "schema_hash"]
