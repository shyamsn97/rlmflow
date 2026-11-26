"""Hooks: policy and observation for one ``Flow.stream``, at a few named seams.

A hook is any object passed in ``stream(g, hooks=[...])`` (or ``run``). Each
method is optional:

- ``start(flow, g)``: the stream started; set timers with
  ``flow.call_later(g, delay, fn)``.
- ``before_step(flow, run)``: ``run`` (a task node) is about to take a step.
  Return a ``Node`` to append first, ``Stop(reason)`` to end the run instead,
  or ``None``.
- ``before_launch(flow, node, task, agent)``: ``node``'s code is about to
  launch a run of ``agent``. Return a reason to refuse it, or ``None``.
- ``on_node(flow, node)``: a node finished; called (and awaited, if it returns
  an awaitable) before the node is yielded. Nodes still unyielded when the
  stream exits go to hooks only.
- ``end(flow, g)``: the stream ended, normally or not.

``flow.elapsed(g)`` is how long the stream has run. Hooks run in list order.
The first ``Stop`` and the first refusal win; every returned node is appended,
in order. An exception from a hook fails the run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stop:
    """End the run now, without its step; ``reason`` is its ``{"stopped": reason}`` result."""

    reason: str


__all__ = ["Stop"]
