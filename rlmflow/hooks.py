"""Hooks: policy and observation for one ``Flow.run_streaming``, at a few named seams.

A hook is any object passed in ``run_streaming(g, hooks=[...])``. Each method
is optional:

- ``start(flow, g)``: the stream started; set timers with
  ``flow.call_later(g, delay, fn)``.
- ``before_turn(flow, agent)``: a turn is about to start. Return a ``Node`` to
  append first, ``Stop(reason)`` to end the agent instead, or ``None``.
- ``before_launch(flow, step, task, config)``: an agent is about to be
  launched. Return a reason to refuse it, or ``None``.
- ``on_step(flow, step)``: a node landed; called (and awaited, if it returns an
  awaitable) before the step is yielded. Nodes still undrained when the
  stream exits go to hooks only.
- ``end(flow, g)``: the stream ended, normally or not.

Hooks take the Flow first, as harnesses do; ``flow.elapsed(g)`` is how long
the stream has run. Hooks run in list order. The first ``Stop`` and the first refusal win; every
returned node is appended, in order. An exception from a hook fails the run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stop:
    """End the agent now, without its turn; ``reason`` is its ``{"stopped": reason}`` result."""

    reason: str


class Hook:
    """Optional base class; a hook needs only the methods it uses."""


__all__ = ["Hook", "Stop"]
