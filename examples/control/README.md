# Control

Examples for steering an agent run after it starts: delegation, branching,
injection, and controller-authored graph edits.

- `controller_injection.py` appends controller-provided nodes onto an agent's
  frontier mid-run.
- `streaming_until.py` shows `Flow.run_streaming(..., until=...)` boundaries
  offline, with a scripted model.
- `delegation/step_until.py` shows the same boundaries with a live model while
  delegated children fan out as independent scheduler tasks.
- `delegation/nonblocking.py` queues background children, then awaits their
  persistent handles for results on a later turn.
- `branching/best_of_n.py` runs independent branches and scores the results.
- `branching/fork_repair.py` compares repair attempts across forked runs.
- `injection/` forks a saved run just before the turn where it delegated,
  steers each fork with a message, and continues both forks in one run.
