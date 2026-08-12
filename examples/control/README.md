# Control

Examples for steering an agent run after it starts: delegation, branching,
injection, and controller-authored graph edits.

- `controller_injection.py` appends controller-provided nodes onto an agent's
  frontier mid-run.
- `delegation/step_until.py` shows `Flow.run_streaming(..., until=...)` boundaries while
  delegated children fan out as independent scheduler tasks.
- `delegation/reuse_repl.py` contrasts an isolated child with a
  `reuse_repl=True` child that sees and mutates the parent's live Python objects.
- `delegation/nonblocking.py` queues background children, then awaits their
  persistent handles for results on a later turn.
- `branching/best_of_n.py` runs independent branches and scores the results.
- `branching/fork_repair.py` compares repair attempts across forked runs.
- `injection/` replaces real supervising nodes in a saved graph, edits minimal
  graph copies, and continues each variant with its own `Flow`.
