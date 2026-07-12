# Control

Examples for steering an agent run after it starts: delegation, branching,
injection, and controller-authored graph edits.

- `controller_injection.py` appends controller-provided nodes with minimal graph
  actions.
- `delegation/step_until.py` shows `Flow.run_streaming(..., until=...)` boundaries while
  delegated children fan out under the shared pool.
- `branching/best_of_n.py` runs independent branches and scores the results.
- `branching/fork_repair.py` compares repair attempts across forked runs.
- `injection/` replaces real supervising nodes in a saved graph, edits minimal
  graph copies, and continues each variant with its own `Flow`.
