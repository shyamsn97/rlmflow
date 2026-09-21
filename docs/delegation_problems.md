# Delegation evaluation problems

Delegation is evaluated as a depth treatment inside one harness, not as a set of spawning modes. The public comparison is `official-rlm` plus `rlmflow-depth-0`, `rlmflow-depth-1`, and `rlmflow-depth-2`.

## Conformance

MiniMaze and Sudoku Search are deterministic offline environments. They are small enough to inspect manually and strict enough to catch runtime drift.

Each row has:

- a root-owned environment instance;
- private child instances replayed from the accepted parent prefix;
- one global action budget and one candidate-evaluation budget;
- immutable candidate snapshots; and
- an explicit root-only `select_candidate(candidate_id)` operation.

A model that finishes without the required selection receives zero task score. The environment bridge, not the prompt, enforces this rule.

MiniMaze grades gem completion, exploration coverage, path efficiency, and terminal completion. Sudoku grades valid placement progress and exact completion. Both use the shared environment scorer for selection compliance, action/probe counts, candidate details, and errors.

Run structural validation without model calls:

```bash
python -m benchmarks.eval.calibrate_search
```

## Paper core

OOLONG-Pairs 32K and OOLONG Yahoo 128K answer the primary scientific question: does recursive depth improve long-context information management under one runner contract? These tasks are not mixed with routing canaries or unrelated reasoning packs.

## Interactive evidence

MazeBench tests persistent visual interaction and produces environment replay artifacts. It is valuable qualitative and behavioral evidence, but it does not replace the paper-core long-context comparison.

## Reading results

Treat topology as observed telemetry:

- `single-agent` means no child was launched;
- `delegation-tree` means recursive children were launched; and
- `dependency-dag` means handle-valued inputs created dependency edges.

Topology is not an experimental arm. Configured depth is the treatment. Compare paired rows within a domain and report confidence intervals, failures, selection compliance, token use, and runtime alongside native score.
