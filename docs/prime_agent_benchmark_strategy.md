# Paper-aligned benchmark strategy

The release benchmark separates three questions that should not be collapsed into one headline number.

## 1. Conformance

Can each runtime execute identical private-fork, shared-budget, candidate, and selection semantics? MiniMaze and Sudoku Search answer this quickly and offline. Failure here invalidates cross-runtime comparisons.

## 2. Paper-core evidence

Does recursive depth improve long-context information management?

- OOLONG-Pairs at 32K tests pair aggregation in the paper's fixed setting.
- OOLONG Yahoo at 128K tests information scaling on the paper-aligned source and context length.

These are the primary release tasks. Pilot uses a small deterministic subset; full uses the complete frozen eligible rows.

## 3. Interactive evidence

How does depth behave in persistent visual environments?

- ARC-AGI-3 provides public microgames and native RHAE scoring.
- MazeBench provides a watchable embodied trajectory and native completion signals.

These domains headline the replay experience, not the long-context claim.

## Arms

Every non-smoke suite uses exactly:

1. `official-rlm`
1. `rlmflow-depth-0`
1. `rlmflow-depth-1`
1. `rlmflow-depth-2`

There are no "no-spawn", "flat-spawn", or "graph-spawn" modes. The rlmflow arms execute one implementation and differ only in `max_depth`. Official RLM is an upstream reference with the same task and environment contract where its runtime permits parity.

## Evidence standard

Freeze task identity before paid runs. Pair by dataset, example, model, and seed. Report native score, failures, token use, runtime, depth, observed topology, selection compliance, and confidence intervals. A release may report no lift. Claiming depth benefit requires depth 2 to beat depth 0 under the shared acceptance policy; visual appeal alone is not evidence.

## Release sequence

1. Run the zero-cost smoke and structural conformance calibration.
1. Run the conformance pilot across three seeds.
1. Sign the accepted pilot report only if protocol and difficulty gates pass.
1. Run the paper-core pilot, then the frozen full split.
1. Run interactive tasks for replay and behavioral evidence.

Paid pilots are intentionally outside CI and outside refactor verification.
