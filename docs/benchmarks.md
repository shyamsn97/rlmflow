# Evaluation suites

The benchmark package has one pipeline:

`SuiteSpec → Dataset → Example → Runner → Prediction → Score → Row`

`benchmarks/eval/suite.py` is the complete public suite registry. There are no compatibility aliases or hidden legacy sets.

## Public suites

- `smoke` runs `synthetic_needle` with the fake runner and makes no network calls.
- `conformance` runs MiniMaze and Sudoku Search. These fast offline environments test private forks, shared budgets, candidate submission, and root selection.
- `core` runs the paper-aligned OOLONG-Pairs 32K and OOLONG Yahoo 128K tasks. This is the primary long-context evidence.
- `interactive` runs the official MazeBench world. It provides persistent visual interaction and replay-oriented evidence.

Use `--split pilot` for bring-up and `--split full` for the frozen evidence split. Split selection is part of each dataset; pilot/full are not separate suite names.

## Comparison contract

Every suite except `smoke` runs exactly:

1. `official-rlm`
1. `rlmflow-depth-0`
1. `rlmflow-depth-1`
1. `rlmflow-depth-2`

The three rlmflow arms instantiate the same runner class. `max_depth` is the only treatment variable. Official RLM is a separate upstream reference and is reported as such.

The runtime bindings share task bytes, filesystem fixtures, environment operation specifications, episode state, action/candidate budgets, scoring, and normalized telemetry. The only intentional environment asymmetry is that rlmflow can represent handle-valued dependency edges while upstream RLM exposes recursive subcalls.

## Commands

```bash
make -f benchmarks/Makefile eval-list-suites
make -f benchmarks/Makefile eval-smoke
make -f benchmarks/Makefile eval-conformance
make -f benchmarks/Makefile eval-core
make -f benchmarks/Makefile eval-interactive
```

Set `EVAL_SPLIT=full` for full manifests. The non-smoke commands use paid model APIs; the refactor and CI checks do not launch them.

Direct CLI use follows the same registry:

```bash
python -m benchmarks.eval --dataset conformance --split pilot --full
```

A direct dataset name requires an explicit `--runner`, preventing accidental departures from the canonical matrix.

## Evidence roles

The `core` suite is the primary scientific comparison. OOLONG-Pairs uses the paper's fixed 32K pair-aggregation setting, while OOLONG Yahoo tests information scaling at 128K. Compare paired rows by dataset, model, runner, and seed.

MazeBench is supporting interactive evidence. It uses the official fixed H×I, top-diagonal, yaw-0 start and a shared action budget. The root explores a persistent ASCII world and is scored on gems collected, with 100 gems as the native win condition. Child trajectories remain private probes. Generated visual reports synchronize execution topology, actions, candidate state, and environment replay.

## Results

The stable serialized units remain `Row` and `Prediction.metrics`, so old JSONL artifacts remain readable. New runners emit one canonical telemetry vocabulary at the runner boundary. Matrix, comparison, Markdown, release-gate, and visual report commands are read-only consumers of that vocabulary.

Environment runs include:

- accepted, primary, and probe action counts;
- private instance and replay events;
- candidate count, selected candidate, and selection regret;
- selection compliance; and
- domain-native score details.

Do not average native scores across domains. Compare runners within each dataset/domain and seed.

## Acceptance

`python -m benchmarks.eval.calibrate_search` performs structural conformance checks without model calls. `benchmarks/eval/acceptance.py` is the single source for required runners, domains, seeds, runtime limits, selection compliance, and difficulty thresholds. A release does not require positive depth lift. Any delegation-benefit claim requires the measured depth-2 arm to beat depth 0 under the shared contract.
