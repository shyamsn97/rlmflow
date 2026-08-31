# Eval Harness

Clean benchmark harness for rlmflow. The core components are:

- `Dataset` - examples and scoring.
- `Model` - inference.
- `Runner` - execution strategy.
- `Logger` - side effects.

Initial datasets:

- `synthetic_needle`
- `official_sniah`
- `oolong`
- `official_codeqa`
- `official_aime_2025`
- `official_sudoku_extreme`
- `official_browsecomp`
- `official_longbench_v2`
- `official_livecodebench`

`rlm-core` is the minimal shared comparison suite: S-NIAH, AIME 2025, Sudoku Extreme, OOLONG, and CodeQA.

`delegation-suite` is the frozen 20-problem task-graph suite spanning ParallelQA, MuSiQue, 2WikiMultiHopQA, GRS-QA, EntailmentBank, DABstep, NATURAL PLAN, PlanBench, LongBench-v2 CodeQA, Sudoku-Bench, and ARC-AGI-2. Its runner records root-plus-descendant usage and deterministic launch, wait, completion, result-consumption, failure, and late-work telemetry.

`official_livecodebench` requires Docker for scoring. Model-generated code runs only in a fresh network-disabled, read-only, capability-dropped container with CPU, memory, process, and wall-clock limits; the harness never executes it with the host Python interpreter.

## Smoke

```bash
make eval-smoke
```

Direct CLI:

```bash
python -m benchmarks.eval \
  --model fake \
  --dataset synthetic_needle \
  --runner fake vanilla rlmflow-local \
  --seeds 0:3 \
  --dataset-param synthetic_needle.records=8 \
  --dataset-param synthetic_needle.filler_words=2
```

## Minimal Official-RLM Comparison

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset rlm-core \
  --runner vanilla rlmflow-local official-rlm \
  --seed 0 \
  --limit 5
```

The short flag limits the number of complete examples; it does not truncate contexts. Omit the cap explicitly with `--full`:

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset rlm-core \
  --runner vanilla rlmflow-local official-rlm \
  --seed 0 \
  --full
```

## Task-Graph Delegation

Run the stable five-task iteration suite:

```bash
make eval-delegation-five EVAL_MODEL=gpt-5-mini
```

It contains ParallelQA 22 and Sudoku as local-restraint controls, plus ParallelQA 94, PlanBench 16, and ARC-AGI 20 for parallel lookup, candidate verification, and multi-branch synthesis.

Compare rlmflow-local with official RLM on one model, or run the standard two-model matrix:

```bash
make eval-delegation-five-compare EVAL_MODEL=gpt-5-mini
make eval-delegation-five-matrix
```

Run the broader ten-task comparison:

```bash
make eval-delegation-ten-compare EVAL_MODEL=gpt-5-mini
```

Rerun only the five failures targeted by the current fixes:

```bash
make eval-delegation-regression EVAL_MODEL=gpt-5-mini
```

Run the seven-source canary:

```bash
make eval-delegation-phase1 EVAL_MODEL=gpt-5-mini
```

Run all 20 frozen problems:

```bash
make eval-delegation EVAL_MODEL=gpt-5-mini
```

The default condition is `current_policy`. Set
`EVAL_DELEGATION_CONDITION=local` or `capability_only` for the controls. All
conditions use the same aggregate token budget; the production runner uses
`max_depth=2`. Dataset revisions, source IDs, licenses, and adaptation metadata
live in `benchmarks/eval/delegation/manifest.py`.

Run the same frozen problems through the upstream `alexzhang13/rlm` implementation, then write a paired JSON and Markdown report:

```bash
pip install rlms
make eval-delegation-official EVAL_MODEL=gpt-5-mini
make eval-delegation-report \
  EVAL_RLMFLOW_RUN=benchmarks/eval/runs/<rlmflow-run> \
  EVAL_OFFICIAL_RUN=benchmarks/eval/runs/<official-run> \
  EVAL_COMPARISON_OUT=benchmarks/eval/comparisons/<comparison-name>
```

## Modal Parallelism

The unit of parallelism is one benchmark row: `(dataset example, runner, seed)`. Each row runs sequentially inside its worker, so rlmflow graph execution is unchanged. To fan rows out to cheap one-CPU Modal workers:

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset rlm-core \
  --runner vanilla rlmflow-local official-rlm \
  --seed 0 \
  --limit 5 \
  --executor modal \
  --parallel 20 \
  --best-of-n 1 \
  --modal-cpu 1 \
  --wandb
```

Set `--best-of-n N` to run each logical row N times and keep only the best scoring attempt in `rows.jsonl`.

Rows and artifacts are written under `benchmarks/eval/runs/<run_id>/`.

The `official_` prefix follows RLM-Bench task names. AIME and Sudoku are comparison controls rather than RLM-paper datasets. Full mode uses benchmark-sized pools: 50 S-NIAH, OOLONG, and Sudoku examples, all 30 AIME problems, and the complete CodeQA subset. Sudoku's source has millions of rows, so its adapter additionally bounds the streamed source pool with `official_sudoku_extreme.sample_window` (default 4096).

`official_browsecomp` is large. Download it once before running:

```bash
python -c "from datasets import load_dataset; load_dataset('Tevatron/browsecomp-plus').save_to_disk('evals/data/browsecomp_plus')"
```

## Adding Components

Datasets:

```python
from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset


@dataset("my_dataset")
class MyDataset(Dataset): ...
```

Runners:

```python
from benchmarks.eval import runner
from benchmarks.eval.types import Runner


@runner("my-runner")
class MyRunner(Runner): ...
```

Decorators are only for component registration. Built-ins are imported explicitly from package `__init__.py` files.
