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

`rlm-core` is the minimal shared comparison suite: S-NIAH, AIME 2025, Sudoku
Extreme, OOLONG, and CodeQA.

`official_livecodebench` requires Docker for scoring. Model-generated code runs
only in a fresh network-disabled, read-only, capability-dropped container with
CPU, memory, process, and wall-clock limits; the harness never executes it with
the host Python interpreter.

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

The short flag limits the number of complete examples; it does not truncate
contexts. Omit the cap explicitly with `--full`:

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset rlm-core \
  --runner vanilla rlmflow-local official-rlm \
  --seed 0 \
  --full
```

## Modal Parallelism

The unit of parallelism is one benchmark row: `(dataset example, runner, seed)`.
Each row runs sequentially inside its worker, so rlmflow graph execution
is unchanged. To fan rows out to cheap one-CPU Modal workers:

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

Set `--best-of-n N` to run each logical row N times and keep only the best
scoring attempt in `rows.jsonl`.

Rows and artifacts are written under `benchmarks/eval/runs/<run_id>/`.

The `official_` prefix follows RLM-Bench task names. AIME and Sudoku are
comparison controls rather than RLM-paper datasets. Full mode uses
benchmark-sized pools: 50 S-NIAH, OOLONG, and Sudoku examples, all 30 AIME
problems, and the complete CodeQA subset. Sudoku's source has millions of rows,
so its adapter additionally bounds the streamed source pool with
`official_sudoku_extreme.sample_window` (default 4096).

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
class MyDataset(Dataset):
    ...
```

Runners:

```python
from benchmarks.eval import runner
from benchmarks.eval.types import Runner


@runner("my-runner")
class MyRunner(Runner):
    ...
```

Decorators are only for component registration. Built-ins are imported
explicitly from package `__init__.py` files.
