# Eval Harness

Clean benchmark harness for rlmflow. The core components are:

- `Dataset` - examples and scoring.
- `Model` - inference.
- `Runner` - execution strategy.
- `Logger` - side effects.

Initial datasets:

- `synthetic_needle`
- `oolong`
- `official_browsecomp`
- `official_longbench_v2`
- `official_livecodebench`
- `official_sudoku_extreme`

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

## Real Run

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset oolong official_browsecomp official_longbench_v2 official_livecodebench official_sudoku_extreme \
  --runner vanilla rlmflow-local official-rlm \
  --seeds 0:20 \
  --wandb
```

## Modal Parallelism

The unit of parallelism is one benchmark row: `(dataset example, runner, seed)`.
Each row runs sequentially inside its worker, so rlmflow graph execution
is unchanged. To fan rows out to cheap one-CPU Modal workers:

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset oolong official_browsecomp official_longbench_v2 official_livecodebench official_sudoku_extreme \
  --runner vanilla rlmflow-local official-rlm \
  --seed 0 \
  --limit 50 \
  --executor modal \
  --parallel 20 \
  --best-of-n 1 \
  --modal-cpu 1 \
  --wandb
```

Set `--best-of-n N` to run each logical row N times and keep only the best
scoring attempt in `rows.jsonl`.

Rows and artifacts are written under `benchmarks/runs/<run_id>/`.

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
