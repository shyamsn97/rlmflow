# benchmarks/

Runnable benchmark harnesses for rlmflow.

The canonical harness is `benchmarks/eval/`. It is intentionally small and built
around four components:

- `Dataset` - yields examples and scores predictions.
- `Model` - wraps inference.
- `Runner` - executes an example (`vanilla`, `rlmflow-local`, `official-rlm`, etc.).
- `Logger` - writes JSONL, console output, reports, or W&B metrics.

Initial datasets:

- `synthetic_needle` - deterministic needle-in-haystack smoke task.
- `oolong` - first real long-context dataset.
- `official_browsecomp` - BrowseComp-Plus deep-research QA over fixed documents.
- `official_longbench_v2` - LongBench-v2 all-domain multiple-choice/QA.
- `official_livecodebench` - LiveCodeBench code generation with public tests.
- `official_sudoku_extreme` - Sudoku Extreme solution checking.

## Running

```bash
python -m benchmarks.eval --help
```

Smoke:

```bash
make eval-smoke
```

Direct equivalent:

```bash
python -m benchmarks.eval \
  --model fake \
  --dataset synthetic_needle \
  --runner fake vanilla rlmflow-local \
  --seeds 0:3 \
  --dataset-param synthetic_needle.records=8 \
  --dataset-param synthetic_needle.filler_words=2 \
  --runner-param rlmflow-local.max_iters=3 \
  --runner-param rlmflow-local.max_depth=1
```

Real run:

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset oolong official_browsecomp official_longbench_v2 official_livecodebench official_sudoku_extreme \
  --runner vanilla rlmflow-local official-rlm \
  --seeds 0:20 \
  --wandb
```

Modal parallel run:

```bash
python -m benchmarks.eval \
  --model openai:gpt-5-mini \
  --dataset oolong official_browsecomp official_longbench_v2 official_livecodebench official_sudoku_extreme \
  --runner vanilla rlmflow-local official-rlm \
  --seed 0 \
  --limit 50 \
  --executor modal \
  --parallel 10 \
  --best-of-n 1 \
  --modal-cpu 1 \
  --wandb
```

Increase `--best-of-n` to duplicate each logical benchmark row and keep the
best-scoring attempt.

`official_browsecomp` is large. Download it once before running:

```bash
python -c "from datasets import load_dataset; load_dataset('Tevatron/browsecomp-plus').save_to_disk('evals/data/browsecomp_plus')"
```

Every run writes:

```text
benchmarks/runs/<run_id>/
  config.json
  rows.jsonl
  summary.json
  report.md
  artifacts/<dataset>/<example_id>/<runner>/
```

## Current API notes

The harness tracks the library's async LLM clients and Node-only runner:

- **Models** (`benchmarks/eval/models/openai.py`, `anthropic.py`) wrap
  `OpenAIClient` / `AnthropicClient`, whose `completion` is async. The sync
  `Model.complete` bridge uses `asyncio.run(...)`.
- **`rlmflow-local` runner** builds `start(query=example.prompt, inputs=...)` and
  drives with `flow.run_streaming(root)` — `Example.prompt` is task text,
  while model and prompt-profile settings live on the root `UserQuery`.
