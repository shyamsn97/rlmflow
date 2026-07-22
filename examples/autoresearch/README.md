# autoresearch

`autoresearch` is a compact autoresearch loop: an agent edits a single GPT
training file, submits GPU trials through Modal, and tries to lower validation
bits per byte on TinyStories.

This example is meant to be easy to understand and cheap enough to iterate on.
It keeps the rlmflow hierarchy/ledger/Modal wrapper, but the ML task is just:

- Hugging Face TinyStories dataset.
- GPT-2 tokenizer.
- A compact GPT model in one editable `train.py`.
- Fixed-time training.
- `val_bpb` as the ranking metric.

## Files

- `prepare.py` — downloads/tokenizes TinyStories and provides fixed dataloaders
  and validation BPB. Agents should not edit this.
- `train.py` — the only file agents edit.
- `program.md` — task-only instructions given to agents.
- `run.py` — rlmflow coordinator that creates trial directories and launches the
  hierarchy. The tree is flat (`--max-depth 1`): root is the planner and each
  turn fans out a parallel wave of implementer children (one per trial), reads
  their results, then plans the next wave. Exploration is wide rather than a long
  serial chain of single batches.
- `modal_runner.py` — Modal execution for each trial directory.
- `pyproject.toml` — dependencies.

## Quick Start

From this directory:

```bash
uv sync
uv run prepare.py
uv run train.py
```

`prepare.py` writes cached token tensors under `~/.cache/autoresearch`.

## Run With rlmflow + Modal

```bash
make run MODEL=gpt-5 GPU=L4 PARALLEL=4 MAX_SUBMISSIONS=16
```

Or from the repo root:

```bash
python examples/autoresearch/run.py \
  --model gpt-5 \
  --gpu L4 \
  --parallel 4 \
  --max-submissions 16
```

Runs are written under:

```text
examples/_runs/autoresearch/<timestamp>/
```

Each run contains `upstream_base/`, copied `trials/`, `ledger.jsonl`, `graph/`,
and a live report:

- `report.json` — only `name`, `hypothesis`, `time_elapsed`, `n`, and `score`
  for each solution, plus the current `best`.
- `summary.md` — the current best solution, a table of all solutions, and plots.
- `scores.svg` and `elapsed.svg` — plots embedded by the Markdown summary.

These report files refresh after every ledger update, so they remain useful
while a run is still in progress or if the coordinator exits early.

## Metric

Lower `val_bpb` is better. The final training summary must contain:

```text
---
val_bpb:          1.234567
training_seconds: 180.1
total_seconds:    195.4
peak_vram_mb:     1234.5
total_tokens_M:   12.3
num_steps:        456
num_params_M:     12.7
depth:            4
```

The host runner parses `val_bpb`, records it in `ledger.jsonl`, and ranks
successful trials by lowest BPB.

## Design

This task deliberately trades realism for faster iteration:

- No custom tokenizer training.
- No Flash Attention kernel dependency.
- No H100 requirement.
- A simple GPT-2-tokenizer BPB evaluator.
- No nanochat-scale architecture.

That makes it a better small example for debugging the rlmflow autoresearch loop
itself before moving back to a larger, more expensive task.
