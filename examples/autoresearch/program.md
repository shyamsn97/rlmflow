# autoresearch

This is a compact autonomous language-modeling research task.

Runtime-specific orchestration details, hierarchy, and examples live in the
runner system prompt. This file describes only the task.

## Files

- `README.md` — repository context.
- `prepare.py` — fixed dataset preparation, batching, and validation metric. Do
  not modify it.
- `train.py` — the editable training script. The agent may change model
  architecture, optimizer, schedule, regularization, batch size, compile mode,
  and training loop details here.

## Task

Train a small GPT-style language model on TinyStories using the GPT-2 tokenizer.
Each experiment has a fixed wall-clock training budget.

The goal is to minimize validation bits per byte:

```text
val_bpb: <float>
```

Lower is better. `val_bpb` is the metric used for ranking.

## How to make progress

Read a prior run log before proposing anything, and let the numbers drive each
hypothesis instead of guessing. A log reports, among other fields, the
smooth-loss trend, the final `lr`, `num_steps`, `total_tokens_M`,
`peak_vram_mb`, `num_params_M`, and `depth`. Form a hypothesis about what is
limiting a given run, change the smallest set of constants that tests it, and
compare the result against the parent's `val_bpb`. Any knob listed under
Constraints is fair game — it is up to you to figure out which ones matter.

## Constraints

Allowed:

- Modify `train.py`.
- Change model depth, width, heads, MLP, normalization, dropout, optimizer,
  schedule, gradient clipping, compile mode, and batch size.
- Simplify code if the validation loss stays the same or improves.

Not allowed:

- Modify `prepare.py`.
- Modify the dataset, validation split, validation function, or metric.
- Add dependencies.
- Run expensive training manually. Use the host `submit_trial(...)` tool.

## Output Format

`train.py` must print a final summary containing:

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

The host runner parses `val_bpb`.

## Logging Results

The host runner writes an append-only `ledger.jsonl`.

Important fields:

```text
n
slug
hypothesis
parent_slug
status
val_bpb
trial_dir
source_path
log_path
```

Use `list_runs()` for a compact best-first view, `best_run()` to find the
current keeper, and `get_run(n)` for full log tails.
