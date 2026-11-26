# autoresearch

`autoresearch` is a compact autoresearch loop: an agent edits a single GPT
training file, submits GPU trials through Modal, and tries to lower validation
bits per byte on TinyStories.

This example is meant to be easy to understand and cheap enough to iterate on.
The ML task is just:

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
- `baselines/<GPU>.json` — the unmodified train.py's result on each GPU. Training
  is fixed-time, so a faster GPU scores better (L4 1.155, A10G 0.903); `--gpu`
  needs one here.
- `dashboard.py` — a live Gradio view of a run: `python dashboard.py <run dir>`.
- `run.py` — the rlmflow side, about 200 lines. A planner launches waves of
  implementers; both are Claude Code sessions by default (`--planner repl` /
  `--implementer repl` for gpt-6-sol REPL agents). Each implementer works in its own copy of the task files
  (rlmflow's default workspaces), edits `train.py`, and calls `submit_trial`.
  That trains the copy on Modal and returns the trial's row. The planner can start
  a new implementer from any earlier trial with `workspace=row["commit"]`.
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
make run GPU=A10G PARALLEL=4 MAX_SUBMISSIONS=16
make run PLANNER=repl MODEL=gpt-6-sol    # a gpt-6-sol REPL planner over Claude Code implementers
```

Claude Code needs `rlmflow[claude]` and a `claude` login (or `ANTHROPIC_API_KEY`).
Or from the repo root:

```bash
python examples/autoresearch/run.py \
  --planner claude_code --implementer claude_code \
  --gpu A10G \
  --parallel 4 \
  --max-submissions 16
```

Runs are written under:

```text
examples/_runs/autoresearch/<timestamp>/
```

Each run contains:

- `files/` — the task files the agents start from. Each implementer's copy is
  under `files/.rlmflow/worktrees/`, and every trial's files are a commit there.
- `graph/` — the run's graph, checkpointed as it goes. The graph is the ledger:
  each trial's row is the saved return value of its `submit_trial` call. View it
  with `rlmflow view <out>/graph`.
- `ledger.jsonl` — an export of those rows for `plot_progress.py`, rewritten as
  the run goes.

A run ends when the planner finishes. If it takes a turn with no submissions
left and doesn't finish, the host loop stops it (`flow.stop`). `--minutes` puts
a time budget on the whole run.

Karpathy-style progress plot (kept vs discarded + running best):

```bash
python plot_progress.py ../_runs/autoresearch/<timestamp> --gpu A10G
# writes progress.png (or progress.svg if matplotlib is unavailable)
```

`--gpu` starts the plot at that GPU's baseline and draws it as a dashed line. Kept points are labelled with each trial's `change`, the few words its implementer passed to `submit_trial`; `--labels labels.json` (`{slug: label}`) labels a run that predates it.

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

The host runner parses `val_bpb` and ranks trials by lowest BPB.

## Design

This task deliberately trades realism for faster iteration:

- No custom tokenizer training.
- No Flash Attention kernel dependency.
- No H100 requirement.
- A simple GPT-2-tokenizer BPB evaluator.
- No nanochat-scale architecture.

That makes it a better small example for debugging the rlmflow autoresearch loop
itself before moving back to a larger, more expensive task.
