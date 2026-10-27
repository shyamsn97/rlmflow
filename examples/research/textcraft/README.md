# TextCraft RAO

This example trains the same recursive rlmflow policy at every agent-tree depth
on a Gymnasium crafting environment. It is based on Platoon's
[`apga/rao-snapshot`](https://github.com/ApGa/platoon/tree/apga/rao-snapshot/)
TextCraft plugin, but does not depend on the Platoon runtime.

The checked-in `data/*.jsonl` splits contain deterministic generation
parameters. Training items use recipe depths 2–4; validation includes unseen
seeds and depth 5. Opaque item names prevent the model from solving recipes from
Minecraft pretraining.

## Setup

Environment and offline tests only:

```bash
pip install -e '.[textcraft]'
```

Real policy updates:

```bash
pip install -e '.[rao]'
```

Tinker authentication must be configured as required by the Tinker SDK.

## Commands

Score recursive rollouts without updating the model:

```bash
python examples/research/rao_textcraft.py collect \
  --model gpt-5-mini --limit 2 --rollouts 4
```

Run grouped RLOO updates using a shared Qwen policy:

```bash
python examples/research/rao_textcraft.py train \
  --base-model Qwen/Qwen3-8B \
  --iterations 10 --tasks-per-iteration 4 --rollouts 4
```

RLOO is the default: RAO's leave-one-out graph advantages are sent through
Tinker's `importance_sampling` policy-gradient loss. Use `--loss-fn cispo` or
`--loss-fn ppo` to keep the same rewards/advantages and change only the
token-level optimizer.

Compare flat and recursive inference on held-out/deeper recipes:

```bash
python examples/research/rao_textcraft.py eval \
  --state-path 'tinker://YOUR_STATE_PATH' --depths 0,2
```

Artifacts are written under `runs/rao-textcraft/`: per-update trajectory JSONL,
metrics, Tinker sampler/optimizer checkpoints, and evaluation graphs.

## Canonical Platoon data

Point `--platoon-data-dir` at
`plugins/textcraft/platoon/textcraft` from a checkout of the pinned Platoon
branch. The loader reads its `recipes/*.json` and matching task JSONL:

```bash
python examples/research/rao_textcraft.py collect \
  --platoon-data-dir ../platoon/plugins/textcraft/platoon/textcraft \
  --split train --model gpt-5-mini
```

## Reward contract

- Every agent has an independent Gym episode view with its own targets,
  termination state, and step budget. Forked views share the same inventory and
  world lock, so a child can finish without terminating its parent.
- Root success is exact net target inventory at episode completion.
- Every child must receive structured `target_items` when launched.
- Child success requires its target to be produced by that child's subtree;
  sibling work cannot leak credit.
- Every Gym step is retained under `environment.steps` node metadata, so the
  TextCraft reward function accepts a completed graph and does not
  depend on a live environment session.
- The TextCraft default delegation bonus is `0`. Pass
  `--delegation-bonus 0.4` for the generic RAO ablation.
- Leave-one-out advantages require at least two rollouts; four is the default.
- Tinker advantages are normalized by active action-token count, and an update
  with all-zero trajectory credit is skipped.

## Differences from Platoon

- This implementation uses the official synchronous Gymnasium API. Each agent
  owns an `EnvSession`; sessions fork independent episode state while sharing a
  lock-serialized inventory and global environment-step log.
- Training is synchronous: collect a fresh group, update once, then refresh the
  Tinker sampler. Platoon's trainer pipelines near-on-policy rollout workers and
  allows bounded staleness.
- Each model turn currently becomes one Tinker datum. Platoon merges
  prefix-compatible turns to reduce duplicated prompt tokens; this changes
  efficiency, not the RAO credit assigned to action tokens.
