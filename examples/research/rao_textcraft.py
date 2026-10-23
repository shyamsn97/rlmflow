"""Train and evaluate Recursive Agent Optimization on Gym TextCraft.

Examples:
    # Rollout/scoring smoke run with an API model (no policy update)
    python examples/research/rao_textcraft.py collect --model gpt-5-mini --limit 2

    # Real grouped RLOO updates through Tinker
    python examples/research/rao_textcraft.py train --iterations 10 --limit 16

    # Compare flat and recursive inference on held-out deeper recipes
    python examples/research/rao_textcraft.py eval --state-path tinker://... --depths 0,2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.research.textcraft.dataset import (  # noqa: E402
    DATA_DIR,
    load_dataset,
    load_platoon_tasks,
)
from examples.research.textcraft.env import (  # noqa: E402
    open_textcraft,
    textcraft_graph_rewards,
)
from rlmflow.llm import client_for  # noqa: E402
from rlmflow.rao import (  # noqa: E402
    Budget,
    Collector,
    RolloutFlow,
    RunConfig,
    TinkerConfig,
    TinkerTrainer,
    stats,
    train,
    trajectories,
    write_jsonl,
)


def add_data_args(parser: argparse.ArgumentParser, *, split: str) -> None:
    parser.add_argument("--split", default=split)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATA_DIR,
        help="generated split JSONL directory",
    )
    parser.add_argument(
        "--platoon-data-dir",
        type=Path,
        help="canonical Platoon textcraft package directory containing recipes/",
    )


def add_rollout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument("--max-children", type=int, default=4)
    parser.add_argument("--max-agents", type=int, default=16)
    parser.add_argument("--max-env-steps", type=int, default=192)
    parser.add_argument("--max-wall-time", type=float, default=None)
    parser.add_argument(
        "--delegation-bonus",
        type=float,
        default=0.0,
        help="TextCraft paper default is 0; pass 0.4 for the RAO ablation",
    )


def add_tinker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--renderer", default="qwen3")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--loss-fn",
        default="rloo",
        choices=("rloo", "importance_sampling", "cispo", "ppo"),
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect and score without updating")
    add_data_args(collect, split="train")
    add_rollout_args(collect)
    collect.add_argument("--model", default="gpt-5-mini")
    collect.add_argument("--out-dir", type=Path, default=Path("runs/rao-textcraft/collect"))

    train_parser = subparsers.add_parser("train", help="run Tinker RLOO updates")
    add_data_args(train_parser, split="train")
    add_rollout_args(train_parser)
    add_tinker_args(train_parser)
    train_parser.add_argument("--iterations", type=int, default=10)
    train_parser.add_argument("--tasks-per-iteration", type=int, default=4)
    train_parser.add_argument("--checkpoint-every", type=int, default=1)
    train_parser.add_argument("--out-dir", type=Path, default=Path("runs/rao-textcraft/train"))

    evaluate = subparsers.add_parser("eval", help="evaluate held-out tasks")
    add_data_args(evaluate, split="validation")
    add_rollout_args(evaluate)
    add_tinker_args(evaluate)
    evaluate.add_argument("--model", default=None, help="API model instead of Tinker state")
    evaluate.add_argument("--depths", default="0,2", help="comma-separated recursion depths")
    evaluate.add_argument("--out-dir", type=Path, default=Path("runs/rao-textcraft/eval"))
    return root


def tasks_for(args: argparse.Namespace):
    if args.platoon_data_dir:
        tasks = load_platoon_tasks(
            args.platoon_data_dir,
            split=args.split,
            limit=args.limit,
        )
    else:
        tasks = load_dataset(args.split, data_dir=args.dataset_dir, limit=args.limit)
    if not tasks:
        raise ValueError("the selected TextCraft dataset is empty")
    return tasks


def budget(args: argparse.Namespace, *, depth: int | None = None) -> Budget:
    return Budget(
        max_depth=args.max_depth if depth is None else depth,
        max_iters=args.max_iters,
        max_children_per_agent=args.max_children,
        max_total_agents=args.max_agents,
        max_env_steps=args.max_env_steps,
        max_wall_time=args.max_wall_time,
    )


def tinker_config(args: argparse.Namespace) -> TinkerConfig:
    return TinkerConfig(
        base_model=args.base_model,
        state_path=args.state_path,
        renderer=args.renderer,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        loss_fn=args.loss_fn,
    )


async def collect_once(
    sampler: Any,
    task_specs: list,
    args: argparse.Namespace,
    *,
    depth: int,
):
    flow = RolloutFlow(sampler, budget=budget(args, depth=depth))
    collector = Collector(
        flow,
        open_textcraft,
        rollouts_per_task=args.rollouts,
        delegation_bonus=args.delegation_bonus,
        reward_fn=textcraft_graph_rewards,
    )
    try:
        trees = await collector.collect(task_specs)
        items = trajectories(trees, flow)
        return trees, items, dict(flow.refusals)
    finally:
        await flow.aclose()


def save_rollouts(trees: list, out_dir: Path) -> None:
    """Save each task's graph once; it holds every rollout of that task."""
    graphs = {tree.graph.id: (tree.task.id, tree.graph) for tree in trees}
    for task_id, graph in graphs.values():
        graph.save(out_dir / task_id)


async def run_collect(args: argparse.Namespace) -> None:
    specs = [task.to_spec() for task in tasks_for(args)]
    trees, items, refusals = await collect_once(
        client_for(args.model),
        specs,
        args,
        depth=args.max_depth,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_rollouts(trees, args.out_dir / "rollouts")
    write_jsonl(items, args.out_dir / "examples.jsonl")
    record = {**stats(items, trees), "refusals": refusals}
    (args.out_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


async def run_train(args: argparse.Namespace) -> None:
    task_specs = [task.to_spec() for task in tasks_for(args)]
    trainer = await asyncio.to_thread(TinkerTrainer, tinker_config(args))
    records = await train(
        task_specs,
        open_textcraft,
        trainer,
        RunConfig(
            out_dir=args.out_dir,
            iterations=args.iterations,
            tasks_per_iteration=args.tasks_per_iteration,
            rollouts_per_task=args.rollouts,
            budget=budget(args),
            checkpoint_every=args.checkpoint_every,
        ),
        collector_kwargs={
            "delegation_bonus": args.delegation_bonus,
            "reward_fn": textcraft_graph_rewards,
        },
    )
    print(json.dumps(records[-1] if records else {}, indent=2))


async def run_eval(args: argparse.Namespace) -> None:
    task_specs = [task.to_spec() for task in tasks_for(args)]
    if args.model:
        sampler = client_for(args.model)
    else:
        trainer = await asyncio.to_thread(TinkerTrainer, tinker_config(args))
        sampler = await asyncio.to_thread(lambda: trainer.sampler)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for depth in [int(value) for value in args.depths.split(",") if value.strip()]:
        trees, items, refusals = await collect_once(
            sampler,
            task_specs,
            args,
            depth=depth,
        )
        results[str(depth)] = {**stats(items, trees), "refusals": refusals}
        save_rollouts(trees, args.out_dir / f"depth-{depth}" / "rollouts")
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


async def main() -> None:
    args = parser().parse_args()
    if args.rollouts < 2:
        raise ValueError("--rollouts must be at least 2 for leave-one-out advantages")
    if args.command == "collect":
        await run_collect(args)
    elif args.command == "train":
        await run_train(args)
    else:
        await run_eval(args)


if __name__ == "__main__":
    asyncio.run(main())
