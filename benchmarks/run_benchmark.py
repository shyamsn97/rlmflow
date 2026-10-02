#!/usr/bin/env python3
"""Run one environment-centered benchmark."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.eval.benchmarks import BENCHMARKS

DEFAULT_RUNNERS = {
    "smoke": ["fake"],
    "delegation": ["prime-agent", "official-rlm", "jaz", "rlmflow-depth-1"],
    "long-context": ["official-rlm", "rlmflow-depth-1"],
    "interactive": ["official-rlm", "rlmflow-depth-1"],
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("benchmark", nargs="?", choices=sorted(BENCHMARKS))
    result.add_argument("--split", choices=("pilot", "full"), default="pilot")
    result.add_argument("--runners", action="append")
    result.add_argument("--model", default="openai:gpt-5-mini")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--output", type=Path)
    result.add_argument("--executor", choices=("local", "docker", "modal"), default="local")
    result.add_argument("--parallelism", type=int, default=1)
    result.add_argument("--attempt-timeout", type=int, default=900)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--list", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.list:
        for name in BENCHMARKS:
            print(name)
        return
    if args.benchmark is None:
        parser().error("benchmark is required unless --list is used")
    runners = _runners(args.runners) or DEFAULT_RUNNERS[args.benchmark]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    output_dir = args.output or (
        Path("benchmarks/runs") / f"{args.benchmark}-{args.split}-{stamp}"
    )
    if args.resume and args.output is None:
        parser().error("--resume requires --output pointing to an existing run")
    BENCHMARKS[args.benchmark](
        split=args.split,
        runners=runners,
        model=args.model,
        seed=args.seed,
        output_dir=output_dir,
        executor=args.executor,
        parallelism=args.parallelism,
        attempt_timeout=args.attempt_timeout,
        resume=args.resume,
    )
    print(output_dir)


def _runners(values: list[str] | None) -> list[str]:
    if not values:
        return []
    return [
        runner.strip()
        for value in values
        for runner in value.split(",")
        if runner.strip()
    ]


if __name__ == "__main__":
    main()
