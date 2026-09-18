"""Download the pinned public ARC-AGI-3 corpus for offline evaluations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from benchmarks.eval.environments.arc_agi_3 import (
    DEFAULT_ARC3_CORPUS,
    FROZEN_PUBLIC_GAME_IDS,
    PILOT_GAME_IDS,
    download_arc3_corpus,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_ARC3_CORPUS,
        help=f"Offline environment_files directory (default: {DEFAULT_ARC3_CORPUS})",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Download only the three pinned pilot games.",
    )
    args = parser.parse_args(argv)
    game_ids = PILOT_GAME_IDS if args.pilot else FROZEN_PUBLIC_GAME_IDS
    path = download_arc3_corpus(game_ids, corpus=args.corpus)
    print(f"ARC-AGI-3 corpus ready: {path} ({len(game_ids)} games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
