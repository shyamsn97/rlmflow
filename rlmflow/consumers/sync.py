"""Workspace sync consumers."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from rlmflow.consumers.base import StreamConsumer
from rlmflow.graph.nodes import Node

DEFAULT_IGNORE = (".venv", "__pycache__", ".pytest_cache", ".git")


class WorkspaceSync(StreamConsumer):
    """Mirror a run directory into a local workspace while Nodes stream."""

    def __init__(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        *,
        every_s: float = 2.0,
        ignore: tuple[str, ...] = DEFAULT_IGNORE,
    ) -> None:
        self.source_dir = Path(source_dir).resolve()
        self.target_dir = Path(target_dir).resolve()
        self.every_s = every_s
        self.ignore = ignore
        self.last = 0.0
        if self.source_dir == self.target_dir:
            raise ValueError("source_dir and target_dir must be different")
        if _is_relative_to(self.target_dir, self.source_dir):
            raise ValueError("target_dir must not be inside source_dir")

    def handle(self, node: Node) -> None:
        now = time.monotonic()
        if self.every_s and now - self.last < self.every_s:
            return
        self.sync()

    def sync(self) -> None:
        if not self.source_dir.exists():
            return
        shutil.copytree(
            self.source_dir,
            self.target_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*self.ignore),
        )
        self.last = time.monotonic()

    def close(self) -> None:
        self.sync()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


__all__ = ["WorkspaceSync"]
