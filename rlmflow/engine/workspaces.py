"""Workspaces: where each agent's files live, and snapshots of them for forks."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rlmflow.graph.nodes import Node


class LocalWorkspace:
    """The default: every agent works in one directory and nothing is snapshotted."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None

    def path(self, agent: Node) -> Path:
        return (self.directory or Path.cwd()).resolve()

    def branch(self, child: Node, parent: Node) -> None:
        return None

    async def snapshot(self, agent: Node) -> Any:
        return None

    async def checkout(self, agent: Node, snapshot: Any) -> None:
        return None


def git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=None if env is None else {**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip() or done.stdout.strip()}")
    return done.stdout.strip()


class GitWorktrees:
    """Each agent works in its own detached git worktree of ``repo``.

    A snapshot is a commit of the worktree's files (tracked and untracked, minus
    ignored ones), made without touching any branch and kept alive by a ref
    under ``refs/rlmflow/``. Worktrees live in ``root``, by default inside the
    repository's git directory so ``git status`` never sees them.
    """

    def __init__(self, repo: str | Path = ".", *, root: str | Path | None = None) -> None:
        self.repo = Path(repo).resolve()
        git_dir = Path(git(self.repo, "rev-parse", "--absolute-git-dir"))
        self.root = Path(root).resolve() if root is not None else git_dir / "rlmflow-worktrees"
        #: Agent id -> the commit its worktree starts from, when not ``HEAD``.
        self.bases: dict[str, str] = {}

    def path(self, agent: Node) -> Path:
        """The agent's worktree, created on first use."""
        tree = self.root / agent.id
        if not tree.exists():
            base = self.bases.pop(agent.id, None) or git(self.repo, "rev-parse", "HEAD")
            self.root.mkdir(parents=True, exist_ok=True)
            git(self.repo, "worktree", "add", "--detach", str(tree), base)
        return tree

    def branch(self, child: Node, parent: Node) -> None:
        """Start ``child`` from ``parent``'s files as they are now."""
        if (self.root / parent.id).exists():
            self.bases[child.id] = self._snapshot(parent)
        elif parent.id in self.bases:
            self.bases[child.id] = self.bases[parent.id]

    async def snapshot(self, agent: Node) -> str:
        return await asyncio.to_thread(self._snapshot, agent)

    async def checkout(self, agent: Node, snapshot: str) -> None:
        await asyncio.to_thread(self._checkout, agent, snapshot)

    def _snapshot(self, agent: Node) -> str:
        tree_dir = self.path(agent)
        head = git(tree_dir, "rev-parse", "HEAD")
        with tempfile.TemporaryDirectory() as scratch:
            index = Path(scratch) / "index"
            real_index = Path(git(tree_dir, "rev-parse", "--path-format=absolute", "--git-path", "index"))
            if real_index.exists():
                shutil.copyfile(real_index, index)
            env = {"GIT_INDEX_FILE": str(index)}
            git(tree_dir, "add", "-A", env=env)
            tree = git(tree_dir, "write-tree", env=env)
        if tree == git(tree_dir, "rev-parse", "HEAD^{tree}"):
            return head
        commit = git(tree_dir, "commit-tree", tree, "-p", head, "-m", f"rlmflow snapshot of {agent.name}")
        git(self.repo, "update-ref", f"refs/rlmflow/{commit}", commit)
        return commit

    def _checkout(self, agent: Node, snapshot: str) -> None:
        tree_dir = self.root / agent.id
        if not tree_dir.exists():
            self.bases[agent.id] = snapshot
            self.path(agent)
            return
        git(tree_dir, "checkout", "--detach", "--force", snapshot)
        git(tree_dir, "clean", "-fd")


__all__ = ["GitWorktrees", "LocalWorkspace"]
