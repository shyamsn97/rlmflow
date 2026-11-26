"""Workspaces: where each agent's files live, and snapshots of them for forks."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from rlmflow.graph.nodes import Node
from rlmflow.runtime.runtime import repl_key


class LocalWorkspace:
    """The default: every agent works in one directory and nothing is snapshotted."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None

    def path(self, agent: Node) -> Path:
        return (self.directory or Path.cwd()).resolve()

    def branch(self, child: Node, parent: Node, source: Any = None) -> None:
        if source is not None and source != "shared":
            raise ValueError(
                "this Flow's agents share one directory; give its runtime a working "
                "directory, or use Flow(workspaces='copies') or 'isolated' to start "
                "agents from other files"
            )

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
        raise RuntimeError(
            f"git {' '.join(args)} failed: {done.stderr.strip() or done.stdout.strip()}"
        )
    return done.stdout.strip()


class GitWorktrees:
    """Each agent works in its own detached git worktree of ``repo``, one per ``repl_id``.

    A snapshot is a commit of the worktree's files (tracked and untracked, minus
    ignored ones), made without touching any branch and kept alive by a ref
    under ``refs/rlmflow/``. Worktrees live in ``root``, by default inside the
    repository's git directory so ``git status`` never sees them.
    """

    def __init__(self, repo: str | Path = ".", *, root: str | Path | None = None) -> None:
        self.repo = Path(repo).resolve()
        git_dir = Path(git(self.repo, "rev-parse", "--absolute-git-dir"))
        self.root = Path(root).resolve() if root is not None else git_dir / "rlmflow-worktrees"
        #: ``repl_id`` -> the commit its worktree starts from, when not ``HEAD``.
        self.bases: dict[str, str] = {}

    def path(self, agent: Node) -> Path:
        """The agent's worktree, created on first use."""
        tree = self.root / repl_key(agent)
        if not tree.exists():
            base = self.bases.pop(repl_key(agent), None) or git(self.repo, "rev-parse", "HEAD")
            self.root.mkdir(parents=True, exist_ok=True)
            git(self.repo, "worktree", "add", "--detach", str(tree), base)
        return tree

    isolates_launches = True

    def branch(self, child: Node, parent: Node, source: Any = None) -> None:
        """Start ``child`` from ``parent``'s files as they are now, or from ``source``'s."""
        if source == "shared":
            raise ValueError("GitWorktrees gives every agent its own worktree; none share one")
        if isinstance(source, str):
            self.bases[repl_key(child)] = git(self.repo, "rev-parse", "--verify", source)
            return
        parent = source if source is not None else parent
        if (self.root / repl_key(parent)).exists():
            self.bases[repl_key(child)] = self._snapshot(parent)
        elif repl_key(parent) in self.bases:
            self.bases[repl_key(child)] = self.bases[repl_key(parent)]

    async def snapshot(self, agent: Node) -> str:
        return await asyncio.to_thread(self._snapshot, agent)

    async def checkout(self, agent: Node, snapshot: str) -> None:
        await asyncio.to_thread(self._checkout, agent, snapshot)

    def adopt(self, agent: Node, commit: str) -> None:
        self._checkout(agent, commit)

    def commit(self, agent: Node) -> str:
        return self._snapshot(agent)

    def _snapshot(self, agent: Node) -> str:
        tree_dir = self.path(agent)
        head = git(tree_dir, "rev-parse", "HEAD")
        with tempfile.TemporaryDirectory() as scratch:
            index = Path(scratch) / "index"
            real_index = Path(
                git(tree_dir, "rev-parse", "--path-format=absolute", "--git-path", "index")
            )
            if real_index.exists():
                shutil.copyfile(real_index, index)
            env = {"GIT_INDEX_FILE": str(index)}
            git(tree_dir, "add", "-A", env=env)
            tree = git(tree_dir, "write-tree", env=env)
        if tree == git(tree_dir, "rev-parse", "HEAD^{tree}"):
            return head
        commit = git(
            tree_dir,
            "commit-tree",
            tree,
            "-p",
            head,
            "-m",
            f"rlmflow snapshot of {agent.name}",
        )
        git(self.repo, "update-ref", f"refs/rlmflow/{commit}", commit)
        return commit

    def _checkout(self, agent: Node, snapshot: str) -> None:
        tree_dir = self.root / repl_key(agent)
        if not tree_dir.exists():
            self.bases[repl_key(agent)] = snapshot
            self.path(agent)
            return
        git(tree_dir, "checkout", "--detach", "--force", snapshot)
        git(tree_dir, "clean", "-fd")


#: ``commit-tree`` needs an identity; snapshots shouldn't depend on the user's git config.
IDENTITY = {
    "GIT_AUTHOR_NAME": "rlmflow",
    "GIT_AUTHOR_EMAIL": "rlmflow@localhost",
    "GIT_COMMITTER_NAME": "rlmflow",
    "GIT_COMMITTER_EMAIL": "rlmflow@localhost",
}
COMMIT = re.compile(r"[0-9a-f]{7,40}")


class Copies:
    """Top-level agents work in ``directory``; each launched agent in a copy of its launcher's files.

    Copies are git worktrees of a hidden repository in ``directory/.rlmflow/git``,
    so they work whether or not ``directory`` is a git checkout, and its own
    ``.git`` is never touched. A copy's snapshot is a commit; forks restore
    copies from them. Files ignored by ``directory``'s ``.gitignore`` and files
    over ``max_file_bytes`` aren't copied. ``directory`` itself is never
    rewound; ``isolate_roots=True`` gives top-level agents copies too.
    """

    isolates_launches = True

    def __init__(
        self,
        directory: str | Path | None = None,
        *,
        isolate_roots: bool = False,
        max_file_bytes: int = 50_000_000,
    ) -> None:
        self.directory = Path(directory or Path.cwd()).resolve()
        self.isolate_roots = isolate_roots
        self.max_file_bytes = max_file_bytes
        self.state = self.directory / ".rlmflow"
        self.git_dir = self.state / "git"
        self.root = self.state / "worktrees"
        #: Files left out of copies for their size.
        self.skipped: set[str] = set()
        #: ``repl_id`` -> the commit its copy starts from, for copies made by ``checkout``.
        self.bases: dict[str, str] = {}
        #: Agents' tools commit from worker threads.
        self._lock = threading.RLock()

    # -- where agents work ----------------------------------------------------

    def owner(self, agent: Node) -> Node | None:
        """The agent whose copy ``agent`` works in; ``None``: ``directory`` itself."""
        while agent.data.get("shares_workspace") and agent.parent() is not None:
            agent = agent.parent()  # type: ignore[assignment]
        return agent if self.isolate_roots or agent.parent() is not None else None

    def path(self, agent: Node) -> Path:
        owner = self.owner(agent)
        if owner is None:
            return self.directory
        tree = self.root / repl_key(owner)
        with self._lock:
            if tree.exists():
                return tree
            base = (
                self.bases.pop(repl_key(owner), None)
                or owner.data.get("workspace_base")
                or self._snapshot_directory()
            )
            self.root.mkdir(parents=True, exist_ok=True)
            self._git(self.directory, "worktree", "add", "--detach", str(tree), base)
            return tree

    def branch(self, child: Node, parent: Node, source: Any = None) -> None:
        """Start ``child`` from ``source``: ``None`` for ``parent``'s files, ``"shared"``
        to work in ``parent``'s directory, another agent, or a commit."""
        if source == "shared":
            child.data["shares_workspace"] = True
            return
        if isinstance(source, str):
            if not COMMIT.fullmatch(source):
                raise ValueError(
                    f"workspace must be 'shared', an agent or a commit, not {source!r}"
                )
            child.data["workspace_base"] = self._git(
                self.directory, "rev-parse", "--verify", f"{source}^{{commit}}"
            )
            return
        child.data["workspace_base"] = self.commit(source if source is not None else parent)

    def commit(self, agent: Node) -> str:
        """A commit of ``agent``'s files as they are now."""
        owner = self.owner(agent)
        return self._snapshot_directory() if owner is None else self._snapshot(owner)

    def directory_commit(self) -> str:
        """A commit of ``directory``'s files as they are now."""
        return self._snapshot_directory()

    def diff(self, a: str, b: str) -> str:
        return self._git(self.directory, "diff", a, b)

    def tree_hash(self, commit: str) -> str:
        """The hash of ``commit``'s files: equal for equal files, whoever committed them."""
        return self._git(self.directory, "rev-parse", f"{commit}^{{tree}}")

    def export(self, commit: str, destination: str | Path) -> Path:
        """Write ``commit``'s files to ``destination``, outside any agent's copy."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            env = {
                "GIT_DIR": str(self.git_dir),
                "GIT_WORK_TREE": str(destination),
                "GIT_INDEX_FILE": str(Path(scratch) / "index"),
            }
            self._init()
            git(destination, "read-tree", commit, env=env)
            git(destination, "checkout-index", "-a", "-f", env=env)
        return destination

    # -- snapshots ------------------------------------------------------------

    async def snapshot(self, agent: Node) -> str | None:
        owner = self.owner(agent)
        if owner is None:
            return None
        return await asyncio.to_thread(self._snapshot, owner)

    async def checkout(self, agent: Node, snapshot: str) -> None:
        """Restore ``agent``'s copy to ``snapshot``; ``directory`` itself is left alone."""
        owner = self.owner(agent)
        if owner is not None:
            await asyncio.to_thread(self._checkout, owner, snapshot)

    def adopt(self, agent: Node, commit: str) -> None:
        """Make ``agent``'s files ``commit``'s; in ``directory``, only overwrite, never delete."""
        owner = self.owner(agent)
        if owner is not None:
            self._checkout(owner, commit)
            return
        with tempfile.TemporaryDirectory() as scratch:
            env = {**self._env(), "GIT_INDEX_FILE": str(Path(scratch) / "index")}
            git(self.directory, "checkout", commit, "--", ".", env=env)

    def _env(self) -> dict[str, str]:
        self._init()
        return {"GIT_DIR": str(self.git_dir), "GIT_WORK_TREE": str(self.directory)}

    def _git(self, cwd: Path, *args: str) -> str:
        self._init()
        return git(cwd, *args, env={"GIT_DIR": str(self.git_dir)})

    def _init(self) -> None:
        ready = self.state / ".gitignore"
        if ready.exists():
            return
        with self._lock:
            if not ready.exists():
                self._create()

    def _create(self) -> None:
        self.git_dir.mkdir(parents=True, exist_ok=True)
        git(self.git_dir, "init", "--bare", "-q")
        (self.git_dir / "info").mkdir(exist_ok=True)
        (self.git_dir / "info" / "exclude").write_text(".rlmflow/\n")
        (self.state / ".gitignore").write_text("*\n")

    def _add_all(self, cwd: Path, env: dict[str, str]) -> None:
        listed = git(cwd, "ls-files", "-z", "-o", "-m", "--exclude-standard", env=env)
        big = []
        for name in filter(None, listed.split("\0")):
            file = cwd / name
            if file.is_file() and file.stat().st_size > self.max_file_bytes:
                big.append(name)
        self.skipped.update(big)
        git(cwd, "add", "-A", "--", ".", *(f":(exclude,literal){name}" for name in big), env=env)

    def _commit(self, cwd: Path, env: dict[str, str], parent: str | None, what: str) -> str:
        with tempfile.TemporaryDirectory() as scratch:
            index = Path(scratch) / "index"
            env = {**env, "GIT_INDEX_FILE": str(index)}
            if parent is not None:
                git(cwd, "read-tree", parent, env=env)
            self._add_all(cwd, env)
            tree = git(cwd, "write-tree", env=env)
        if parent is not None and tree == git(cwd, "rev-parse", f"{parent}^{{tree}}", env=env):
            return parent
        commit = git(
            cwd,
            "commit-tree",
            tree,
            *(("-p", parent) if parent is not None else ()),
            "-m",
            f"rlmflow snapshot of {what}",
            env={**env, **IDENTITY},
        )
        git(cwd, "update-ref", f"refs/rlmflow/{commit}", commit, env=env)
        return commit

    def _snapshot_directory(self) -> str:
        with self._lock:
            return self._snapshot_directory_locked()

    def _snapshot_directory_locked(self) -> str:
        env = self._env()
        try:
            last = git(
                self.directory, "rev-parse", "--verify", "-q", "refs/rlmflow/directory", env=env
            )
        except RuntimeError:
            last = None
        commit = self._commit(self.directory, env, last, str(self.directory))
        git(self.directory, "update-ref", "refs/rlmflow/directory", commit, env=env)
        return commit

    def _snapshot(self, agent: Node) -> str:
        tree = self.path(agent)
        return self._commit(tree, {}, git(tree, "rev-parse", "HEAD"), agent.name)

    def _checkout(self, agent: Node, snapshot: str) -> None:
        tree = self.root / repl_key(agent)
        if not tree.exists():
            self.bases[repl_key(agent)] = snapshot
            self.path(agent)
            return
        git(tree, "checkout", "--detach", "--force", snapshot)
        git(tree, "clean", "-fd")


def make_workspaces(workspaces: Any, runtime: Any) -> Any:
    """``Flow(workspaces=...)``: ``"copies"``, ``"isolated"``, ``"shared"``, an object, or ``None``.

    ``None`` is ``"copies"`` when the runtime runs on this machine in a
    directory you gave it (and git is installed), and ``"shared"`` otherwise.
    """
    directory = runtime.working_directory
    if workspaces is None:
        local = runtime.location is None and directory is not None
        workspaces = "copies" if local and shutil.which("git") is not None else "shared"
    if workspaces == "shared":
        return LocalWorkspace(directory)
    if workspaces == "copies":
        return Copies(directory)
    if workspaces == "isolated":
        return Copies(directory, isolate_roots=True)
    if isinstance(workspaces, str):
        raise ValueError(
            f"workspaces must be 'copies', 'isolated', 'shared' or an object, not {workspaces!r}"
        )
    return workspaces


__all__ = ["Copies", "GitWorktrees", "LocalWorkspace", "make_workspaces"]
