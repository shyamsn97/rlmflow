"""Git worktrees: one directory per agent, snapshots on nodes, forks restore files."""

import asyncio
import shutil

import pytest
from helpers import ScriptedFlow, StubLLM, code_nodes, first_user, trailing_user_text
from helpers import reasoning_repl as block

from rlmflow import AgentConfig, GitWorktrees, Graph
from rlmflow.runtime import LocalRuntime
from rlmflow.workspaces import Copies, LocalWorkspace, git

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.email", "t@example.com")
    git(path, "config", "user.name", "t")
    (path / "notes.txt").write_text("base\n")
    git(path, "add", "notes.txt")
    git(path, "commit", "-q", "-m", "base")
    return path


def run(flow, g):
    async def go():
        try:
            await flow.arun(g)
        finally:
            await flow.aclose()

    asyncio.run(go())


def test_code_nodes_run_in_the_agents_worktree_and_carry_snapshots(repo):
    def reply(messages):
        last = trailing_user_text(messages)
        if "first" in last:
            return block("open('notes.txt', 'a').write('second\\n')\nprint('second')")
        if "second" in last:
            return block("submit(open('notes.txt').read())")
        return block(
            "import os\nopen('notes.txt', 'a').write('first\\n')\nprint('first', os.getcwd())"
        )

    workspaces = GitWorktrees(repo)
    flow = ScriptedFlow(
        StubLLM(reply), runtime=LocalRuntime(repl_timeout=10), workspaces=workspaces
    )
    g = Graph()
    root = g.start("edit")
    run(flow, g)

    assert root.final_result() == "base\nfirst\nsecond\n"
    assert (repo / "notes.txt").read_text() == "base\n"  # the checkout itself is untouched
    assert git(repo, "status", "--porcelain") == ""
    first, second, _ = code_nodes(g, root)
    assert str(workspaces.path(root)) in first.output
    assert git(repo, "show", f"{first.data['workspace']}:notes.txt") == "base\nfirst"
    assert git(repo, "show", f"{second.data['workspace']}:notes.txt") == "base\nfirst\nsecond"


def test_children_start_from_the_parents_files_in_their_own_worktrees(repo):
    def reply(messages):
        task = first_user(messages)
        if task == "lead":
            if "['base\\nlead\\nchild a\\n', 'base\\nlead\\nchild b\\n']" in trailing_user_text(
                messages
            ):
                return block("submit(open('notes.txt').read())")
            return block(
                "open('notes.txt', 'a').write('lead\\n')\n"
                "a = launch_agent('child a', config='worker')\n"
                "b = launch_agent('child b', config='worker')\n"
                "print(await asyncio.gather(a, b))"
            )
        return block(
            f"open('notes.txt', 'a').write({task + chr(10)!r})\nsubmit(open('notes.txt').read())"
        )

    g = Graph()
    root = g.start("lead")
    flow = ScriptedFlow(
        StubLLM(reply),
        configs=[AgentConfig(name="worker")],
        runtime=LocalRuntime(repl_timeout=10),
        workspaces=GitWorktrees(repo),
    )
    run(flow, g)

    assert root.final_result() == "base\nlead\n"
    assert [c.final_result() for c in root.children()] == [
        "base\nlead\nchild a\n",
        "base\nlead\nchild b\n",
    ]


def test_a_fork_rebuilds_its_repl_then_restores_the_files_at_the_fork_point(repo):
    def reply(messages):
        last = trailing_user_text(messages)
        if "resume" in last:
            return block("submit([count, open('notes.txt').read()])")
        if "one" in last:
            return block("count += 1\nopen('notes.txt', 'a').write('two\\n')\nprint('two')")
        if "two" in last:
            return block("submit('original')")
        return block("count = 1\nopen('notes.txt', 'a').write('one\\n')\nprint('one')")

    g = Graph()
    root = g.start("edit")
    run(
        ScriptedFlow(
            StubLLM(reply), runtime=LocalRuntime(repl_timeout=10), workspaces=GitWorktrees(repo)
        ),
        g,
    )
    assert root.final_result() == "original"

    second = code_nodes(g, root)[1]
    branch = g.fork(second)
    copy = branch.agent(root.id)
    branch.send(copy, "resume here")
    workspaces = GitWorktrees(repo)
    run(
        ScriptedFlow(StubLLM(reply), runtime=LocalRuntime(repl_timeout=10), workspaces=workspaces),
        branch,
    )

    # Replaying the first node appended to notes.txt again; the checkout undid
    # that before the second node ran for real.
    assert copy.final_result() == [2, "base\none\ntwo\n"]
    assert workspaces.path(copy) != workspaces.path(root)
    assert (workspaces.path(root) / "notes.txt").read_text() == "base\none\ntwo\n"


# -- Copies: the default when the runtime has a working directory ---------------


def copies_flow(reply, directory, **kwargs):
    return ScriptedFlow(
        StubLLM(reply),
        configs=[AgentConfig(name="worker")],
        runtime=LocalRuntime(directory, repl_timeout=10),
        **kwargs,
    )


def test_auto_gives_launched_agents_copies_and_leaves_the_directory_alone(tmp_path):
    (tmp_path / "notes.txt").write_text("base\n")

    def reply(messages):
        task = first_user(messages)
        if task == "lead":
            if "child:" in trailing_user_text(messages):
                return block("submit(open('notes.txt').read())")
            return block(
                "open('notes.txt', 'a').write('lead\\n')\n"
                "child = launch_agent('child', config='worker')\n"
                "print('child:', await child)"
            )
        return block("open('notes.txt', 'a').write('child\\n')\nsubmit(open('notes.txt').read())")

    flow = copies_flow(reply, tmp_path)
    assert isinstance(flow.workspaces, Copies)
    g = Graph()
    root = g.start("lead")
    run(flow, g)

    child = root.children()[0]
    assert child.final_result() == "base\nlead\nchild\n"
    assert root.final_result() == "base\nlead\n"
    assert (tmp_path / "notes.txt").read_text() == "base\nlead\n"
    assert not any(n.data.get("workspace") for n in root.nodes())
    commit = next(n.data["workspace"] for n in child.nodes() if n.data.get("workspace"))
    assert git(tmp_path, "show", f"{commit}:notes.txt", env=flow.workspaces._env()) == (
        "base\nlead\nchild"
    )
    assert (
        git(tmp_path, "ls-tree", "--name-only", commit, env=flow.workspaces._env()) == "notes.txt"
    )


def test_shared_and_commit_workspaces(tmp_path):
    (tmp_path / "notes.txt").write_text("base\n")

    def reply(messages):
        task = first_user(messages)
        if task == "lead":
            if "done" in trailing_user_text(messages):
                return block("submit(open('notes.txt').read())")
            return block(
                "a = launch_agent('a', config='worker')\n"
                "await a\n"
                "s = launch_agent('s', config='worker', workspace='shared')\n"
                "await s\n"
                "c = launch_agent('c', config='worker', workspace=a)\n"
                "await c\n"
                "checkout(c)\n"
                "print('done')"
            )
        return block(
            f"open('notes.txt', 'a').write({task + chr(10)!r})\nsubmit(open('notes.txt').read())"
        )

    g = Graph()
    root = g.start("lead")
    run(copies_flow(reply, tmp_path), g)

    a, s, c = root.children()
    assert a.final_result() == "base\na\n"
    assert s.final_result() == "base\ns\n"
    assert c.final_result() == "base\na\nc\n"
    # checkout into the directory overwrites, never deletes
    assert root.final_result() == "base\na\nc\n"


def test_shared_workspaces_never_copy(tmp_path):
    flow = ScriptedFlow(
        StubLLM(lambda m: block("submit(1)")),
        runtime=LocalRuntime(tmp_path),
        workspaces="shared",
    )
    assert isinstance(flow.workspaces, LocalWorkspace)
    assert isinstance(ScriptedFlow(StubLLM(lambda m: "")).workspaces, LocalWorkspace)


def test_big_and_ignored_files_are_not_copied(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored/\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "x").write_text("x")
    (tmp_path / "big.bin").write_bytes(b"0" * 2000)
    (tmp_path / "small.txt").write_text("s")
    workspaces = Copies(tmp_path, max_file_bytes=1000)
    commit = workspaces._snapshot_directory()
    files = git(tmp_path, "ls-tree", "-r", "--name-only", commit, env=workspaces._env())
    assert files.split() == [".gitignore", "small.txt"]
    assert workspaces.skipped == {"big.bin"}


def test_a_launch_can_start_from_an_agents_commit(tmp_path):
    (tmp_path / "notes.txt").write_text("base\n")

    def reply(messages):
        task, last = first_user(messages), trailing_user_text(messages)
        if task == "lead":
            if "second" in last:
                return block("submit(open('notes.txt').read())")
            if "first" in last:
                return block(
                    "c = launch_agent('c', config='worker', workspace=a.commit)\n"
                    "print('second', await c)"
                )
            return block("a = launch_agent('a', config='worker')\nprint('first', await a)")
        return block(
            f"open('notes.txt', 'a').write({task + chr(10)!r})\nsubmit(open('notes.txt').read())"
        )

    g = Graph()
    root = g.start("lead")
    run(copies_flow(reply, tmp_path), g)

    assert [c.final_result() for c in root.children()] == ["base\na\n", "base\na\nc\n"]
    assert root.final_result() == "base\n"
