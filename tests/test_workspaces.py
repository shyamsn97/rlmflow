"""Git worktrees: one directory per agent, snapshots on nodes, forks restore files."""

import asyncio
import shutil

import pytest
from helpers import StubLLM, first_user, trailing_user_text
from helpers import reasoning_repl as block

from rlmflow import CODE, AgentConfig, Flow, GitWorktrees, Graph
from rlmflow.engine.workspaces import git
from rlmflow.harnesses.repl.runtime import LocalRuntime

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


def test_cells_run_in_the_agents_worktree_and_carry_snapshots(repo):
    def reply(messages):
        last = trailing_user_text(messages)
        if "first" in last:
            return block("open('notes.txt', 'a').write('second\\n')\nprint('second')")
        if "second" in last:
            return block("submit(open('notes.txt').read())")
        return block("import os\nopen('notes.txt', 'a').write('first\\n')\nprint('first', os.getcwd())")

    workspaces = GitWorktrees(repo)
    flow = Flow(StubLLM(reply), runtime=LocalRuntime(repl_timeout=10), workspaces=workspaces)
    g = Graph()
    root = g.create_agent("edit")
    run(flow, g)

    assert g.result(root) == "base\nfirst\nsecond\n"
    assert (repo / "notes.txt").read_text() == "base\n"  # the checkout itself is untouched
    assert git(repo, "status", "--porcelain") == ""
    first, second = [n for n in g.history(root) if n.type == CODE]
    assert str(workspaces.path(root)) in first.data["output"]
    assert git(repo, "show", f"{first.data['workspace']}:notes.txt") == "base\nfirst"
    assert git(repo, "show", f"{second.data['workspace']}:notes.txt") == "base\nfirst\nsecond"


def test_children_start_from_the_parents_files_in_their_own_worktrees(repo):
    def reply(messages):
        task = first_user(messages)
        if task == "lead":
            if "['base\\nlead\\nchild a\\n', 'base\\nlead\\nchild b\\n']" in trailing_user_text(messages):
                return block("submit(open('notes.txt').read())")
            return block(
                "open('notes.txt', 'a').write('lead\\n')\n"
                "a = launch_agent('child a', config='worker')\n"
                "b = launch_agent('child b', config='worker')\n"
                "print(await asyncio.gather(a, b))"
            )
        return block(f"open('notes.txt', 'a').write({task + chr(10)!r})\nsubmit(open('notes.txt').read())")

    g = Graph()
    g.add_config(AgentConfig(name="worker"))
    root = g.create_agent("lead", config=AgentConfig(max_depth=1))
    run(Flow(StubLLM(reply), runtime=LocalRuntime(repl_timeout=10), workspaces=GitWorktrees(repo)), g)

    assert g.result(root) == "base\nlead\n"
    assert [g.result(c) for c in g.children(root)] == [
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
    root = g.create_agent("edit")
    run(Flow(StubLLM(reply), runtime=LocalRuntime(repl_timeout=10), workspaces=GitWorktrees(repo)), g)
    assert g.result(root) == "original"

    first = next(n for n in g.history(root) if n.type == CODE)
    fork = g.fork(first, name="fork")
    g.send(fork, "resume here")
    workspaces = GitWorktrees(repo)
    run(Flow(StubLLM(reply), runtime=LocalRuntime(repl_timeout=10), workspaces=workspaces), g)

    # The rebuilt cell appended to notes.txt again; the checkout undid that.
    assert g.result(fork) == [1, "base\none\n"]
    assert workspaces.path(fork) != workspaces.path(root)
    assert (workspaces.path(root) / "notes.txt").read_text() == "base\none\ntwo\n"
