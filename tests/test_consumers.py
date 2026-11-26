import asyncio

import pytest
from helpers import ScriptedFlow, agent_graph
from helpers import repl as block

from rlmflow import Checkpoint, Graph
from rlmflow.view import LiveTree


class Echo:
    def chat(self, messages):
        return block("submit('ok')")


def test_consumers_are_called_on_each_node_in_the_callers_loop(tmp_path):
    flow = ScriptedFlow(Echo())
    g, root = agent_graph("go")
    seen = []
    checkpoint = Checkpoint(tmp_path / "run")
    tree = LiveTree(rich=False, every_s=0)

    async def main():
        async for node in flow.stream(g):
            seen.append(node.id)
            checkpoint(node)
            tree(node)

    asyncio.run(main())
    checkpoint.close()
    tree.close()
    assert seen and root.final_result() == "ok"
    assert Graph.load(tmp_path / "run").agent_root(root.id).final_result() == "ok"


def test_hooks_get_every_node_and_the_end():
    flow = ScriptedFlow(Echo())
    g, root = agent_graph("go")
    tree = LiveTree(rich=False, every_s=0)
    flow.run(g, hooks=[tree])
    assert root.final_result() == "ok"
    assert "ok" in tree.render()


def test_a_checkpoint_refuses_a_second_graph(tmp_path):
    checkpoint = Checkpoint(tmp_path / "run")
    (_, ra), (_, rb) = agent_graph("a"), agent_graph("b")
    checkpoint(ra)
    with pytest.raises(ValueError, match="multiple graphs"):
        checkpoint(rb)
