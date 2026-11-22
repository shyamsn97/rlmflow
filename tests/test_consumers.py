import asyncio

import pytest
from helpers import ScriptedFlow, agent_graph
from helpers import repl as block

from rlmflow import Graph, Step
from rlmflow.consumers import Checkpoint, LiveTree


class Echo:
    def chat(self, messages):
        return block("submit('ok')")


def test_consumers_are_called_on_each_step_in_the_callers_loop(tmp_path):
    flow = ScriptedFlow(Echo())
    g, root = agent_graph("go")
    seen = []
    checkpoint = Checkpoint(tmp_path / "run")
    tree = LiveTree(rich=False, every_s=0)

    async def main():
        async for step in flow.run_streaming(g):
            seen.append(step.node.id)
            checkpoint(step)
            tree(step)

    asyncio.run(main())
    checkpoint.close()
    tree.close()
    assert seen and root.final_result() == "ok"
    assert Graph.load(tmp_path / "run").agent(root.id).final_result() == "ok"


def test_a_checkpoint_refuses_a_second_graph(tmp_path):
    checkpoint = Checkpoint(tmp_path / "run")
    (a, ra), (b, rb) = agent_graph("a"), agent_graph("b")
    checkpoint(Step.latest(a, ra))
    with pytest.raises(ValueError, match="multiple graphs"):
        checkpoint(Step.latest(b, rb))
