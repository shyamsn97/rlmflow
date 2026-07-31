import asyncio

import pytest
from helpers import StubLLM

from rlmflow import ExecOutput, Flow, start


def test_step_advances_exactly_one_state_transition():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    root = start("query")

    produced = asyncio.run(flow.step(root))

    assert produced.type == "llm_output"
    assert root.steps == [produced]


def test_step_rejects_a_non_tail_node():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    root = start("query")
    root.submit(ExecOutput(content="later"))

    with pytest.raises(ValueError, match="current tail"):
        asyncio.run(flow.step(root))
