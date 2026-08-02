import asyncio

from helpers import StubLLM

from rlmflow import Flow, start


def test_step_advances_exactly_one_state_transition():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    root = start("query")

    transition = asyncio.run(flow.step(root))
    produced = transition.created

    assert produced.type == "llm_output"
    assert transition.submitted is root
    assert not transition.is_agent_start
    assert transition.error is None
    assert root.frontier is produced
    assert root.transcript() == [root, produced]


def test_a_run_can_be_driven_by_hand_one_step_at_a_time():
    flow = Flow(StubLLM(lambda _messages: '```repl\ndone("ok")\n```'))
    root = start("query")

    produced = [asyncio.run(flow.step(root.frontier)).created for _ in range(3)]

    assert [node.type for node in produced] == ["llm_output", "exec_action", "done_output"]
    assert root.terminal and root.result() == "ok"
