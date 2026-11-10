from helpers import (
    ScriptedFlow,
    StubLLM,
    agent_graph,
    code_nodes,
    counting_replies,
    repl,
    run_result,
    turns,
)

from rlmflow import AgentConfig, LocalRuntime


def test_immediate_submission_is_one_turn():
    llm = StubLLM(counting_replies(repl("submit('answer')")))
    flow = ScriptedFlow(llm, runtime=LocalRuntime(repl_timeout=5))
    g, root = agent_graph("Answer.")

    assert run_result(flow, g, root) == "answer"
    [turn] = turns(g, root)
    assert root.nodes() == [root, turn]
    assert turn.code == "submit('answer')" and turn.data["terminal"]


def test_stdout_submission_is_accepted_without_an_extra_turn():
    llm = StubLLM(counting_replies(repl("print('count=4')\nsubmit('Answer: 4')")))
    flow = ScriptedFlow(llm, runtime=LocalRuntime(repl_timeout=5))
    g, root = agent_graph("Count the records.")

    assert run_result(flow, g, root) == "Answer: 4"
    [turn] = turns(g, root)
    assert turn.output == "count=4"


def test_finish_alias_is_gone():
    flow = ScriptedFlow(StubLLM(counting_replies(repl("finish('legacy')"), repl("submit('ok')"))))
    g, root = agent_graph("Answer.")

    assert run_result(flow, g, root) == "ok"
    first = code_nodes(g, root)[0]
    assert "NameError" in first.output and first.error is True


def test_output_schema_failure_is_a_code_error_and_the_agent_retries():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    flow = ScriptedFlow(
        StubLLM(counting_replies(repl("submit({'nope': 1})"), repl("submit({'ok': True})")))
    )
    g, root = agent_graph("Return an object.")

    assert run_result(flow, g, root, root_config=AgentConfig(output_schema=schema)) == {"ok": True}
    rejected = code_nodes(g, root)[0]
    assert rejected.error is True and "ok" in rejected.output
    assert root.nodes()[-1].data["terminal"]


def test_output_schema_accepts_a_json_text_reply():
    flow = ScriptedFlow(StubLLM(counting_replies('{"n": 3}')))
    g, root = agent_graph("Count.")
    config = AgentConfig(output_schema={"type": "object", "properties": {"n": {"type": "integer"}}})

    assert run_result(flow, g, root, root_config=config) == {"n": 3}


def test_plain_text_reply_is_the_final_answer():
    flow = ScriptedFlow(StubLLM(counting_replies("The answer is 4.")))
    g, root = agent_graph("Answer.")

    assert run_result(flow, g, root) == "The answer is 4."
    [turn] = turns(g, root)
    assert turn.text == "The answer is 4." and not turn.runs_code
