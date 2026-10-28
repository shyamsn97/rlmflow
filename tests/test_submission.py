from helpers import ScriptedFlow, StubLLM, agent_graph, counting_replies, repl, run_result

from rlmflow import CODE, DONE, AgentConfig, LocalRuntime


def test_immediate_submission_adds_no_model_turn():
    llm = StubLLM(counting_replies(repl("submit('answer')")))
    flow = ScriptedFlow(llm, runtime=LocalRuntime(repl_timeout=5))
    g, root = agent_graph("Answer.")

    assert run_result(flow, g, root) == "answer"
    history = g.history(root)
    assert [node.type for node in history] == ["agent_start", "llm_output", "done"]
    assert history[-1].data["code"] == "submit('answer')"


def test_stdout_submission_is_accepted_without_an_extra_turn():
    llm = StubLLM(counting_replies(repl("print('count=4')\nsubmit('Answer: 4')")))
    flow = ScriptedFlow(llm, runtime=LocalRuntime(repl_timeout=5))
    g, root = agent_graph("Count the records.")

    assert run_result(flow, g, root) == "Answer: 4"
    assert sum(node.type == "llm_output" for node in g.history(root)) == 1
    assert g.frontier(root).data["output"] == "count=4"


def test_finish_alias_is_gone():
    flow = ScriptedFlow(StubLLM(counting_replies(repl("finish('legacy')"), repl("submit('ok')"))))
    g, root = agent_graph("Answer.")

    assert run_result(flow, g, root) == "ok"
    first = next(node for node in g.history(root) if node.type == CODE)
    assert "NameError" in first.data["output"] and first.data["error"] is True


def test_output_schema_failure_is_a_code_error_and_the_agent_retries():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    flow = ScriptedFlow(
        StubLLM(counting_replies(repl("submit({'nope': 1})"), repl("submit({'ok': True})")))
    )
    g, root = agent_graph("Return an object.")

    assert run_result(flow, g, root, root_config=AgentConfig(output_schema=schema)) == {"ok": True}
    rejected = next(node for node in g.history(root) if node.type == CODE)
    assert rejected.data["error"] is True and "ok" in rejected.data["output"]
    assert g.frontier(root).type == DONE


def test_output_schema_accepts_a_json_text_reply():
    flow = ScriptedFlow(StubLLM(counting_replies('{"n": 3}')))
    g, root = agent_graph("Count.")
    config = AgentConfig(output_schema={"type": "object", "properties": {"n": {"type": "integer"}}})

    assert run_result(flow, g, root, root_config=config) == {"n": 3}


def test_plain_text_reply_is_the_final_answer():
    flow = ScriptedFlow(StubLLM(counting_replies("The answer is 4.")))
    g, root = agent_graph("Answer.")

    assert run_result(flow, g, root) == "The answer is 4."
    assert [node.type for node in g.history(root)] == ["agent_start", "llm_output", "done"]
