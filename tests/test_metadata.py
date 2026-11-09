import math

import pytest
from helpers import (
    ScriptedFlow,
    StubLLM,
    agent_graph,
    counting_replies,
    ran,
    repl,
    run_result,
    turns,
)

from rlmflow import (
    AgentConfig,
    Graph,
    LLMClient,
    LLMUsage,
    LocalRuntime,
    persistence,
    record_metadata,
)
from rlmflow.graph import calls
from rlmflow.tools import tool


def test_node_metadata_is_strict_durable_and_forkable(tmp_path):
    g, root = agent_graph("metadata")
    record_metadata("experiment", {"run": "abc", "seed": 7}, node=root)
    code_node = ran(output="ok")
    record_metadata("metrics", {"score": 1.0}, node=code_node)
    g.append(root, code_node)

    g.save(tmp_path / "graph")
    loaded = Graph.load(tmp_path / "graph")
    loaded_root = loaded.agent(root.id)
    forked = loaded.fork(loaded.tail(loaded_root))

    assert loaded_root.metadata == root.metadata == {"experiment": {"run": "abc", "seed": 7}}
    assert loaded_root.root_data.metadata == root.root_data.metadata
    assert loaded.tail(loaded_root).metadata == {"metrics": {"score": 1.0}}
    assert loaded.history(forked)[-2].metadata == {"metrics": {"score": 1.0}}
    assert persistence.node_record(root)["metadata"] == root.metadata
    assert persistence.node_record(code_node)["metadata"] == {"metrics": {"score": 1.0}}

    with pytest.raises(TypeError, match="strict JSON"):
        record_metadata("bad", {"nan": math.nan}, node=root)


def test_custom_proxy_tool_can_annotate_its_code():
    @tool("Tag this execution.", cache=True)
    async def tag(value: str) -> str:
        record_metadata("custom.tags", {"value": value}, append=True)
        return value.upper()

    flow = ScriptedFlow(
        StubLLM(counting_replies(repl("print(await tag('alpha'))\nsubmit('done')"))),
        runtime=LocalRuntime(repl_timeout=5),
        tools=[tag],
    )
    g = Graph()
    root = g.start("tag it")

    assert run_result(flow, g, root) == "done"
    [code_node] = turns(g, root)
    assert code_node.code.startswith("print(await tag")
    assert code_node.metadata["custom.tags"] == [{"value": "alpha"}]
    assert code_node.calls == [
        {"tool": "tag", "key": calls.key("tag", ["alpha"], {}), "value": "ALPHA"}
    ]


class QueryLLM(LLMClient):
    def __init__(self):
        self.calls = 0
        self.last_usage = LLMUsage(input_tokens=3, output_tokens=2)

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        if len(messages) == 1 and messages[0]["content"] == "ask me":
            return "an answer"
        return repl("first = await llm_query('ask me')\nsubmit(first)")


def test_llm_query_calls_are_recorded_on_the_code():
    llm = QueryLLM()
    flow = ScriptedFlow(llm, runtime=LocalRuntime(repl_timeout=5), configs=[AgentConfig(model="m")])
    g = Graph()
    root = g.start("use llm_query")

    assert run_result(flow, g, root) == "an answer"
    assert llm.calls == 2
    [code_node] = turns(g, root)
    [call] = code_node.metadata["llm_query.calls"]
    assert call["model"] == "m" and call["usage"]["input_tokens"] == 3
    assert code_node.calls == [
        {
            "tool": "llm_query",
            "key": calls.key("llm_query", "ask me", "m", None, None, None),
            "value": "an answer",
        }
    ]


def test_a_model_call_records_usage_model_and_timing_on_its_node():
    flow = ScriptedFlow(
        StubLLM(counting_replies(repl("submit('ok')"))),
        runtime=LocalRuntime(repl_timeout=5),
        configs=[AgentConfig(model="m")],
    )
    g = Graph()
    root = g.start("answer")

    flow.run(g)
    [reply] = g.history(root)[1:]
    assert (reply.role, reply.code, reply.data["terminal"]) == ("assistant", "submit('ok')", True)
    assert reply.data["usage"]["input_tokens"] >= 0
    assert reply.data["model"] == "m"
    assert reply.finished_at >= reply.started_at
