import math

import pytest
from helpers import ScriptedFlow, StubLLM, agent_graph, counting_replies, repl, run_result

from rlmflow import (
    CODE,
    LLM,
    AgentConfig,
    Graph,
    LLMClient,
    LLMUsage,
    LocalRuntime,
    Node,
    persistence,
    record_metadata,
)
from rlmflow.tools import tool


def test_node_metadata_is_strict_durable_and_forkable(tmp_path):
    g, root = agent_graph("metadata")
    record_metadata("experiment", {"run": "abc", "seed": 7}, node=root)
    cell = Node(CODE, code="", output="ok")
    record_metadata("metrics", {"score": 1.0}, node=cell)
    g.append(root, cell)

    g.save(tmp_path / "graph")
    loaded = Graph.load(tmp_path / "graph")
    loaded_root = loaded.agent(root.id)
    forked = loaded.fork(loaded.history(loaded_root)[-1])
    record = persistence.agent_record(root)

    assert loaded_root.metadata["experiment"] == {"run": "abc", "seed": 7}
    assert loaded_root.metadata == root.metadata
    assert loaded.history(loaded_root)[-1].metadata == {"metrics": {"score": 1.0}}
    assert loaded.history(forked)[-1].metadata == {"metrics": {"score": 1.0}}
    assert record["metadata"] == root.metadata
    assert persistence.node_record(cell)["metadata"] == {"metrics": {"score": 1.0}}

    with pytest.raises(TypeError, match="strict JSON"):
        record_metadata("bad", {"nan": math.nan}, node=root)


def test_custom_proxy_tool_can_annotate_its_cell():
    @tool("Tag this execution.")
    async def tag(value: str) -> str:
        record_metadata("custom.tags", {"value": value}, append=True)
        return value.upper()

    flow = ScriptedFlow(
        StubLLM(counting_replies(repl("print(await tag('alpha'))\nsubmit('done')"))),
        runtime=LocalRuntime(repl_timeout=5),
        tools=[tag],
    )
    g = Graph()
    root = g.add_agent("tag it")

    assert run_result(flow, g, root) == "done"
    cell = g.history(root)[-2]
    assert cell.data["code"].startswith("print(await tag")
    assert cell.metadata["custom.tags"] == [{"value": "alpha"}]
    assert cell.data["calls"] == [{"tool": "tag", "value": "ALPHA"}]


class QueryLLM(LLMClient):
    def __init__(self):
        self.calls = 0
        self.last_usage = LLMUsage(input_tokens=3, output_tokens=2)

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        if len(messages) == 1 and messages[0]["content"] == "ask me":
            return "an answer"
        return repl("first = await llm_query('ask me')\nsubmit(first)")


def test_llm_query_calls_are_recorded_on_the_cell():
    llm = QueryLLM()
    flow = ScriptedFlow(llm, runtime=LocalRuntime(repl_timeout=5), configs=[AgentConfig(model="m")])
    g = Graph()
    root = g.add_agent("use llm_query")

    assert run_result(flow, g, root) == "an answer"
    assert llm.calls == 2
    cell = g.history(root)[-2]
    [call] = cell.metadata["llm_query.calls"]
    assert call["model"] == "m" and call["usage"]["input_tokens"] == 3
    assert cell.data["calls"] == [{"tool": "llm_query", "value": "an answer"}]


def test_llm_node_records_usage_model_and_timing():
    flow = ScriptedFlow(
        StubLLM(counting_replies(repl("submit('ok')"))),
        runtime=LocalRuntime(repl_timeout=5),
        configs=[AgentConfig(model="m")],
    )
    g = Graph()
    root = g.add_agent("answer")

    flow.run(g)
    assert [node.type for node in g.history(root)] == [LLM, CODE, "done"]
    llm_node = next(node for node in g.history(root) if node.type == LLM)

    assert llm_node.data["usage"]["input_tokens"] >= 0
    assert llm_node.data["model"] == "m"
    assert llm_node.finished_at >= llm_node.started_at
