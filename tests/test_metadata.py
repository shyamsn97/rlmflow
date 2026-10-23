import math

import pytest
from helpers import StubLLM, agent_graph, counting_replies, repl, run_result

from rlmflow import (
    ExecAction,
    ExecOutput,
    Flow,
    Graph,
    LLMClient,
    LLMUsage,
    LocalRuntime,
    ops,
    persistence,
    reads,
    record_metadata,
)
from rlmflow.tools import tool


def test_node_metadata_is_strict_durable_and_forkable(tmp_path):
    g, root = agent_graph("metadata")
    root.annotate("experiment", {"run": "abc", "seed": 7})
    output = ExecOutput(content="ok", metadata={"metrics": {"score": 1.0}})
    ops.append(g, root, output)

    g.save(tmp_path / "graph")
    loaded = Graph.load(tmp_path / "graph")
    loaded_root = loaded.node(root.id)
    forked = ops.fork(loaded, reads.frontier(loaded, loaded_root))
    record = persistence.node_record(root)

    assert loaded_root.metadata == {"experiment": {"run": "abc", "seed": 7}}
    assert reads.frontier(loaded, loaded_root).metadata == {"metrics": {"score": 1.0}}
    assert reads.frontier(loaded, forked).metadata == {"metrics": {"score": 1.0}}
    assert record["metadata"]["extensions"] == root.metadata

    with pytest.raises(TypeError, match="strict JSON"):
        root.annotate("bad", {"nan": math.nan})


def test_custom_proxy_tool_can_annotate_its_active_action():
    @tool("Tag this execution.", proxy=True)
    async def tag(value: str) -> str:
        record_metadata("custom.tags", {"value": value}, append=True)
        return value.upper()

    flow = Flow(
        StubLLM(counting_replies(repl("print(await tag('alpha'))\nsubmit('done')"))),
        runtime=LocalRuntime(repl_timeout=5),
        tools=[tag],
    )
    g = Graph()
    root = g.create_agent("tag it")

    assert run_result(flow, g, root, close_repls=True) == "done"
    action = next(node for node in reads.history(g, root) if isinstance(node, ExecAction))
    assert action.metadata["custom.tags"] == [{"value": "alpha"}]
    receipts = action.metadata["tools.calls"]
    assert any(item["name"] == "tag" and item["status"] == "ok" for item in receipts)


class CacheLLM(LLMClient):
    def __init__(self):
        self.calls = 0
        self.last_usage = LLMUsage(input_tokens=3, output_tokens=2)

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        if len(messages) == 1 and messages[0]["content"] == "cache me":
            return "cached answer"
        return repl(
            "first = await llm_query('cache me')\n"
            "second = await llm_query('cache me')\n"
            "submit([first, second])"
        )


def test_llm_query_cache_is_opt_in_and_observable():
    llm = CacheLLM()
    flow = Flow(
        llm,
        runtime=LocalRuntime(repl_timeout=5),
        use_llm_query=True,
        llm_query_cache=True,
    )
    g = Graph()
    root = g.create_agent("use the cache")

    assert run_result(flow, g, root, close_repls=True) == ["cached answer", "cached answer"]
    assert llm.calls == 2  # one agent turn plus one unique llm_query
    action = next(node for node in reads.history(g, root) if isinstance(node, ExecAction))
    calls = action.metadata["llm_query.calls"]
    assert [call["cache_hit"] for call in calls] == [False, True]
    assert calls[0]["cache_key"] == calls[1]["cache_key"]


def test_llm_turn_metadata_exposes_usage_and_timing():
    flow = Flow(
        StubLLM(counting_replies(repl("submit('ok')"))),
        runtime=LocalRuntime(repl_timeout=5),
    )
    g = Graph()
    root = g.create_agent("answer")

    flow.run(g, close_repls=True)
    llm_node = next(node for node in reads.history(g, root) if node.type == "llm_output")

    assert llm_node.metadata["llm"]["usage"]["input_tokens"] >= 0
    assert llm_node.metadata["timing"]["duration_ms"] >= 0
