import json

from rflow import (
    Flow,
    Graph,
    LLMOutput,
)

from helpers import (
    StubLLM,
    first_user,
)


def test_minimal_root_output_schema_validates_done_and_prompts_schema():
    seen = {}
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }

    def reply(messages):
        seen["system"] = messages[0]["content"]
        return '```repl\ndone({"answer": 42})\n```'

    flow = Flow(StubLLM(reply))
    graph = Graph(query="q")

    assert json.loads(flow.run(graph=graph, output_schema=schema)) == {"answer": 42}
    assert graph.output_schema == schema
    assert "JSON Schema" in seen["system"]
    assert '"answer"' in seen["system"]



def test_minimal_output_schema_prompted_and_enforced_regardless_of_flag():
    # An explicit output_schema is a hard requirement, independent of the
    # enable_structured_output capability flag.
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    seen = {}

    def reply(messages):
        seen["system"] = messages[0]["content"]
        return '```repl\ndone({"answer": 5})\n```'

    flow = Flow(StubLLM(reply), enable_structured_output=False)
    graph = Graph(query="q", output_schema=schema)

    assert json.loads(flow.run(graph=graph)) == {"answer": 5}
    assert "This run requires structured output" in seen["system"]



def test_minimal_structured_output_option_teaches_subagent_schema_when_enabled():
    flow = Flow(StubLLM(lambda _messages: "unused"), max_depth=1)
    agent = Graph(query="q")
    prompt = flow.build_system_prompt(agent)
    assert "Structured Subagent Output" in prompt
    assert "output_schema" in prompt



def test_minimal_structured_output_option_absent_when_flag_disabled():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        max_depth=1,
        enable_structured_output=False,
    )
    agent = Graph(query="q")
    assert "Structured Subagent Output" not in flow.build_system_prompt(agent)



def test_minimal_structured_output_option_absent_without_subagents():
    flow = Flow(StubLLM(lambda _messages: "unused"), max_depth=0)
    agent = Graph(query="q")
    assert "Structured Subagent Output" not in flow.build_system_prompt(agent)



def test_minimal_first_turn_note_present_then_drops_after_output():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    agent = Graph(query="q")
    assert "First Turn" in flow.build_system_prompt(agent)
    agent.commit(LLMOutput(content="```repl\nprint(1)\n```", code="print(1)"))
    assert "First Turn" not in flow.build_system_prompt(agent)



def test_minimal_first_turn_note_mentions_inputs_when_present():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    with_inputs = Graph(query="q", inputs={"doc": "x"})
    assert "INPUTS" in flow.build_system_prompt(with_inputs)



def test_minimal_static_system_prompt_within_size_guard():
    from rflow.prompts import MAX_STATIC_PROMPT_CHARS, SYSTEM_PROMPT

    assert len(SYSTEM_PROMPT) <= MAX_STATIC_PROMPT_CHARS



def test_minimal_prompt_builder_names_lists_sections_in_order():
    from rflow.prompts import DEFAULT_BUILDER

    names = DEFAULT_BUILDER.names
    assert names[0] == "role"
    assert names[-1] == "first-turn"
    assert {"tools", "inputs", "status"} <= set(names)



def test_minimal_invalid_structured_done_records_error_then_recovers():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }
    replies = iter(
        [
            '```repl\ndone({"answer": "wrong"})\n```',
            '```repl\ndone({"answer": 7})\n```',
        ]
    )
    flow = Flow(StubLLM(lambda _messages: next(replies)), max_iters=2)
    graph = Graph(query="q", output_schema=schema)

    assert json.loads(flow.run(graph=graph)) == {"answer": 7}
    assert [node.type for node in graph.nodes] == [
        "user_query",
        "llm_output",
        "exec_action",
        "error_output",
        "llm_output",
        "exec_action",
        "done_output",
    ]
    assert "StructuredOutputError" in graph.nodes[3].output



def test_minimal_child_output_schema_returns_parsed_result_to_parent():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
    }

    def reply(messages):
        task = first_user(messages)
        if task == "child":
            return '```repl\ndone({"answer": 5})\n```'
        return (
            "```repl\n"
            f'results = await launch_subagents([{{"name": "child", "query": "child", "output_schema": {schema!r}}}])\n'
            'done(str(results[0]["answer"] + 1))\n'
            "```"
        )

    flow = Flow(StubLLM(reply), max_depth=1)
    graph = Graph(query="parent")

    assert flow.run(graph=graph) == "6"
    assert graph["root.child"].output_schema == schema



def test_minimal_uses_default_prompt_builder_and_inputs_namespace():
    seen = {}

    def reply(messages):
        seen["system"] = messages[0]["content"]
        return '```repl\nprint(INPUTS["x"])\ndone("ok")\n```'

    flow = Flow(StubLLM(reply))
    graph = Graph(query="q")

    assert flow.run(graph=graph, inputs={"x": "available"}) == "ok"
    assert "Recursive Coding Agent" in seen["system"]
    assert "Available in the REPL" in seen["system"]
    assert "Status" in seen["system"]
    assert "available" in graph.nodes[-1].output
    assert "[done] ok" in graph.nodes[-1].output

