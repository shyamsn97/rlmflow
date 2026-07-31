from helpers import StubLLM
from pydantic import BaseModel

from rlmflow import ErrorOutput, ExecOutput, Flow, LLMOutput, start


def test_messages_are_a_pure_projection_of_agent_steps():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    agent = start("query")
    agent.submit(LLMOutput(content="assistant", code="print('x')")).submit(
        ExecOutput(content="observation")
    )
    before = list(agent.steps)

    messages = flow.messages(agent)

    assert agent.steps == before
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"] == "observation"


def test_errors_are_projected_as_user_observations():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    agent = start("query")
    agent.submit(ErrorOutput(content="NameError: missing"))

    assert flow.messages(agent)[-1]["role"] == "user"
    assert flow.messages(agent)[-1]["content"].endswith("NameError: missing")


def test_explicit_output_schema_is_in_system_prompt():
    class Answer(BaseModel):
        value: str

    flow = Flow(StubLLM(lambda _messages: "unused"))
    agent = start("query", output_schema=Answer.model_json_schema())

    assert '"value"' in flow.messages(agent)[0]["content"]


def test_inputs_manifest_lists_names_without_dumping_values():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    agent = start("query", inputs={"document": "SECRET_VALUE"})

    system = flow.messages(agent)[0]["content"]

    assert "document" in system
    assert "SECRET_VALUE" not in system
