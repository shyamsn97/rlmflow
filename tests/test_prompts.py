from helpers import StubLLM
from pydantic import BaseModel

from rlmflow import ErrorOutput, ExecOutput, Flow, LLMOutput, start


def test_messages_are_a_pure_projection_of_the_transcript():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query")
    root.append(LLMOutput(content="assistant", code="print('x')")).append(
        ExecOutput(content="observation")
    )
    before = [node.id for node in root.transcript()]

    messages = flow.messages(root.frontier)

    assert [node.id for node in root.transcript()] == before
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"] == "observation"


def test_errors_are_projected_as_user_observations():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query")
    root.append(ErrorOutput(content="NameError: missing"))

    assert flow.messages(root.frontier)[-1]["role"] == "user"
    assert flow.messages(root.frontier)[-1]["content"].endswith("NameError: missing")


def test_explicit_output_schema_is_in_system_prompt():
    class Answer(BaseModel):
        value: str

    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query", output_schema=Answer.model_json_schema())

    assert '"value"' in flow.messages(root.frontier)[0]["content"]
