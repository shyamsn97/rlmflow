import inspect
from dataclasses import dataclass

import pytest
from helpers import StubLLM

from rlmflow import (
    ErrorOutput,
    ExecAction,
    ExecOutput,
    FinalQuery,
    Flow,
    Input,
    LLMOutput,
    PinnedPromptSectionError,
    Prompt,
    PromptProfile,
    UserQuery,
    start,
    tool,
)


def test_prompt_orders_updates_and_removes_named_sections():
    prompt = Prompt()
    prompt.add_section("later", "later", order=20, source="base")
    prompt.add_section("first", "first", order=10, source="base")
    prompt.add_section("peer", "peer", order=20, source="base")

    prompt.update_section("later", "updated", source="harness")

    assert [section.key for section in prompt.sections()] == ["first", "later", "peer"]
    assert prompt.render() == "first\n\nupdated\n\npeer"
    assert prompt.get("later").source == "harness"
    assert prompt.remove_section("peer").content == "peer"
    assert prompt.get("peer") is None


def test_pinned_sections_are_idempotent_but_cannot_be_changed_or_removed():
    prompt = Prompt().add_section("protocol", "fixed", pinned=True, source="base")

    prompt.add_section("protocol", "fixed", pinned=True, source="base")
    with pytest.raises(PinnedPromptSectionError, match="replace pinned.*protocol"):
        prompt.add_section("protocol", "changed", source="harness")
    with pytest.raises(PinnedPromptSectionError, match="remove pinned.*protocol"):
        prompt.remove_section("protocol")


def test_system_and_user_prompts_use_the_same_prompt_class():
    flow = Flow(StubLLM(lambda _messages: "done"))
    root = start(
        "query",
        args={"document": "large input"},
        max_depth=1,
        max_iters=6,
    )

    system = flow.build_system_prompt(root)
    user = flow.default_user_prompt(root)

    assert isinstance(system, Prompt)
    assert isinstance(user, Prompt)
    assert system.get("protocol.core").pinned
    assert system.get("policy.execution").content
    assert system.get("context.arguments").content
    assert system.get("capabilities.depth").content
    assert "Plan the immediate next action" in system.get("protocol.core").content
    assert system.get("policy.evidence").content
    assert user.get("turn.instruction").content == "query"


def test_first_turn_prompt_keeps_one_task_message_and_offers_submit():
    flow = Flow(StubLLM(lambda _messages: "done"))
    root = start("query", args={"document": "large input"}, max_iters=4)

    messages = flow.build_messages(root)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"] == "query"
    assert "Compute answers from your inputs" in messages[0]["content"]
    assert "submit" in flow.build_tools(root)


def test_input_description_appears_without_copying_the_value_into_the_prompt():
    root = start(
        "query",
        args={
            "rows": Input(
                [{"private": "secret-value"}],
                description="Parsed records with user_id and label fields.",
            )
        },
    )
    flow = Flow(StubLLM(lambda _messages: "done"))

    manifest = flow.build_system_prompt(root).get("context.arguments").content

    assert root.resolved_args() == {"rows": [{"private": "secret-value"}]}
    assert "list, 1 items" in manifest
    assert "Parsed records with user_id and label fields." in manifest
    assert "secret-value" not in manifest


def test_prompt_documents_tools_added_by_a_flow_subclass():
    @tool("Inspect the host-owned state.", proxy=True)
    def inspect_state(key: str) -> dict[str, object]:
        return {"key": key}

    class DynamicToolFlow(Flow):
        def build_tools(self, node):
            return {**super().build_tools(node), "inspect_state": inspect_state}

    flow = DynamicToolFlow(StubLLM(lambda _messages: "done"))
    prompt = flow.build_system_prompt(start("query")).render()

    assert (
        "`inspect_state(key: str) -> dict[str, object]`: Inspect the host-owned state."
        in prompt
    )


def test_default_prompts_are_fresh_and_node_hooks_have_explicit_names():
    flow = Flow(StubLLM(lambda _messages: "done"))
    root = start("query")

    first_system = flow.default_system_prompt(root)
    second_system = flow.default_system_prompt(root)
    first_user = flow.default_user_prompt(root)
    second_user = flow.default_user_prompt(root)

    assert first_system is not second_system
    assert first_user is not second_user
    assert list(inspect.signature(UserQuery.build_system_prompt).parameters) == [
        "self",
        "system_prompt",
        "flow",
    ]
    assert list(inspect.signature(UserQuery.build_messages).parameters) == [
        "self",
        "user_prompt",
        "flow",
    ]


def test_node_customizes_prepared_system_prompt():
    @dataclass
    class ReviewQuery(UserQuery):
        def build_system_prompt(self, system_prompt, flow):
            system_prompt.add_section(
                "review.policy",
                "Check compatibility.",
                order=800,
                source=type(self).__name__,
            )
            return system_prompt

    root = start("query")
    review = ReviewQuery(content="review this")
    root.append(review)

    prompt = Flow(StubLLM(lambda _messages: "done")).build_system_prompt(review)

    assert prompt.get("protocol.core").pinned
    assert prompt.get("review.policy").content == "Check compatibility."


def test_node_cannot_return_a_prompt_that_drops_pinned_system_sections():
    @dataclass
    class UnsafeQuery(UserQuery):
        def build_system_prompt(self, system_prompt, flow):
            return Prompt().add_section("replacement", "unsafe")

    root = start("query")
    query = UnsafeQuery(content="replace policy")
    root.append(query)

    with pytest.raises(PinnedPromptSectionError, match="dropped or replaced.*protocol.core"):
        Flow(StubLLM(lambda _messages: "done")).build_system_prompt(query)


def test_flow_and_profile_modifiers_update_system_prompt_by_key():
    def global_memory(prompt, _flow, _node):
        prompt.add_section("memory.playbook", "global lesson", order=800)

    def research_memory(prompt, _flow, _node):
        prompt.add_section("memory.playbook", "research lesson", order=800)
        prompt.add_section("profile.research", "cite sources", order=810)

    flow = Flow(
        StubLLM(lambda _messages: "done"),
        system_prompt_modifiers=global_memory,
        prompt_profiles={
            "research": PromptProfile(
                system_modifiers=research_memory,
            )
        },
    )
    root = start("query", prompt_profile="research")

    prompt = flow.build_system_prompt(root)

    assert prompt.get("memory.playbook").content == "research lesson"
    assert "global lesson" not in prompt.render()
    assert prompt.render().endswith("research lesson\n\ncite sources")


def test_live_modifier_is_resolved_fresh_each_turn():
    state = {"lesson": "first"}

    def live_memory(prompt, _flow, _node):
        prompt.add_section("memory.live", state["lesson"], order=800)

    flow = Flow(StubLLM(lambda _messages: "done"), system_prompt_modifiers=live_memory)
    root = start("query")

    assert flow.build_system_prompt(root).get("memory.live").content == "first"
    state["lesson"] = "second"
    assert flow.build_system_prompt(root).get("memory.live").content == "second"


def test_host_owned_dict_is_an_ace_style_live_store():
    memory = {"lesson": "old"}

    def apply_memory(prompt, _flow, _node):
        for key, content in memory.items():
            prompt.add_section(f"memory.{key}", content, order=800)

    flow = Flow(StubLLM(lambda _messages: "done"), system_prompt_modifiers=apply_memory)
    root = start("query")

    assert flow.build_system_prompt(root).render().endswith("old")
    memory["lesson"] = "new"
    assert flow.build_system_prompt(root).render().endswith("new")
    del memory["lesson"]
    assert not flow.build_system_prompt(root).render().endswith("new")


def test_whole_system_sources_and_prompt_instances_remain_supported():
    root = start("query")
    callable_source = Flow(
        StubLLM(lambda _messages: "done"),
        system_prompt=lambda _flow, _node: "callable protocol",
    )

    def add_extra(prompt, _flow, _node):
        prompt.add_section("owned.extra", "extra", order=10)

    replaced = Flow(
        StubLLM(lambda _messages: "done"),
        system_prompt="owned protocol",
        system_prompt_modifiers=add_extra,
    )

    assert callable_source.build_system_prompt(root).render() == "callable protocol"
    assert replaced.build_system_prompt(root).render() == "owned protocol\n\nextra"


def test_query_nodes_modify_one_user_instruction_without_changing_system_prompt():
    flow = Flow(StubLLM(lambda _messages: "done"))
    root = start("query", max_iters=4)
    plan = UserQuery(content="Continue the task.")
    root.append(plan)
    final = FinalQuery()

    plan_system = flow.build_system_prompt(plan).render()
    plan_prompt = flow.default_user_prompt(plan)
    root.frontier.append(final)
    final_system = flow.build_system_prompt(final).render()
    final_prompt = flow.default_user_prompt(final)

    assert [section.key for section in plan_prompt] == ["turn.instruction"]
    assert plan_prompt.render() == "Continue the task."
    assert [section.key for section in final_prompt] == ["turn.instruction"]
    assert final_prompt.render() == final.content
    assert plan_system != final_system  # final mode hides action-only tools
    assert "protocol.core" in flow.build_system_prompt(plan)


def test_host_can_update_turn_instruction_through_user_modifier():
    def replace_instruction(prompt, _flow, _node):
        prompt.update_section("turn.instruction", "Use the host planning policy.")

    flow = Flow(
        StubLLM(lambda _messages: "done"),
        user_prompt_modifiers=replace_instruction,
    )
    root = start("query")
    plan = UserQuery(content="Continue the task.")
    root.append(plan)

    messages = flow.build_messages(plan)

    assert messages[-1]["content"].count("Use the host planning policy.") == 1
    assert "Continue the task" not in messages[-1]["content"]


def test_custom_query_modifies_its_user_prompt_directly():
    @dataclass
    class ReviewQuery(UserQuery):
        def build_messages(self, user_prompt, flow):
            user_prompt.add_section(
                "review.policy",
                "Review carefully.",
                order=1010,
                source=type(self).__name__,
            )
            return super().build_messages(user_prompt, flow)

    root = start("query")
    review = ReviewQuery(content="review this")
    root.append(review)
    flow = Flow(StubLLM(lambda _messages: "done"))

    messages = flow.build_messages(review)
    assert messages[-1]["content"] == "review this\n\nReview carefully."


def test_custom_query_can_emit_multiple_current_turn_messages():
    builds = 0

    @dataclass
    class BoardQuery(UserQuery):
        def build_messages(self, user_prompt, flow):
            nonlocal builds
            builds += 1
            return [
                {"role": "user", "content": "current board"},
                {"role": "user", "content": str(user_prompt)},
            ]

    root = start("query")
    query = BoardQuery(content="choose a move")
    root.append(query)

    flow = Flow(StubLLM(lambda _messages: "done"))
    messages = flow.build_messages(query)
    again = flow.build_messages(query)

    assert messages[-2:] == [
        {"role": "user", "content": "current board"},
        {"role": "user", "content": "choose a move"},
    ]
    assert again == messages
    assert builds == 1

    reply = LLMOutput(content="move left")
    query.append(reply)
    followup = UserQuery(content="continue")
    reply.append(followup)
    later = flow.build_messages(followup)
    assert later[-4:-1] == [
        {"role": "user", "content": "current board"},
        {"role": "user", "content": "choose a move"},
        {"role": "assistant", "content": "move left"},
    ]


def test_message_snapshots_are_immutable_and_provider_copies_are_independent():
    query = UserQuery(content="review")
    query.snapshot_messages([{"role": "user", "content": "stored"}])

    copied = query.message_dicts()
    copied[0]["content"] = "changed"

    assert query.message_dicts() == [{"role": "user", "content": "stored"}]
    with pytest.raises(RuntimeError, match="already has a different"):
        query.snapshot_messages([{"role": "user", "content": "replacement"}])


def test_engine_nodes_initialize_their_durable_message_snapshots():
    assert LLMOutput(content="answer").message_dicts() == [
        {"role": "assistant", "content": "answer"}
    ]
    assert ExecOutput(content="stdout").message_dicts() == [
        {"role": "user", "content": "stdout"}
    ]
    assert ErrorOutput(content="failure").message_dicts() == [
        {"role": "user", "content": "failure"}
    ]
    assert ExecAction(code="print('work')").message_dicts() == []


def test_build_messages_rejects_malformed_custom_messages():
    @dataclass
    class BrokenQuery(UserQuery):
        def build_messages(self, user_prompt, flow):
            return [{"role": "user"}]

    root = start("query")
    broken = BrokenQuery(content="broken")
    root.append(broken)

    with pytest.raises(ValueError, match="exactly 'role' and 'content'"):
        Flow(StubLLM(lambda _messages: "done")).build_messages(broken)


def test_prompt_string_uses_rendered_sections():
    prompt = Prompt().add_section("a", "first", order=10)
    prompt.add_section("b", "second", order=20)

    assert str(prompt) == prompt.render() == "first\n\nsecond"
