import asyncio

import pytest
from helpers import StubLLM

from rlmflow import (
    AgentDirectory,
    AgentStart,
    DoneOutput,
    ExecAction,
    Flow,
    start,
)
from rlmflow.tools.agents import build_agent_directory


def agent_tree():
    root = start("root")
    root_action = ExecAction(code="launch")
    root.append(root_action)
    researcher = AgentStart(
        content="research",
        config=root.config.child("researcher"),
    )
    root_action.append(researcher)
    reviewer = AgentStart(
        content="review",
        config=root.config.child("reviewer"),
    )
    root_action.append(reviewer)
    researcher.append(DoneOutput(result={"finding": "duplicate keys"}))

    reviewer_action = ExecAction(code="launch child")
    reviewer.append(reviewer_action)
    checker = AgentStart(
        content="check",
        config=reviewer.config.child("checker"),
    )
    reviewer_action.append(checker)
    return root, root_action, researcher, reviewer, checker


def test_directory_queries_the_recursive_tree():
    root, root_action, researcher, reviewer, checker = agent_tree()

    agents = build_agent_directory(
        reviewer,
        running_nodes=(root_action, reviewer),
    )

    assert agents.self.name == "reviewer"
    assert agents.get() is agents.self
    assert agents.get("root.researcher") is agents.get(researcher.id)
    assert agents.get("researcher") is agents.get_siblings()[0]
    assert agents.get_parent().agent_id == root.id
    assert [agent.name for agent in agents.get_children()] == ["checker"]
    assert [agent.name for agent in agents.get_siblings()] == ["researcher"]
    assert [agent.name for agent in agents.all()] == [
        "root",
        "researcher",
        "reviewer",
        "checker",
    ]
    assert agents.get_children("root") == [
        agents.get("researcher"),
        agents.get("reviewer"),
    ]
    assert agents.get_parent(agents.get(checker.id)).name == "reviewer"


def test_directory_reports_statuses_and_completed_results():
    _root, root_action, researcher, reviewer, checker = agent_tree()

    agents = build_agent_directory(
        reviewer,
        running_nodes=(root_action, reviewer),
    )

    assert agents.get("root").status == "waiting"
    assert agents.get(researcher.id).status == "completed"
    assert agents.get(researcher.id).result() == {"finding": "duplicate keys"}
    assert not hasattr(agents.get(researcher.id), "get_result")
    assert agents.get(reviewer.id).status == "running"
    assert agents.get(checker.id).status == "idle"


def test_directory_exposes_dependency_edges_and_blocked_status():
    root = start("root")
    action = ExecAction(code="launch")
    root.append(action)
    source = AgentStart(content="source", config=root.config.child("source"))
    action.append(source)
    consumer = AgentStart(
        content="consumer",
        config=root.config.child(
            "consumer",
            inputs={"evidence": {"$rlmflow_agent": source.id}},
        ),
    )
    action.append(consumer)

    blocked = build_agent_directory(root)
    assert blocked.get("consumer").status == "blocked"
    assert blocked.get("consumer").dependency_paths == ("root.source",)
    assert blocked.get("consumer").blocked_on_paths == ("root.source",)
    assert blocked.get("source").dependent_paths == ("root.consumer",)
    assert "consumer [blocked]" in blocked.render_graph()
    assert blocked.render_graph(view="dependencies") == ("consumer [blocked]\n└── source [idle]")

    source.append(DoneOutput(result="ready"))
    ready = build_agent_directory(root)
    assert ready.get("consumer").status == "idle"
    assert ready.get("consumer").blocked_on_paths == ()


def test_directory_is_a_snapshot_and_normalizes_results():
    class Answer:
        def __str__(self):
            return "custom answer"

    root = start("root")
    action = ExecAction(code="launch")
    root.append(action)
    child = AgentStart(
        content="work",
        config=root.config.child("worker"),
    )
    action.append(child)
    before = build_agent_directory(root)

    child.append(DoneOutput(result=Answer()))
    after = build_agent_directory(root)

    assert before.get("worker").status == "idle"
    with pytest.raises(asyncio.InvalidStateError):
        before.get("worker").result()
    assert after.get("worker").status == "completed"
    assert after.get("worker").result() == "custom answer"


def test_directory_renders_a_bounded_tree(capsys):
    _root, root_action, _researcher, reviewer, _checker = agent_tree()
    agents = build_agent_directory(
        reviewer,
        running_nodes=(root_action, reviewer),
    )

    rendered = agents.render_graph(show_results=True, max_result_chars=18)
    assert rendered.splitlines() == [
        "root [waiting]",
        '├── researcher [completed] -> {"finding":"dup...',
        "└── reviewer [running] (you)",
        "    └── checker [idle]",
    ]

    agents.print_graph()
    assert capsys.readouterr().out == (
        "root [waiting]\n"
        "├── researcher [completed]\n"
        "└── reviewer [running] (you)\n"
        "    └── checker [idle]\n"
    )


def test_directory_rejects_invalid_queries():
    root, _root_action, _researcher, reviewer, _checker = agent_tree()
    agents = build_agent_directory(reviewer)

    assert agents.get("missing") is None
    with pytest.raises(KeyError, match="unknown agent"):
        agents.get_children("missing")
    with pytest.raises(TypeError, match="selector"):
        agents.get(123)
    with pytest.raises(ValueError, match="max_result_chars"):
        agents.render_graph(max_result_chars=0)
    with pytest.raises(ValueError, match="view"):
        agents.render_graph(view="invalid")
    with pytest.raises(ValueError, match="viewer"):
        AgentDirectory(viewer_id="missing", root_id=root.id, agents=agents.agents)


def test_agent_tree_is_opt_in_for_namespace_and_prompt():
    root = start("query")
    disabled = Flow(StubLLM(lambda _messages: "unused"))
    enabled = Flow(StubLLM(lambda _messages: "unused"), use_agent_tree=True)

    assert "AGENTS" not in disabled.build_tools(root)
    assert "`AGENTS` is a read-only snapshot" not in disabled.build_messages(root)[0]["content"]

    directory = enabled.build_tools(root)["AGENTS"]
    assert isinstance(directory, AgentDirectory)
    assert directory.self.agent_id == root.id
    prompt = enabled.build_messages(root)[0]["content"]
    assert "`AGENTS` is a read-only snapshot" in prompt
    assert "`await agent.wait_for_result()` waits automatically" in prompt
    assert "`AGENTS.print_graph(show_results=True)`" in prompt


def test_agents_name_is_reserved_even_when_disabled():
    flow = Flow(StubLLM(lambda _messages: "unused"))

    with pytest.raises(ValueError, match="reserved"):
        flow.inject("AGENTS", {})
    with pytest.raises(ValueError, match="reserved"):
        flow.remove_tool("AGENTS")
