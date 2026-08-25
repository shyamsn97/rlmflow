import pytest

from rlmflow import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    LLMOutput,
    UserQuery,
    start,
)
from rlmflow.consumers import (
    ConsumerGroup,
    FlowTUI,
    GraphCheckpointer,
    LiveGraphTree,
    LiveTreeRenderer,
    render_tree,
)
from rlmflow.consumers.tui import agent_table, overview_table


def test_render_tree_shows_live_child_progress_and_results():
    root = start("find the needle")
    output = root.append(LLMOutput(content="search"))
    action = output.append(ExecAction(code="await launch_subagent(...)"))
    child = action.append(
        AgentStart(
            content="search batch",
            config=root.config.child("batch"),
        )
    )

    running = render_tree(root)
    assert "root: children running 1/1" in running
    assert "root.batch: pending" in running

    child.append(DoneOutput(result="found"))
    finished = render_tree(root)
    assert "root: children running 0/1" in finished
    assert "root.batch: done found" in finished


def test_live_tree_renderer_handles_streamed_nodes(capsys):
    root = start("query")
    node = root.append(LLMOutput(content="thinking"))

    LiveTreeRenderer(clear=False).handle(node)

    output = capsys.readouterr().out
    assert "query" in output
    assert "root: planning" in output


def test_live_graph_tree_has_plain_fallback(capsys):
    root = start("query")
    node = root.append(LLMOutput(content="thinking"))
    viewer = LiveGraphTree(
        title="search",
        rich=False,
        clear=False,
        footer=lambda: "footer",
    )
    viewer.track(root, label="first")
    viewer.label(root.id, "renamed")

    viewer.handle(node)
    viewer.close()

    output = capsys.readouterr().out
    assert "search" in output
    assert "root: planning" in output
    assert "footer" in output


def test_tui_panels_use_node_tree():
    root = start("query")
    root.append(LLMOutput(content="thinking"))

    assert FlowTUI(root).root is root
    assert overview_table(root) is not None
    assert agent_table(root) is not None


def test_consumer_group_renders_and_checkpoints(tmp_path, capsys):
    root = start("query")
    node = root.append(LLMOutput(content="thinking"))
    checkpoint = GraphCheckpointer(tmp_path / "run")
    consumers = ConsumerGroup([LiveTreeRenderer(clear=False), checkpoint])

    consumers.handle(node)
    consumers.close()

    loaded = AgentStart.load(tmp_path / "run")
    assert loaded.frontier.type == "llm_output"
    assert "root: planning" in capsys.readouterr().out


def test_checkpointer_flushes_by_node_count_and_on_close(tmp_path):
    root = start("query")
    checkpoint = GraphCheckpointer(
        tmp_path / "run",
        interval_s=None,
        interval_nodes=2,
    )

    first = root.append(UserQuery(content="one"))
    checkpoint.handle(first)
    assert not (tmp_path / "run" / "graph.json").exists()

    second = first.append(UserQuery(content="two"))
    checkpoint.handle(second)
    assert AgentStart.load(tmp_path / "run").stats.node_count == 3

    checkpoint.handle(second.append(UserQuery(content="three")))
    checkpoint.close()
    assert AgentStart.load(tmp_path / "run").stats.node_count == 4


def test_checkpointer_rejects_ambiguous_zero_thresholds(tmp_path):
    with pytest.raises(ValueError, match="interval_s"):
        GraphCheckpointer(tmp_path, interval_s=0)
    with pytest.raises(ValueError, match="interval_nodes"):
        GraphCheckpointer(tmp_path, interval_nodes=0)


@pytest.mark.parametrize("terminal", [DoneOutput(result="ok"), ErrorOutput(content="boom")])
def test_checkpointer_flushes_terminal_events_immediately(tmp_path, terminal):
    root = start("query")
    node = root.append(terminal)
    checkpoint = GraphCheckpointer(
        tmp_path / "run",
        interval_s=None,
        interval_nodes=None,
    )

    checkpoint.handle(node)

    assert AgentStart.load(tmp_path / "run").frontier.type == node.type
