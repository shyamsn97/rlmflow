from rlmflow import AgentStart, DoneOutput, ExecAction, LLMOutput, start
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
