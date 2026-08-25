"""The figure, the timeline, the stepper, and the CLI that reaches them."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

import rlmflow
from rlmflow import DoneOutput, ExecAction, ExecOutput, LLMOutput, persistence
from rlmflow.cli import main
from rlmflow.graph.nodes import AgentStart, Node, UserQuery, start
from rlmflow.view import (
    NODE_COLORS,
    NODE_SHAPES,
    figure_title,
    graph_svg,
    node_color,
    render_html,
    render_steps,
    replay,
    save_html,
    save_svg,
    snapshot,
    steps,
    timeline,
)
from rlmflow.view.figure import LABEL_ADVANCE, LABEL_SIZE, layout_graph


@pytest.fixture
def ran() -> AgentStart:
    """A run with a sub-agent, so the figure has a fan in it."""
    root = start("do a thing", max_depth=2)
    action = root.append(LLMOutput(content="delegate")).append(
        ExecAction(code="await launch_subagent('sub')")
    )
    child = action.append(AgentStart(content="sub", config=root.config.child("count")))
    child.append(LLMOutput(content="calculate")).append(ExecAction(code="print(2 + 2)")).append(
        ExecOutput(content="4")
    ).append(DoneOutput(content="four", result="four"))
    action.append(ExecOutput(content="four")).append(
        DoneOutput(content="finished", result="finished")
    )
    return root


def dimensions(svg: str) -> tuple[float, float]:
    match = re.search(r'width="(\d+)" height="(\d+)"', svg)
    assert match, "figure has no size"
    return float(match.group(1)), float(match.group(2))


def test_every_node_type_has_a_colour_and_a_shape() -> None:
    assert set(NODE_COLORS) == set(NODE_SHAPES)


def test_timeline_is_creation_order_and_parents_come_first(ran: AgentStart) -> None:
    ordered = timeline(ran)
    assert ordered[0] is ran
    assert [n.created_at for n in ordered] == sorted(n.created_at for n in ordered)
    seen: set[str] = set()
    for node in ordered:
        if node.parent is not None:
            assert node.parent.id in seen or node.parent.id not in {n.id for n in ordered}
        seen.add(node.id)


def test_steps_cover_the_timeline_once(ran: AgentStart) -> None:
    walked = steps(ran)
    assert [step.index for step in walked] == list(range(len(walked)))
    assert [step.node.id for step in walked] == [node.id for node in timeline(ran)]
    assert all(step.elapsed >= 0 for step in walked)


def test_figure_draws_a_marker_per_node(ran: AgentStart) -> None:
    ordered = timeline(ran)
    svg = graph_svg(ordered, title=figure_title(ran, ordered))
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert f"{len(ordered)} nodes" in svg
    assert node_color(ordered[0]) in svg


def test_figure_keeps_labels_and_markers_inside_the_canvas(ran: AgentStart) -> None:
    ordered = timeline(ran)
    width, height = dimensions(graph_svg(ordered, title="t"))
    layout = layout_graph(ordered)
    for placed in layout.placed:
        half = max(placed.r, len(placed.label) * LABEL_SIZE * layout.scale * LABEL_ADVANCE / 2)
        assert placed.x - half >= -0.5
        assert placed.x + half <= width + 0.5
        assert placed.y + placed.r <= height + 0.5


def test_deep_chains_stay_a_readable_size() -> None:
    """A long run must not turn into a mile-high image."""
    root = start("deep")
    node: Node = root
    for i in range(400):
        node = node.append(UserQuery(content=f"step {i}"))
    _, height = dimensions(graph_svg(timeline(root), title="deep"))
    assert height <= 4000, f"figure is {height}px tall"


def test_svg_and_html_render_past_the_recursion_limit() -> None:
    root = start("deep")
    node: Node = root
    for i in range(2_000):
        node = node.append(UserQuery(content=f"step {i}"))

    assert graph_svg(timeline(root), title="deep").startswith("<svg")
    assert render_html(root).startswith("<!doctype html>")


def test_layout_handles_ten_thousand_node_chain() -> None:
    root = start("deep")
    node: Node = root
    for i in range(9_999):
        node = node.append(UserQuery(content=f"step {i}"))

    layout = layout_graph(list(root.walk()))

    assert len(layout.placed) == 10_000


def test_empty_graph_renders_a_placeholder() -> None:
    assert "empty graph" in graph_svg([])


def test_stepper_embeds_one_figure_with_a_frame_per_node(ran: AgentStart) -> None:
    ordered = timeline(ran)
    html = render_html(ran)
    assert html.count('class="rlm-n"') == len(ordered)
    assert html.count('id="rlm-ring"') == 1
    assert html.count("<svg") == 1, "the figure should be drawn once, not once per step"
    assert html.count('"kind":') == len(ordered)


def test_saved_files_land_where_asked(ran: AgentStart, tmp_path) -> None:
    svg = save_svg(ran, tmp_path / "nested" / "g.svg")
    html = save_html(ran, tmp_path / "r.html")
    assert svg.read_text().startswith("<svg")
    assert "<!doctype html>" in html.read_text()


def test_save_svg_clamps_a_step_out_of_range(ran: AgentStart, tmp_path) -> None:
    svg = save_svg(ran, tmp_path / "g.svg", step=10_000).read_text()
    assert svg.startswith("<svg")


def test_cli_view_prints_the_tree_and_timeline(ran: AgentStart, tmp_path, capsys) -> None:
    persistence.save(ran, tmp_path / "graph")
    assert main(["view", "show", str(tmp_path / "graph")]) == 0
    out = capsys.readouterr().out
    assert "steps" in out
    assert "llm_output" in out


def test_cli_render_writes_both_formats(ran: AgentStart, tmp_path, capsys) -> None:
    persistence.save(ran, tmp_path / "graph")
    graph = str(tmp_path / "graph")

    assert main(["render", "svg", graph, str(tmp_path / "g.svg")]) == 0
    assert main(["render", "html", graph, str(tmp_path / "r.html")]) == 0

    capsys.readouterr()
    assert (tmp_path / "g.svg").read_text().startswith("<svg")
    assert "rlm-ring" in (tmp_path / "r.html").read_text()


def test_cli_view_reports_a_bad_path_and_a_bad_step(ran: AgentStart, tmp_path, capsys) -> None:
    assert main(["view", "show", str(tmp_path / "missing")]) == 1
    assert "not found" in capsys.readouterr().err
    persistence.save(ran, tmp_path / "graph")
    assert main(["view", "show", str(tmp_path / "graph"), "--step", "9999"]) == 1
    assert "out of range" in capsys.readouterr().err


def test_cli_view_prints_one_step(ran: AgentStart, tmp_path, capsys) -> None:
    persistence.save(ran, tmp_path / "graph")
    assert main(["view", "show", str(tmp_path / "graph"), "--step", "1"]) == 0
    assert "step 1/" in capsys.readouterr().out


def test_replay_grows_a_node_at_a_time(ran: AgentStart) -> None:
    ordered = timeline(ran)
    snaps = list(replay(ran))
    assert len(snaps) == len(ordered)
    assert [len(list(snap.walk())) for snap in snaps] == list(range(1, len(ordered) + 1))


def test_replay_snapshots_are_separate_trees(ran: AgentStart) -> None:
    """They can be kept, compared, and rendered out of order."""
    before = len(list(ran.walk()))
    snaps = list(replay(ran))
    assert len({id(snap) for snap in snaps}) == len(snaps)
    assert len(list(ran.walk())) == before, "replaying must not touch the run"
    assert all(snap is not ran for snap in snaps)


def test_snapshot_rebuilds_agents_and_frontiers(ran: AgentStart) -> None:
    whole = snapshot(timeline(ran))
    assert len(whole.sub_agents) == len(ran.sub_agents)
    assert whole.terminal == ran.terminal
    assert whole.result() == ran.result()
    assert whole.frontier.id == ran.frontier.id
    for node in whole.walk():
        assert node.root is whole
        if node.parent is not None:
            assert node in node.parent.children


def test_snapshot_refuses_a_headless_prefix(ran: AgentStart) -> None:
    with pytest.raises(ValueError, match="at least a root"):
        snapshot([])
    with pytest.raises(ValueError, match="start at an agent"):
        snapshot(timeline(ran)[1:])


def test_render_steps_is_one_ascii_tree_per_step(ran: AgentStart) -> None:
    frames = list(render_steps(ran))
    assert len(frames) == len(timeline(ran))
    assert ran.config.path in frames[-1]


def test_view_holds_optional_renderers_back_until_asked() -> None:
    """Importing rlmflow must not drag in Gradio or a rasteriser.

    Checked in a clean interpreter, because ``sys.modules`` in this one says
    only that no other test happened to import Gradio first — and some
    dependencies (the OpenEnv client, for one) import it for their own UI.
    """
    probe = (
        "import sys, rlmflow, rlmflow.view; "
        "print([n for n in ('gradio', 'cairosvg', 'PIL') if n in sys.modules])"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert loaded.stdout.strip() == "[]", f"rlmflow imported {loaded.stdout.strip()}"
    assert {"open_viewer", "save_frames", "save_gif"} <= set(dir(rlmflow.view))
    with pytest.raises(AttributeError, match="no attribute"):
        _ = rlmflow.view.nope


def test_figure_and_panels_render_without_gradio(ran: AgentStart) -> None:
    from rlmflow.view.app import _panel, _transcript, figure_html

    ordered = timeline(ran)
    html = figure_html(ordered, 2, title="root")
    assert "<svg" in html and "max-width:100%" in html
    assert "agent_start" in _panel(steps(ran)[0])
    assert ran.config.path in _transcript(ordered, ran.config.path)
    assert "not started" in _transcript(ordered, "root.nope")


def test_frames_and_gif_export(ran: AgentStart, tmp_path) -> None:
    pytest.importorskip("cairosvg")
    pytest.importorskip("PIL")
    from rlmflow.view.images import save_frames, save_gif

    written = save_frames(ran, tmp_path / "frames", every=2)
    assert written and all(path.exists() for path in written)
    assert len(written) == len(range(0, len(timeline(ran)), 2))
    gif = save_gif(ran, tmp_path / "run.gif", every=3)
    assert gif.read_bytes().startswith(b"GIF89a")


def test_cli_view_prints_every_ascii_step(ran: AgentStart, tmp_path, capsys) -> None:
    persistence.save(ran, tmp_path / "graph")
    assert main(["view", "show", str(tmp_path / "graph"), "--frames-only"]) == 0
    out = capsys.readouterr().out
    assert out.count("=== step ") == len(timeline(ran))


def test_cli_view_exports_images(ran: AgentStart, tmp_path, capsys) -> None:
    pytest.importorskip("cairosvg")
    persistence.save(ran, tmp_path / "graph")
    graph = str(tmp_path / "graph")
    assert main(["view", "show", graph, "--tree"]) == 0
    assert main(["render", "frames", graph, str(tmp_path / "frames"), "--every", "2"]) == 0
    assert main(["render", "gif", graph, str(tmp_path / "run.gif"), "--every", "2"]) == 0
    assert "wrote" in capsys.readouterr().out
    assert list((tmp_path / "frames").glob("*.png"))
    assert (tmp_path / "run.gif").exists()
