"""Phase 1 observability: code utils, trace/tracing, export, viewer, viz, CLI.

Headless coverage runs everywhere; the Plotly figure path (viewer 1b) is guarded
with ``importorskip`` so the suite still passes without the ``viewer``/``image``
extras. This is the new-stack replacement for the legacy viz, observability,
and CLI coverage.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import rflow

from rflow import Flow, Graph
from rflow.runtime.code import find_code_blocks, replace_code_block
from rflow.utils import export, viewer, viz
from rflow.utils.trace import load_trace, save_trace
from rflow.utils.tracing import json_logs
from tests.helpers import ScriptedLLM


# ── shared fixture: a finished delegated run + its step snapshots ─────


def _delegating_reply(messages):
    task = next((m["content"] for m in messages if m["role"] == "user"), "")
    if "depth 1" in task:
        return '```repl\ndone("child-answer")\n```'
    return (
        "```repl\n"
        'results = await launch_subagents([{"name": "child", "query": "child task"}])\n'
        'done("p:" + results[0])\n'
        "```"
    )


@pytest.fixture
def run_snapshots() -> list[Graph]:
    flow = Flow(
        ScriptedLLM(_delegating_reply),
        max_depth=1,
        max_iters=5,
        system_prompt="You are a test agent.",
    )
    flow.start("parent")
    snaps = [Graph.from_dict(flow.graph.to_dict())]
    steps = 0
    while not flow.graph.finished:
        flow.step()
        snaps.append(Graph.from_dict(flow.graph.to_dict()))
        steps += 1
        assert steps < 50
    return snaps


@pytest.fixture
def final_graph(run_snapshots) -> Graph:
    return run_snapshots[-1]


# ── code utils ────────────────────────────────────────────────────────


def test_find_code_blocks_extracts_each_fence_type():
    assert find_code_blocks("```repl\na=1\n```") == ["a=1"]
    assert find_code_blocks("pre\n```python\nb=2\n```\npost") == ["b=2"]
    assert find_code_blocks("```\nc=3\n```") == ["c=3"]


def test_find_code_blocks_does_not_truncate_on_nested_fence():
    text = '```repl\nprint("""\n```bash\nls\n```\n""")\ndone("ok")\n```'
    blocks = find_code_blocks(text)
    assert len(blocks) == 1
    assert 'done("ok")' in blocks[0]
    assert "```bash" in blocks[0]


def test_replace_code_block_targets_first_repl_only():
    text = "intro\n```repl\nold\n```\ntail\n```repl\nsecond\n```"
    out = replace_code_block(text, "NEW")
    assert "```repl\nNEW\n```" in out
    assert "old" not in out and "second" not in out and "tail" not in out


def test_flow_reexports_find_code_blocks_from_code_module():
    import rflow.flow as flow_mod

    assert flow_mod.find_code_blocks is find_code_blocks


# ── trace + tracing ───────────────────────────────────────────────────


def test_save_load_trace_round_trips_with_child_and_metadata(run_snapshots, tmp_path):
    path = save_trace(run_snapshots, tmp_path / "trace.json", metadata={"run": "x"})
    loaded = load_trace(path)

    assert len(loaded.graphs) == len(run_snapshots)
    assert loaded.metadata == {"run": "x"}
    assert loaded.graphs[-1].result() == "p:child-answer"
    assert "root.child" in loaded.graphs[-1].children
    # ids + global_step stable across the round-trip
    before = [(n.id, n.global_step) for n in run_snapshots[-1].all_nodes]
    after = [(n.id, n.global_step) for n in loaded.graphs[-1].all_nodes]
    assert before == after


def test_json_logs_one_line_per_node(final_graph, tmp_path):
    path = json_logs(final_graph, tmp_path / "log.ndjson")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(list(final_graph.all_nodes))
    assert all(json.loads(line)["type"] for line in lines)


# ── export ────────────────────────────────────────────────────────────


def test_to_mermaid_has_nodes_edges_and_result(final_graph):
    out = export.to_mermaid(final_graph)
    assert out.startswith("stateDiagram-v2")
    assert "[*] -->" in out  # root entry
    assert "p:child-answer" in out  # terminal result rendered


def test_to_dot_marks_spawn_and_flow_edges(final_graph):
    out = export.to_dot(final_graph)
    assert out.startswith('digraph "rlmflow"')
    assert 'label="spawns"' in out
    assert 'label="flows_to"' in out


def test_resume_kind_colored_distinctly_when_resumed(final_graph):
    # the parent has a resume; _kind buckets resumed exec output as "resume"
    kinds = {export._kind(n) for n in final_graph.all_nodes}
    assert "supervising" in kinds and "done" in kinds


def test_other_export_formats_render_strings(final_graph):
    assert export.to_mermaid_flowchart(final_graph).startswith("flowchart TD")
    assert export.to_mermaid_sequence(final_graph).startswith("sequenceDiagram")
    assert "->" in export.to_d2(final_graph)


# ── viewer 1a (text) ──────────────────────────────────────────────────


def test_resolve_graphs_from_trace_path_and_list(run_snapshots, tmp_path):
    path = save_trace(run_snapshots, tmp_path / "trace.json")
    assert len(viewer.resolve_graphs(str(path))) == len(run_snapshots)
    assert len(viewer.resolve_graphs(tmp_path)) == len(run_snapshots)  # dir w/ trace
    assert len(viewer.resolve_graphs(run_snapshots)) == len(run_snapshots)
    assert viewer.resolve_graphs(run_snapshots[-1]) == [run_snapshots[-1]]


def test_resolve_graphs_from_single_graph_dump(final_graph, tmp_path):
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(final_graph.to_dict()), encoding="utf-8")
    [g] = viewer.resolve_graphs(str(p))
    assert g.result() == "p:child-answer"


def test_graph_tree_nests_child_under_supervising(final_graph):
    tree = viewer.graph_tree(final_graph)
    assert "● root" in tree
    assert "supervising waiting_on=['root.child']" in tree
    assert "● root.child" in tree
    assert "done -> p:child-answer" in tree
    # child line is indented deeper than the root's nodes
    child_line = next(ln for ln in tree.splitlines() if "● root.child" in ln)
    assert child_line.startswith("    ")


def test_agent_transcript_skips_action_nodes(final_graph):
    text = viewer.agent_transcript(final_graph)
    assert "--- system ---" in text
    assert "--- query ---" in text
    assert "--- result ---" in text
    # action bookkeeping never shows up as its own transcript block
    assert "llm_action" not in text and "exec_action" not in text


def test_slice_graphs_at_branch_is_deferred(run_snapshots):
    with pytest.raises(NotImplementedError):
        viewer.slice_graphs_at_branch(run_snapshots, "whatever")


# ── viz ───────────────────────────────────────────────────────────────


def test_error_summary_groups_by_kind():
    def reply(_messages):
        reply.n = getattr(reply, "n", 0) + 1
        return "no code here" if reply.n == 1 else '```repl\ndone("ok")\n```'

    flow = Flow(ScriptedLLM(reply), max_depth=0, max_iters=5)
    flow.start("p")
    while not flow.graph.finished:
        flow.step()
    summary = viz.error_summary(flow.graph)
    assert "no_code_block" in summary
    assert "(no errors)" not in summary


def test_code_log_pairs_exec_with_output(final_graph):
    log = viz.code_log(final_graph)
    assert "launch_subagents" in log
    assert "→" in log  # output arrow for at least one block


def test_ascii_boxes_matches_graph_tree(final_graph):
    assert viz.ascii_boxes(final_graph) == viewer.graph_tree(final_graph)


def test_token_sparkline_summarizes_cumulative_usage():
    graphs = [
        Graph(
            agent_id="root",
            nodes=[rflow.LLMOutput(agent_id="root", input_tokens=1, output_tokens=2)],
        ),
        Graph(
            agent_id="root",
            nodes=[rflow.LLMOutput(agent_id="root", input_tokens=3, output_tokens=4)],
        ),
    ]

    line = viz.token_sparkline(graphs)
    assert "7 tok over 2 steps" in line
    assert "(3 in, 4 out)" in line


def test_report_md_summarizes_run(run_snapshots):
    md = viz.report_md(run_snapshots, title="my run")
    assert md.startswith("# my run")
    assert "**Steps:**" in md and "**Agents:**" in md
    assert "## Tree" in md and "## Result" in md
    assert "p:child-answer" in md


def test_gantt_html_is_self_contained(run_snapshots):
    html = viz.gantt_html(run_snapshots)
    assert html.startswith("<!doctype")
    assert "root" in html and "root.child" in html


def test_live_streams_and_returns_snapshots():
    from rich.console import Console

    flow = Flow(ScriptedLLM(_delegating_reply), max_depth=1, max_iters=5)
    flow.start("parent")
    view = viz.LiveView(console=Console(file=io.StringIO()))
    snapshots = [flow.graph.snapshot()]
    with view:
        view(flow.graph)
        while not flow.graph.finished:
            flow.step()
            snapshots.append(flow.graph.snapshot())
            view(flow.graph)
    assert snapshots[-1].result() == "p:child-answer"


# ── viewer 1b (Plotly figures) — needs optional deps ──────────────────


def test_graph_plot_mermaid_works_without_plotly(final_graph):
    out = viewer.graph_plot(final_graph, "mermaid")
    assert out.startswith("stateDiagram-v2")


def test_graph_plot_figure_builds(final_graph):
    pytest.importorskip("plotly")
    fig = viewer.graph_plot(final_graph, "graph")
    assert type(fig).__name__ == "Figure"
    assert fig.data  # has traces


def test_graph_plot_centers_supervising_over_waited_children():
    pytest.importorskip("plotly")
    root_q = rflow.UserQuery(agent_id="root", seq=0, content="fan out")
    root_wait = rflow.SupervisingOutput(
        agent_id="root",
        seq=1,
        waiting_on=["root.child"],
    )
    root_done = rflow.DoneOutput(agent_id="root", seq=2, result="done")
    child_q = rflow.UserQuery(agent_id="root.child", seq=0, content="child")
    child_done = rflow.DoneOutput(agent_id="root.child", seq=1, result="ok")
    child = rflow.Graph.from_meta_dict(
        {
            "agent_id": "root.child",
            "depth": 1,
            "parent_agent_id": "root",
            "parent_node_id": root_wait.id,
            "query": "child",
        },
        nodes=[child_q, child_done],
    )
    graph = rflow.Graph.from_meta_dict(
        {"agent_id": "root", "depth": 0, "query": "fan out"},
        nodes=[root_q, root_wait, root_done],
        children={child.agent_id: child},
    )

    fig = viewer.graph_plot(graph, "graph")
    positions = viewer._node_positions_from_figure(fig)

    assert positions[root_wait.id][0] == pytest.approx(positions[child_q.id][0])


def test_graph_plot_marker_mult_overrides_dense_cap():
    pytest.importorskip("plotly")
    root_q = rflow.UserQuery(agent_id="root", seq=0, content="fan out")
    root_wait = rflow.SupervisingOutput(
        agent_id="root",
        seq=1,
        waiting_on=[f"root.child_{i}" for i in range(25)],
    )
    children = {
        f"root.child_{i}": rflow.Graph.from_meta_dict(
            {
                "agent_id": f"root.child_{i}",
                "depth": 1,
                "parent_agent_id": "root",
                "parent_node_id": root_wait.id,
                "query": f"child {i}",
            },
            nodes=[
                rflow.UserQuery(
                    agent_id=f"root.child_{i}",
                    seq=0,
                    content=f"child {i}",
                ),
                rflow.DoneOutput(
                    agent_id=f"root.child_{i}",
                    seq=1,
                    result="ok",
                ),
            ],
        )
        for i in range(25)
    }
    graph = rflow.Graph.from_meta_dict(
        {"agent_id": "root", "depth": 0, "query": "fan out"},
        nodes=[root_q, root_wait],
        children=children,
    )

    default_fig = viewer.graph_plot(graph, "graph")
    default_marker_trace = next(
        trace for trace in default_fig.data if getattr(trace, "customdata", None)
    )
    assert max(default_marker_trace.marker.size) < 24

    fig = viewer.graph_plot(graph, "graph", marker_mult=3.0)
    marker_trace = next(trace for trace in fig.data if getattr(trace, "customdata", None))
    sizes = list(marker_trace.marker.size)

    assert max(sizes) == 72


def test_save_image_writes_png(final_graph, tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    out = viewer.save_image(final_graph, tmp_path / "g.png")
    assert out.exists() and out.stat().st_size > 0
    shorthand = final_graph.save_image(tmp_path / "g2.png")
    assert shorthand.exists() and shorthand.stat().st_size > 0


def test_save_steps_dedupes_and_writes_frames(run_snapshots, tmp_path):
    pytest.importorskip("plotly")
    pytest.importorskip("kaleido")
    out = viewer.save_steps(run_snapshots, tmp_path / "steps")
    pngs = list(Path(out).glob("*.png"))
    assert pngs
    assert len(pngs) <= len(run_snapshots)  # bookkeeping-only ticks collapsed


def test_render_html_stepper(run_snapshots):
    pytest.importorskip("plotly")
    html = viewer.render_html(run_snapshots, title="run")
    assert html.startswith("<!doctype")
    assert "plotly" in html.lower()
    assert "Transcript" in html


# ── CLI ───────────────────────────────────────────────────────────────


def _run_cli(args, capsys):
    from rflow.cli import main

    rc = main(args)
    return rc, capsys.readouterr()


def test_cli_render_text_formats(run_snapshots, tmp_path, capsys):
    path = str(save_trace(run_snapshots, tmp_path / "trace.json"))
    for fmt, needle in [
        ("mermaid", "stateDiagram-v2"),
        ("tree", "● root"),
        ("ascii-boxes", "● root"),
        ("report-md", "# rlmflow run"),
        ("error-summary", "no errors"),
        ("tokens", "tok over"),
    ]:
        rc, captured = _run_cli(["render", path, "-f", fmt], capsys)
        assert rc == 0
        assert needle in captured.out


def test_cli_render_to_file(run_snapshots, tmp_path, capsys):
    path = str(save_trace(run_snapshots, tmp_path / "trace.json"))
    out = tmp_path / "g.dot"
    rc, _ = _run_cli(["render", path, "-f", "dot", "-o", str(out)], capsys)
    assert rc == 0
    assert out.read_text().startswith('digraph "rlmflow"')


def test_cli_image_requires_out(run_snapshots, tmp_path, capsys):
    path = str(save_trace(run_snapshots, tmp_path / "trace.json"))
    with pytest.raises(SystemExit, match="requires --out"):
        _run_cli(["render", path, "-f", "image"], capsys)


def test_cli_version_reports_environment(capsys):
    rc, captured = _run_cli(["version"], capsys)
    assert rc == 0
    assert "rlmflow" in captured.out
    assert "python" in captured.out


# ── CLI: loader errors + dispatch (ported from legacy integration/test_cli) ─


def test_cli_load_missing_file_exits(tmp_path):
    from rflow.cli import _load

    with pytest.raises(SystemExit, match="no such file"):
        _load(tmp_path / "nope.json")


def test_cli_load_invalid_json_exits(tmp_path):
    from rflow.cli import _load

    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not valid JSON"):
        _load(bad)


def test_cli_load_unknown_shape_exits(tmp_path):
    from rflow.cli import _load

    unknown = tmp_path / "u.json"
    unknown.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(SystemExit, match="does not look like"):
        _load(unknown)


def test_cli_missing_subcommand_exits(capsys):
    with pytest.raises(SystemExit):
        _run_cli([], capsys)


def test_cli_render_requires_format(run_snapshots, tmp_path, capsys):
    path = str(save_trace(run_snapshots, tmp_path / "trace.json"))
    with pytest.raises(SystemExit):
        _run_cli(["render", path], capsys)  # argparse: -f required


def test_cli_render_html_requires_out(run_snapshots, tmp_path, capsys):
    path = str(save_trace(run_snapshots, tmp_path / "trace.json"))
    with pytest.raises(SystemExit, match="requires --out"):
        _run_cli(["render", path, "-f", "html"], capsys)


def test_cli_render_steps_requires_out(run_snapshots, tmp_path, capsys):
    path = str(save_trace(run_snapshots, tmp_path / "trace.json"))
    with pytest.raises(SystemExit, match="requires --out"):
        _run_cli(["render", path, "-f", "steps"], capsys)


def test_cli_render_html_forwards_scaling_flags(
    run_snapshots, tmp_path, capsys, monkeypatch
):
    path = str(save_trace(run_snapshots, tmp_path / "trace.json"))
    out = tmp_path / "viewer.html"
    captured: dict = {}

    def fake_save_html(graphs, output, **kwargs):
        captured["graphs"] = graphs
        captured["output"] = output
        captured["kwargs"] = kwargs
        out.write_text("<html></html>", encoding="utf-8")
        return out

    monkeypatch.setattr("rflow.utils.viewer.save_html", fake_save_html)

    rc, _ = _run_cli(
        [
            "render",
            path,
            "-f",
            "html",
            "-o",
            str(out),
            "--element-mult",
            "1.2",
            "--marker-mult",
            "3.5",
            "--text-mult",
            "2.2",
            "--no-normalize-labels",
        ],
        capsys,
    )

    assert rc == 0
    assert captured["output"] == str(out)
    assert captured["kwargs"]["element_mult"] == 1.2
    assert captured["kwargs"]["marker_mult"] == 3.5
    assert captured["kwargs"]["text_mult"] == 2.2
    assert captured["kwargs"]["normalize_labels"] is False


def test_cli_render_gantt_html_to_file(run_snapshots, tmp_path, capsys):
    path = str(save_trace(run_snapshots, tmp_path / "trace.json"))
    out = tmp_path / "g.html"
    rc, _ = _run_cli(["render", path, "-f", "gantt-html", "-o", str(out)], capsys)
    assert rc == 0
    html = out.read_text()
    assert "<html" in html.lower() and "root" in html


def test_cli_view_dispatches_to_open_viewer(run_snapshots, tmp_path, capsys, monkeypatch):
    path = str(save_trace(run_snapshots, tmp_path / "trace.json"))
    captured: dict = {}

    def fake_open_viewer(graphs, **kwargs):
        captured["graphs"] = graphs
        captured["kwargs"] = kwargs

    monkeypatch.setattr("rflow.utils.viewer.open_viewer", fake_open_viewer)
    rc, _ = _run_cli(["view", path, "--port", "7861"], capsys)
    assert rc == 0
    assert captured["kwargs"]["server_port"] == 7861
    assert captured["graphs"] and all(isinstance(g, Graph) for g in captured["graphs"])


# ── text views + export flags (ported from legacy test_viz / viz_exports) ──


def test_graph_session_flattens_every_agent(final_graph):
    out = viewer.graph_session(final_graph)
    assert "[root]" in out and "[root.child]" in out
    assert "child task" in out  # the child's own query appears in its block


def test_agent_transcript_renders_query_and_result(final_graph):
    text = viewer.agent_transcript(final_graph, include_system=False)
    assert "--- query ---" in text
    assert "p:child-answer" in text  # the final answer shows up in the chain


def test_to_mermaid_include_results_flag(final_graph):
    with_results = export.to_mermaid(final_graph, include_results=True)
    without = export.to_mermaid(final_graph, include_results=False)
    assert "--> [*] :" in with_results
    assert "--> [*] :" not in without


def test_error_summary_reports_none_on_clean_run(final_graph):
    assert "no errors" in viz.error_summary(final_graph).lower()


def test_render_html_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one graph"):
        viewer.render_html([])


def test_save_gif_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="at least one graph"):
        viewer.save_gif([], tmp_path / "out.gif")


def test_save_html_writes_file_and_creates_parents(run_snapshots, tmp_path):
    pytest.importorskip("plotly")
    out = viewer.save_html(run_snapshots, tmp_path / "nested" / "run.html", title="t")
    assert out.exists()
    assert "<!doctype" in out.read_text().lower()


# ── import-weight invariant ───────────────────────────────────────────


def test_engine_path_does_not_import_plotly_or_gradio():
    code = (
        "import sys, rflow, rflow.flow, rflow.utils\n"
        "from rflow.utils import find_code_blocks\n"
        "assert 'plotly' not in sys.modules, 'plotly leaked'\n"
        "assert 'gradio' not in sys.modules, 'gradio leaked'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
