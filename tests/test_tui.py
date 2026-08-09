"""Panels and consumer behaviour for the Textual dashboard.

The panel helpers are pure ``root -> renderable`` functions, so they are checked
by rendering to a plain string and reading it. Driving the ``App`` itself needs a
terminal, so ``FlowTUI`` is exercised through the consumer surface instead.
"""

from __future__ import annotations

import io
import subprocess
import sys

from rich.console import Console

from rlmflow import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    LLMUsage,
    start,
)
from rlmflow.consumers.tui import (
    FlowTUI,
    agent_table,
    error_table,
    latest_table,
    node_counts_table,
    node_panel,
    overview_table,
    waiting_table,
)


def _text(renderable) -> str:
    out = io.StringIO()
    Console(file=out, width=200, force_terminal=False).print(renderable)
    return out.getvalue()


def _delegating_run() -> tuple[AgentStart, ExecAction, AgentStart]:
    """A root that launched one child and is waiting on it."""
    root = start("research the thing")
    thinking = root.append(
        LLMOutput(
            content="I will inspect it.",
            code='print("checking")',
            usage=LLMUsage(10, 6),
        )
    )
    action = thinking.append(ExecAction(code="await launch_subagents(...)"))
    child = action.append(AgentStart(content="lookup", config=root.config.child("lookup")))
    return root, action, child


def test_overview_table_counts_agents_nodes_and_tokens():
    root, _action, child = _delegating_run()
    child.append(LLMOutput(content="looking", usage=LLMUsage(4, 1)))

    rendered = _text(overview_table(root, busy=True))
    rows = {
        line.rsplit(" ", 1)[0].strip(): line.rsplit(" ", 1)[1]
        for line in rendered.splitlines()
        if line.strip()
    }

    assert rows["status"] == "running"
    assert rows["agents"] == "2"
    assert rows["max depth"] == "1"
    # The child's spend is rolled into the root's total.
    assert rows["tokens in"] == "14"
    assert rows["tokens out"] == "7"


def test_overview_table_status_tracks_terminal_state():
    root = start("query")

    assert "ready" in _text(overview_table(root))

    root.append(DoneOutput(result="answer"))
    assert "done" in _text(overview_table(root))


def test_agent_table_lists_each_agent_with_frontier_and_turns():
    root, _action, child = _delegating_run()
    child.append(DoneOutput(result="child answer"))

    rendered = _text(agent_table(root))

    assert "root.lookup" in rendered
    assert "active" in rendered  # the root is still on its ExecAction
    assert "done" in rendered  # the child answered
    assert "exec_action" in rendered
    assert "done_output" in rendered


def test_node_counts_table_tallies_node_types():
    root, _action, _child = _delegating_run()

    rendered = _text(node_counts_table(root))

    assert "agent_start" in rendered
    assert "llm_output" in rendered
    assert "exec_action" in rendered


def test_waiting_table_reports_running_children_then_clears():
    root, _action, child = _delegating_run()

    assert "1/1 running" in _text(waiting_table(root))

    child.append(DoneOutput(result="child answer"))
    assert "0/1 running" in _text(waiting_table(root))


def test_waiting_table_says_none_without_delegation():
    root = start("query")
    thinking = root.append(LLMOutput(content="thinking"))

    assert "none" in _text(waiting_table(root))

    # An ExecAction frontier that launched no children is not "waiting" either.
    thinking.append(ExecAction(code="print(1)"))
    assert "none" in _text(waiting_table(root))


def test_error_table_lists_failures_and_says_none_when_clean():
    root = start("query")
    thinking = root.append(LLMOutput(content="thinking"))

    assert "none" in _text(error_table(root))

    action = thinking.append(ExecAction(code="boom()"))
    action.append(ErrorOutput(content="NameError: boom is not defined", error="exec"))
    rendered = _text(error_table(root))
    assert "NameError: boom is not defined" in rendered
    assert "root" in rendered


def test_latest_table_reports_recent_nodes_and_is_bounded():
    root = start("query")
    node = root.append(LLMOutput(content="thinking"))

    assert "-" in _text(latest_table([]))

    rendered = _text(latest_table([root, node]))
    assert "llm_output" in rendered
    assert "agent_start" in rendered

    many = [node] * 30
    assert _text(latest_table(many, limit=3)).count("llm_output") == 3


def test_node_panel_prefers_the_result_of_a_done_node():
    root = start("query")
    done = root.append(DoneOutput(content="", result="the answer"))

    rendered = _text(node_panel(done))

    assert "the answer" in rendered
    assert "done_output" in rendered


def test_node_panel_falls_back_to_the_node_type_when_empty():
    root = start("query")
    node = root.append(LLMOutput(content=""))

    assert "(llm_output)" in _text(node_panel(node))


def test_flow_tui_tracks_the_root_off_the_streamed_node():
    root = start("query")
    node = root.append(LLMOutput(content="thinking"))
    ui = FlowTUI()

    ui.handle(node)

    assert ui.root is root
    assert ui.latest == [node]


def test_flow_tui_keeps_only_the_last_hundred_nodes():
    root = start("query")
    ui = FlowTUI(root)
    node = root.append(LLMOutput(content="thinking"))

    for _ in range(150):
        ui.handle(node)

    assert len(ui.latest) == 100


def test_flow_tui_refreshes_the_dashboard_while_open():
    root = start("query")
    node = root.append(LLMOutput(content="thinking"))

    class FakeApp:
        def __init__(self) -> None:
            self.refreshes = 0

        def refresh_dashboard(self) -> None:
            self.refreshes += 1

    ui = FlowTUI(root)
    ui.app = FakeApp()

    ui.handle(node)
    ui.close()

    assert ui.app.refreshes == 2


def test_importing_rlmflow_does_not_pull_in_textual():
    """``FlowTUI`` is exported eagerly, so its Textual imports must stay lazy."""
    code = (
        "import sys, rlmflow\n"
        "assert hasattr(rlmflow, 'FlowTUI')\n"
        "assert 'textual' not in sys.modules, 'textual leaked into the import path'\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_panels_survive_an_exec_output_only_run():
    """A run with no delegation and no errors still renders every panel."""
    root = start("query")
    thinking = root.append(LLMOutput(content="thinking", code="print(1)"))
    action = thinking.append(ExecAction(code="print(1)"))
    action.append(ExecOutput(content="1"))

    for table in (
        overview_table(root),
        agent_table(root),
        node_counts_table(root),
        waiting_table(root),
        error_table(root),
        latest_table(list(root.walk())),
    ):
        assert _text(table).strip()
