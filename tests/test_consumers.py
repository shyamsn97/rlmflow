from __future__ import annotations

import json

from rlmflow import ConsumerGroup, Graph, GraphCheckpointer, StreamConsumer, WorkspaceSync


class Recorder(StreamConsumer):
    def __init__(self) -> None:
        self.events = []
        self.closed = False

    def handle(self, event, graph):
        self.events.append((event.type, graph.graph_id if graph is not None else None))

    def close(self) -> None:
        self.closed = True


def test_consumer_group_fans_out_and_closes():
    graph = Graph(query="q")
    event = graph.append("next")
    first = Recorder()
    second = Recorder()
    group = ConsumerGroup([first, second])

    group.handle(event, graph)
    group.close()

    assert first.events == second.events == [("append_node", graph.graph_id)]
    assert first.closed and second.closed


def test_graph_checkpointer_saves_current_graph(tmp_path):
    graph = Graph(query="q")
    event = graph.append("next")
    path = tmp_path / "graph"

    GraphCheckpointer(path).handle(event, graph)

    session = path / "agents" / "root" / "session.jsonl"
    rows = [json.loads(line) for line in session.read_text().splitlines()]
    assert [row["content"] for row in rows] == ["q", "next"]


def test_workspace_sync_mirrors_source_and_ignores_heavy_dirs(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "artifact.txt").write_text("ok")
    (source / ".venv").mkdir()
    (source / ".venv" / "ignored.txt").write_text("skip")
    graph = Graph(query="q")
    event = graph.append("next")

    WorkspaceSync(source, target, every_s=0.0).handle(event, graph)

    assert (target / "artifact.txt").read_text() == "ok"
    assert not (target / ".venv").exists()
