"""The task board: its state is the board calls saved on the nodes that made them."""

import threading

from helpers import stepping

from rlmflow import AgentConfig, Flow, Graph, TaskBoard
from rlmflow.tools.board import events
from rlmflow.view.figure import graph_svg

POSTER = """\
posted = board_post("sim", "Write sim.js")
claimed = board_claim("sim")
board_publish("sim api", "Boids.step(flock, dt)")
submit(claimed["ok"])"""

LATE = """\
seen = board_read()
claimed = board_claim("sim")
submit({"ok": claimed["ok"], "upto": seen["upto"]})"""


class Scripted:
    def chat(self, messages, **_):
        task = next(str(m["content"]) for m in messages if m["role"] == "user")
        return f"```repl\n{POSTER if 'poster' in task else LATE}\n```"


def run_board() -> tuple[Flow, Graph]:
    flow = Flow(
        configs=[AgentConfig(model="scripted", max_depth=0)],
        tools=[TaskBoard()],
        llm_clients={"scripted": Scripted()},
        workers=1,
    )
    g = Graph()
    g.start("You are the poster.", name="poster")
    g.start("You are late.", name="late")
    flow.run(g)
    return flow, g


def test_the_board_is_the_events_saved_on_the_nodes_that_made_them():
    _, g = run_board()
    poster, late = g.agents()

    assert poster.final_result() is True
    assert late.final_result() == {"ok": False, "upto": 3}
    assert [(e["op"], e["by"], e["ok"]) for e in events(g)] == [
        ("post", "poster", True),
        ("claim", "poster", True),
        ("publish", "poster", True),
        ("claim", "late", False),
    ]
    board = TaskBoard.state(g)
    assert board["tasks"]["sim"]["owner"] == "poster"
    assert board["findings"][0]["text"] == "Boids.step(flock, dt)"
    assert board["upto"] == 4


def test_a_saved_run_loads_with_its_board(tmp_path):
    _, g = run_board()
    g.save(tmp_path / "run")

    assert TaskBoard.state(Graph.load(tmp_path / "run")) == TaskBoard.state(g)


def test_a_fork_has_the_board_as_it_was_at_the_cut():
    _, g = run_board()
    poster = g.agents()[0]

    branch = g.fork(poster)

    assert TaskBoard.state(branch)["tasks"] == {}
    assert TaskBoard.state(g)["tasks"]["sim"]["status"] == "claimed"


def test_the_figure_draws_the_board_lane_from_the_saved_calls():
    _, g = run_board()

    assert ">board</text>" in graph_svg(g)
    assert ">board</text>" not in graph_svg(Graph())


def test_exactly_one_of_many_concurrent_claims_wins():
    g = Graph()
    workers = [g.start(f"worker {i}", name=f"w{i}") for i in range(8)]
    board = TaskBoard()
    with stepping(g, workers[0]):
        board.board_post("sim", "Write sim.js")
    wins = []

    def claim(agent):
        with stepping(g, agent):
            wins.append(board.board_claim("sim")["ok"])

    threads = [threading.Thread(target=claim, args=(w,)) for w in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert wins.count(True) == 1
