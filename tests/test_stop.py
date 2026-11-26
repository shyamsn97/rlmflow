import asyncio

from helpers import ScriptedFlow, agent_graph, first_user
from helpers import repl as block

from rlmflow import Graph


class Parked:
    """Answers the parent with ``parent_code``; parks every child until it is cancelled."""

    def __init__(self, parent_code, children=1):
        self.parent_code = parent_code
        self.children = children
        self.started: set[str] = set()
        self.cancelled: set[str] = set()
        self.ready = asyncio.Event()

    async def chat(self, messages):
        query = first_user(messages)
        if query == "parent":
            return self.parent_code
        self.started.add(query)
        if len(self.started) == self.children:
            self.ready.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.add(query)
            raise


def test_stop_cancels_a_model_call_and_the_awaiting_parent_gets_the_reason():
    llm = Parked(block("child = launch_agent('child')\nsubmit(await child)"))
    flow = ScriptedFlow(llm)
    g, root = agent_graph("parent")

    async def drive():
        run = asyncio.ensure_future(flow.arun(g))
        await asyncio.wait_for(llm.ready.wait(), timeout=5)
        child = root.children()[0]
        ended = await flow.stop(child, reason="no gain in 4 evals")
        await asyncio.wait_for(run, timeout=5)
        return child, ended

    child, ended = asyncio.run(drive())
    assert llm.cancelled == {"child"}
    assert child.status() == "done"
    assert child.final_result() == {"stopped": "no gain in 4 evals"}
    assert [node.result for node in ended] == [{"stopped": "no gain in 4 evals"}]
    assert root.final_result() == {"stopped": "no gain in 4 evals"}


def test_stop_ends_the_subtree_unless_told_not_to():
    llm = Parked(block("a = launch_agent('a')\nb = launch_agent('b')\nawait a"), children=2)
    flow = ScriptedFlow(llm)

    async def drive(subtree):
        g, root = agent_graph("parent")
        run = asyncio.ensure_future(flow.arun(g))
        await asyncio.wait_for(llm.ready.wait(), timeout=5)
        await flow.stop(root, reason="enough", subtree=subtree)
        statuses = {a.name: a.status() for a in root.children()}
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        return root, statuses

    root, statuses = asyncio.run(drive(subtree=True))
    assert root.final_result() == {"stopped": "enough"}
    assert set(statuses.values()) == {"done"}

    llm.started.clear()
    llm.ready = asyncio.Event()
    root, statuses = asyncio.run(drive(subtree=False))
    assert root.final_result() == {"stopped": "enough"}
    assert set(statuses.values()) == {"running"}


def test_a_stopped_agent_survives_save_load_and_reopens_on_a_message(tmp_path):
    llm = Parked(block("print('working')"), children=0)
    flow = ScriptedFlow(llm)
    g, root = agent_graph("solo")

    async def stop_it():
        await flow.stop(root, reason="manual")

    asyncio.run(stop_it())
    assert root.final_result() == {"stopped": "manual"}

    loaded = Graph.load(g.save(tmp_path / "run"))
    again = loaded.agent_root(root.id)
    assert again.final_result() == {"stopped": "manual"}
    assert again.nodes()[-1].data["stopped"] == "manual"

    loaded.send(again, "[stopped earlier] please wrap up", sender="user")
    assert again.status() == "running"
