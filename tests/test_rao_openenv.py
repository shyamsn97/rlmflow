"""RAO against a real OpenEnv server, over the real WebSocket protocol.

Skipped unless the ``openenv`` extra is installed. Everything here talks to an
actual ``Environment`` served by ``create_app`` through a real
``GenericEnvClient``, so it catches the API drift that fakes cannot.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time

import pytest

pytest.importorskip("openenv", reason="needs the openenv extra")
pytest.importorskip("uvicorn", reason="needs uvicorn to serve a real env")

import uvicorn
from openenv import GenericEnvClient
from openenv.core.env_server import (
    Action,
    Environment,
    Observation,
    State,
    create_app,
)

from rlmflow.rao import (
    Budget,
    Collector,
    EnvSession,
    EpisodeOver,
    RolloutFlow,
    TaskSpec,
    open_env,
)
from rlmflow.rao.export import trajectories
from tests.helpers import first_user
from tests.test_rao import RecordingLLM, block


class GatherAction(Action):
    item: str = ""


class GatherObservation(Observation):
    inventory: list[str] = []
    message: str = ""


class GatherEnv(Environment):
    """Pays 1.0 the first time each needed item is gathered."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    NEEDED = ("wood", "stone")

    def __init__(self):
        super().__init__()
        self.inventory: list[str] = []
        self.steps = 0

    def reset(self, seed=None, episode_id=None, **kwargs):
        self.inventory, self.steps = [], 0
        return GatherObservation(
            inventory=[], message=f"gather {' and '.join(self.NEEDED)}", reward=0.0
        )

    def step(self, action: GatherAction) -> GatherObservation:
        self.steps += 1
        reward = 0.0
        if action.item in self.NEEDED and action.item not in self.inventory:
            self.inventory.append(action.item)
            reward = 1.0
        return GatherObservation(
            inventory=list(self.inventory),
            message=f"have {self.inventory}",
            reward=reward,
            done=len(self.inventory) == len(self.NEEDED),
        )

    def state(self) -> State:
        return State(episode_id="test", step_count=self.steps)


@pytest.fixture(scope="module")
def base_url():
    """Serve GatherEnv on a loopback port for the module's lifetime."""
    app = create_app(GatherEnv, GatherAction, GatherObservation, max_concurrent_envs=4)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "the OpenEnv server did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_a_real_openenv_client_satisfies_the_env_protocol(base_url):
    session = EnvSession(GenericEnvClient(base_url=base_url))

    async def main():
        observation = await session.reset()
        first = await session.act("root.w", item="wood")
        useless = await session.act("root", item="dirt")
        last = await session.act("root", item="stone")
        with pytest.raises(EpisodeOver):
            await session.act("root", item="stone")  # the env said done
        await session.close()
        return observation, first, useless, last

    observation, first, useless, last = asyncio.run(main())

    # A plain dict went out as the action and the server validated it.
    assert observation["message"] == "gather wood and stone"
    assert first.reward == pytest.approx(1.0)
    # Reward is Optional[float] on the wire; a no-op step must read as 0.0.
    assert useless.reward == pytest.approx(0.0)
    assert last.observation["inventory"] == ["wood", "stone"]
    assert last.done  # the env's own terminal condition, read off the result
    assert session.reward_for("root.w") == pytest.approx(1.0)
    assert session.reward_for("root") == pytest.approx(1.0)


def test_a_subagent_gathers_in_its_parents_real_env(base_url):
    """The whole point, end to end: two agents, one live episode."""

    def reply(messages):
        goal = first_user(messages)
        if "chop wood" in goal:
            return block("print(await env_step(item='wood'))\nfinish('got wood')")
        turns = sum(1 for message in messages if message["role"] == "assistant")
        if turns == 0:
            return block(
                "handle = await launch_subagent("
                "'chop wood', model='default', name='w')\n"
                "print(await handle.wait_for_result())"
            )
        return block("print(await env_step(item='stone'))\nfinish('built')")

    async def main():
        flow = RolloutFlow(RecordingLLM(reply), budget=Budget(max_depth=1, max_iters=5))

        async def open_env(_task):
            return GenericEnvClient(base_url=base_url)

        collector = Collector(flow, open_env, rollouts_per_task=1)
        try:
            trees = await collector.collect([TaskSpec(id="gather", goal="gather wood and stone")])
            return trees, trajectories(trees, flow)
        finally:
            await flow.aclose()

    trees, items = asyncio.run(main())
    tree = trees[0]

    assert tree.usable and tree.complete
    # One episode, both agents in it, each credited for its own actions.
    assert [step.action["item"] for step in tree.session.steps] == ["wood", "stone"]
    assert tree.session.actors() == {"root", "root.w"}
    assert tree.scores["root.w"].local == pytest.approx(1.0)
    assert tree.scores["root"].local == pytest.approx(1.0)
    assert tree.scores["root"].delegation == pytest.approx(1.0)
    assert tree.session.total_reward == pytest.approx(2.0)
    assert {item.agent_id for item in items} == {"root", "root.w"}


def test_open_env_connects_to_a_running_server(base_url):
    async def main():
        session = EnvSession(await open_env(base_url=base_url))
        step = await session.act("root", item="wood")
        await session.close()
        return step

    assert asyncio.run(main()).reward == pytest.approx(1.0)
    with pytest.raises(ValueError, match="base_url, hub_id, or docker_image"):
        asyncio.run(open_env())


def test_each_rollout_gets_its_own_episode_on_one_server(base_url):
    """Concurrent sessions are what let G rollouts share one served env."""

    async def main():
        sessions = [EnvSession(GenericEnvClient(base_url=base_url)) for _ in range(3)]
        for session in sessions:
            await session.reset()
        # Every rollout gathers the same first item; each must be paid for it.
        rewards = await asyncio.gather(
            *(session.act(f"root{index}", item="wood") for index, session in enumerate(sessions))
        )
        for session in sessions:
            await session.close()
        return rewards

    rewards = asyncio.run(main())

    assert [step.reward for step in rewards] == [pytest.approx(1.0)] * 3
