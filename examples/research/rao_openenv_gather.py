"""A recursive rollout on a real OpenEnv environment, scored the RAO way.

Serves a small OpenEnv `Environment` on localhost, then lets one agent tree work
it: the root delegates gathering to subagents, every agent acts on the *same*
episode, and each is credited for the reward its own actions earned plus a bonus
for what its children earned.

No Docker and no Hub account needed. Point `--base-url` at any running OpenEnv
server (or use `open_env(hub_id=...)`) to run against a real one instead.

    python examples/research/rao_openenv_gather.py --model gpt-4o-mini

Needs the openenv extra: `pip install 'rlmflow[openenv]'`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
import time

from examples.common import add_model_args
from rlmflow.llm import client_for
from rlmflow.rao import Budget, Collector, RolloutFlow, TaskSpec
from rlmflow.rao.export import stats, trajectories

RECIPE = ("wood", "stone", "fiber")


def build_env_app():
    """A tiny OpenEnv environment: one reward the first time each item is gathered."""
    from openenv.core.env_server import (
        Action,
        Environment,
        Observation,
        State,
        create_app,
    )

    class GatherAction(Action):
        item: str = ""

    class GatherObservation(Observation):
        inventory: list[str] = []
        missing: list[str] = []

    class GatherEnv(Environment):
        # Lets every rollout's connection own its own episode of this env.
        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self):
            super().__init__()
            self.inventory: list[str] = []
            self.steps = 0

        def observe(self, reward: float = 0.0) -> GatherObservation:
            missing = [item for item in RECIPE if item not in self.inventory]
            return GatherObservation(
                inventory=list(self.inventory),
                missing=missing,
                reward=reward,
                done=not missing,
            )

        def reset(self, seed=None, episode_id=None, **kwargs):
            self.inventory, self.steps = [], 0
            return self.observe()

        def step(self, action: GatherAction) -> GatherObservation:
            self.steps += 1
            new = action.item in RECIPE and action.item not in self.inventory
            if new:
                self.inventory.append(action.item)
            return self.observe(1.0 if new else 0.0)

        def state(self) -> State:
            return State(episode_id="gather", step_count=self.steps)

    return create_app(GatherEnv, GatherAction, GatherObservation, max_concurrent_envs=8)


def serve() -> tuple[str, object]:
    """Run the env server on a loopback port in a background thread."""
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(build_env_app(), host="127.0.0.1", port=port, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("the OpenEnv server did not come up")
    return f"http://127.0.0.1:{port}", server


GOAL = f"""Gather every item in this recipe: {", ".join(RECIPE)}.

You share one environment with your subagents. `await env_step(item="wood")`
takes one action and returns {{"observation", "reward", "done"}}; the observation
lists what you still need. You are scored on the reward your own actions earn,
plus a bonus for what your subagents earn, so delegate one item per subagent
with `launch_subagent(...)` rather than gathering everything yourself.
Call `finish(...)` with a short summary when the recipe is complete."""


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser, default="gpt-4o-mini")
    parser.add_argument("--base-url", default=None, help="an existing OpenEnv server")
    parser.add_argument("--rollouts", type=int, default=2, help="rollouts of the task")
    args = parser.parse_args()

    server = None
    base_url = args.base_url
    if base_url is None:
        base_url, server = serve()
        print(f"serving a local OpenEnv gather env at {base_url}")

    from rlmflow.rao import open_env

    async def connect(_task: TaskSpec):
        """A fresh client per rollout, so each rollout gets its own episode."""
        return await open_env(base_url=base_url)

    flow = RolloutFlow(
        client_for(args.model),
        budget=Budget(max_depth=2, max_iters=8, max_children_per_agent=3, max_env_steps=24),
    )
    collector = Collector(flow, connect, rollouts_per_task=args.rollouts)

    try:
        trees = await collector.collect([TaskSpec(id="gather", goal=GOAL)])
        items = trajectories(trees, flow)
    finally:
        await flow.aclose()
        if server is not None:
            server.should_exit = True

    for tree in trees:
        print(f"\nrollout {tree.index}: {'ok' if tree.usable else tree.error}")
        for step in tree.session.steps:
            print(f"  {step.agent_id:<14} {step.action} -> {step.reward}")
        for agent_id, score in sorted(tree.scores.items()):
            print(
                f"  {agent_id:<14} local={score.local:.2f} "
                f"delegation={score.delegation:.2f} advantage={score.advantage:+.2f}"
            )

    print("\n" + json.dumps(stats(items), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
