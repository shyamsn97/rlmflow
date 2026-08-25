"""RAO on OpenEnv: credit math on hand-built trees, and the rest on fakes."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from rlmflow import AgentStart, DoneOutput, ExecAction, start
from rlmflow.rao import (
    Budget,
    Collector,
    EnvSession,
    EpisodeOver,
    IncompleteTreeError,
    RolloutFlow,
    RolloutTree,
    TaskSpec,
    TurnSample,
    assign_advantages,
    assign_depth_weights,
    plain,
    read_jsonl,
    score_tree,
    stats,
    trajectories,
    write_jsonl,
)
from rlmflow.rao.trainer import JsonlTrainer, RunConfig, arrays, build_datums, train
from tests.helpers import first_user

# -- Fakes ----------------------------------------------------------------


class FakeEnv:
    """Pays out per action, and asserts it is never stepped by two agents at once."""

    def __init__(self, rewards=None, observation=None):
        self.rewards = rewards or {}
        self.observation = observation or {"goal": "collect wood"}
        self.actions = []
        self.inside = False
        self.closed = False
        self.resets = 0

    async def reset(self, **kwargs):
        self.resets += 1
        # Shaped like OpenEnv's StepResult, whose reward is None at reset.
        return SimpleNamespace(observation=self.observation, reward=None, done=False)

    async def step(self, action):
        assert not self.inside, "two agents stepped one env at the same time"
        self.inside = True
        try:
            await asyncio.sleep(0)  # let any racing sibling interleave if it can
            self.actions.append(action)
            return SimpleNamespace(
                observation={"taken": len(self.actions)},
                reward=self.rewards.get(action.get("item")),
                done=bool(action.get("stop")),
            )
        finally:
            self.inside = False

    async def close(self):
        self.closed = True


class RecordingLLM:
    """A client that hands back the tokens it 'sampled'."""

    records_tokens = True

    def __init__(self, reply, tokens=(4, 5)):
        self.reply = reply
        self.tokens = list(tokens)
        self.calls = 0

    def chat(self, messages, sample_sink=None, **_kwargs):
        self.calls += 1
        text = self.reply(messages) if callable(self.reply) else self.reply
        if sample_sink is not None:
            sample_sink.append(
                TurnSample(
                    prompt_tokens=[1, 2, 3],
                    sampled_tokens=list(self.tokens),
                    logprobs=[-0.5] * len(self.tokens),
                    stop_reason="stop",
                )
            )
        return text


class BareLLM:
    """A client that cannot report tokens, so its trajectories are untrainable."""

    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, **_kwargs):
        return self.reply(messages) if callable(self.reply) else self.reply


class FakeTypes:
    """Enough of ``tinker.types`` to check datum shapes without the SDK."""

    class ModelInput:
        def __init__(self, tokens):
            self.tokens = list(tokens)

        @classmethod
        def from_ints(cls, tokens):
            return cls(tokens)

    class Datum:
        def __init__(self, model_input, loss_fn_inputs):
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs


def block(code):
    return f"thinking\n```python\n{code}\n```"


# -- Hand-built trees for the credit math ---------------------------------


def tree(query="build a house", **overrides):
    return start(query, max_depth=3, **overrides)


def child_of(agent, name, goal="sub goal"):
    action = agent.frontier.append(
        ExecAction(code=f"launch_subagent({goal!r}, model='default', name={name!r})")
    )
    return action.append(AgentStart(content=goal, config=agent.config.child(name)))


def finish(agent, result="ok"):
    agent.frontier.append(DoneOutput(result=result))
    return agent


def scores_of(root, local, **kwargs):
    return score_tree(root, local, **kwargs)


def test_delegation_bonus_is_the_mean_of_immediate_children():
    root = tree()
    first = child_of(root, "a")
    second = child_of(root, "b")
    for agent in (first, second, root):
        finish(agent)

    scores = scores_of(root, {"root": 0.8, "root.a": 1.0, "root.b": 0.5}, delegation_bonus=0.4)

    assert scores["root"].delegation == pytest.approx(0.75)
    assert scores["root"].reward == pytest.approx(0.8 + 0.4 * 0.75)
    # A leaf has no delegation term at all, rather than a zero-mean one.
    assert scores["root.a"].delegation == 0.0
    assert scores["root.a"].reward == pytest.approx(1.0)


def test_a_grandchild_credits_its_parent_but_not_its_grandparent():
    root = tree()
    middle = child_of(root, "a")
    leaf = child_of(middle, "b")
    for agent in (leaf, middle, root):
        finish(agent)

    scores = scores_of(root, {"root": 0.0, "root.a": 0.0, "root.a.b": 1.0}, delegation_bonus=0.5)

    assert scores["root.a"].reward == pytest.approx(0.5)
    # The bonus uses each child's *local* score, so it does not compound upward.
    assert scores["root"].reward == pytest.approx(0.0)


def test_scoring_refuses_a_tree_with_an_agent_still_running():
    root = tree()
    child_of(root, "a")
    finish(root)

    with pytest.raises(IncompleteTreeError, match="root.a"):
        scores_of(root, {"root": 1.0, "root.a": 1.0})


def test_scoring_refuses_a_missing_local_score():
    root = tree()
    finish(child_of(root, "a"))
    finish(root)

    with pytest.raises(KeyError, match="root.a"):
        scores_of(root, {"root": 1.0})


def test_advantages_match_the_papers_worked_example():
    root = tree()
    first, second = child_of(root, "a"), child_of(root, "b")
    for agent in (first, second, root):
        finish(agent)
    scored = scores_of(root, {"root": 0.8, "root.a": 1.0, "root.b": 0.5}, delegation_bonus=0.4)

    others = []
    for reward in (0.4, 0.7, 1.0):
        sibling = finish(tree())
        others.append(("root", scores_of(sibling, {"root": reward})))

    assign_advantages([("root", scored), *others])

    # baseline for the first tree is mean(0.4, 0.7, 1.0) = 0.7
    assert scored["root"].advantage == pytest.approx(0.4)
    assert scored["root.a"].advantage == pytest.approx(0.3)
    # A weak child goes negative under a root that succeeded.
    assert scored["root.b"].advantage == pytest.approx(-0.2)


def test_one_rollout_has_no_baseline_and_so_no_gradient():
    scored = scores_of(finish(tree()), {"root": 1.0})

    assign_advantages([("root", scored)])

    assert scored["root"].advantage == 0.0


def test_depth_weights_equalize_each_depths_total_pull():
    batch = []
    for index in range(4):
        root = tree()
        kids = [child_of(root, f"c{n}") for n in range(2)]
        for agent in (*kids, root):
            finish(agent)
        local = {"root": 1.0, "root.c0": 1.0, "root.c1": 1.0}
        batch.append(scores_of(root, local))
        assert index >= 0

    weights = assign_depth_weights(batch)

    # 4 roots and 8 children, so the average depth holds 6 trajectories.
    assert weights[0] == pytest.approx(6 / 4)
    assert weights[1] == pytest.approx(6 / 8)
    assert batch[0]["root"].weight == pytest.approx(1.5)
    assert batch[0]["root.c0"].weight == pytest.approx(0.75)


def test_credit_is_advantage_times_weight():
    scored = scores_of(finish(tree()), {"root": 1.0})
    scored["root"].advantage, scored["root"].weight = 0.5, 2.0

    assert scored["root"].credit == pytest.approx(1.0)


# -- The env session ------------------------------------------------------


def test_reward_is_attributed_to_the_agent_that_acted():
    env = FakeEnv(rewards={"wood": 1.0, "plank": 0.25})
    session = EnvSession(env)

    async def main():
        await session.reset()
        await session.act("root.a", item="wood")
        await session.act("root", item="plank")
        await session.act("root", item="nothing")

    asyncio.run(main())

    assert session.reward_for("root.a") == pytest.approx(1.0)
    assert session.reward_for("root") == pytest.approx(0.25)
    assert session.reward_for("root.b") == 0.0
    assert session.total_reward == pytest.approx(1.25)
    assert session.actors() == {"root", "root.a"}


def test_concurrent_agents_serialize_on_one_env():
    env = FakeEnv(rewards={"wood": 1.0})
    session = EnvSession(env)

    async def main():
        await session.reset()
        await asyncio.gather(*(session.act(f"root.c{n}", item="wood") for n in range(8)))

    asyncio.run(main())

    # FakeEnv.step asserts non-reentrancy, so arriving here is the assertion.
    assert len(env.actions) == 8
    assert [step.index for step in session.steps] == list(range(8))


def test_env_steps_are_capped_per_rollout():
    session = EnvSession(FakeEnv(), max_steps=2)

    async def main():
        await session.reset()
        await session.act("root", item="wood")
        await session.act("root", item="wood")
        with pytest.raises(EpisodeOver, match="capped at 2"):
            await session.act("root", item="wood")

    asyncio.run(main())
    assert len(session.steps) == 2


def test_acting_after_the_episode_ends_is_refused():
    session = EnvSession(FakeEnv())

    async def main():
        await session.reset()
        await session.act("root", stop=True)
        with pytest.raises(EpisodeOver, match="over"):
            await session.act("root", item="wood")

    asyncio.run(main())
    assert session.done


def test_reset_clears_a_previous_episodes_steps():
    session = EnvSession(FakeEnv(rewards={"wood": 1.0}))

    async def main():
        await session.reset()
        await session.act("root", item="wood")
        await session.reset()

    asyncio.run(main())

    assert session.steps == []
    assert session.reward_for("root") == 0.0


def test_observations_are_reduced_to_plain_data():
    from dataclasses import dataclass

    @dataclass
    class Observation:
        text: str

    class Model:
        def model_dump(self):
            return {"text": "pydantic"}

    assert plain(Observation(text="dataclass")) == {"text": "dataclass"}
    assert plain(Model()) == {"text": "pydantic"}
    assert plain({"text": "plain"}) == {"text": "plain"}


def test_actions_go_out_as_dicts_unless_a_typed_action_is_given():
    """Dicts are what GenericEnvClient takes; typed clients need their Action."""

    class Action:
        def __init__(self, **fields):
            self.fields = fields

    untyped, typed = EnvSession(FakeEnv()), EnvSession(FakeEnv(), action_cls=Action)

    assert untyped.action(item="wood") == {"item": "wood"}
    built = typed.action(item="wood")
    assert isinstance(built, Action)
    assert built.fields == {"item": "wood"}


def test_a_client_that_returns_values_synchronously_still_works():
    """OpenEnv's reset/step hand back an object that need not be a coroutine."""

    class SyncClient:
        def __init__(self):
            self.closed = False

        def reset(self, **kwargs):
            return SimpleNamespace(observation={"ready": True}, reward=None, done=False)

        def step(self, action):
            return SimpleNamespace(observation={"ok": True}, reward=1.0, done=False)

        def close(self):
            self.closed = True

    client = SyncClient()
    session = EnvSession(client)

    async def main():
        assert await session.reset() == {"ready": True}
        step = await session.act("root", message="hi")
        await session.close()
        return step

    step = asyncio.run(main())

    assert step.reward == pytest.approx(1.0)
    assert client.closed


# -- Rollouts ------------------------------------------------------------


def delegating(child_goal="chop wood", child_item="wood", root_item="plank"):
    """Root launches one child, waits for it, then acts itself."""

    def reply(messages):
        goal = first_user(messages)
        if child_goal in goal:
            return block(f"print(await env_step(item={child_item!r}))\nfinish('chopped')")
        turns = sum(1 for message in messages if message["role"] == "assistant")
        if turns == 0:
            return block(
                f"handle = await launch_subagent("
                f"{child_goal!r}, model='default', name='w')\n"
                "print(await handle.wait_for_result())"
            )
        return block(f"print(await env_step(item={root_item!r}))\nfinish('built')")

    return reply


def run_collect(llm, envs, *, budget=None, rollouts=1, tasks=None):
    async def main():
        flow = RolloutFlow(llm, budget=budget or Budget(max_depth=1, max_iters=5))
        pending = list(envs)

        async def open_env(_task):
            return pending.pop(0)

        collector = Collector(flow, open_env, rollouts_per_task=rollouts)
        try:
            trees = await collector.collect(tasks or [TaskSpec(id="t0", goal="build a house")])
        finally:
            await flow.aclose()
        return flow, trees

    return asyncio.run(main())


def test_a_subagent_acts_on_its_parents_environment():
    env = FakeEnv(rewards={"wood": 1.0, "plank": 0.5})

    flow, trees = run_collect(RecordingLLM(delegating()), [env])
    tree_one = trees[0]

    # One env, one step log, both agents in it: the shared world is the point.
    assert env.resets == 1
    assert [action["item"] for action in env.actions] == ["wood", "plank"]
    assert tree_one.session.actors() == {"root", "root.w"}
    assert tree_one.complete and tree_one.usable
    assert tree_one.scores["root.w"].local == pytest.approx(1.0)
    assert tree_one.scores["root"].local == pytest.approx(0.5)
    assert tree_one.scores["root"].delegation == pytest.approx(1.0)
    assert env.closed


def test_the_initial_observation_reaches_the_root_as_inputs():
    env = FakeEnv(observation={"goal": "collect wood"})

    _flow, trees = run_collect(
        RecordingLLM(lambda _messages: block("finish('done')")),
        [env],
        budget=Budget(max_depth=0, max_iters=2),
    )

    assert json.loads(trees[0].root.config.inputs["observation"]) == {"goal": "collect wood"}


def test_one_turn_sample_is_recorded_per_model_turn_in_transcript_order():
    llm = RecordingLLM(delegating())

    flow, trees = run_collect(llm, [FakeEnv(rewards={"wood": 1.0})])
    root = trees[0].root
    child = root.sub_agents[0]

    assert len(flow.turns(root)) == 2  # launch, then act
    assert len(flow.turns(child)) == 1
    assert len(flow.samples) == llm.calls == 3
    assert all(turn.trainable for turn in flow.turns(root))


def test_a_client_that_cannot_report_tokens_records_nothing():
    flow, trees = run_collect(
        BareLLM(lambda _messages: block("finish('done')")),
        [FakeEnv()],
        budget=Budget(max_depth=0, max_iters=2),
    )

    assert trees[0].usable  # it still scores; it just cannot be trained on
    assert flow.samples == {}
    assert trajectories(trees, flow) == []


def test_children_are_capped_per_agent():
    def reply(messages):
        if first_user(messages).startswith("sub "):
            return block("finish('sub done')")
        return block(
            "for index in range(4):\n"
            "    try:\n"
            "        await launch_subagent("
            "'sub %d' % index, model='default', name='c%d' % index)\n"
            "    except Exception as error:\n"
            "        print('refused', error)\n"
            "finish('capped')"
        )

    flow, trees = run_collect(
        RecordingLLM(reply),
        [FakeEnv()],
        budget=Budget(max_depth=1, max_iters=4, max_children_per_agent=2),
    )

    assert len(trees[0].root.sub_agents) == 2
    assert flow.refusals["max_children_per_agent"] == 2


def test_total_agents_are_capped_per_rollout():
    def reply(messages):
        if first_user(messages).startswith("sub "):
            return block("finish('sub done')")
        return block(
            "for index in range(3):\n"
            "    try:\n"
            "        await launch_subagent("
            "'sub %d' % index, model='default', name='c%d' % index)\n"
            "    except Exception as error:\n"
            "        print('refused', error)\n"
            "finish('capped')"
        )

    flow, trees = run_collect(
        RecordingLLM(reply),
        [FakeEnv()],
        budget=Budget(max_depth=1, max_iters=4, max_total_agents=2),
    )

    assert len(trees[0].agents) == 2  # the root and one child
    assert flow.refusals["max_total_agents"] == 2


def test_a_tree_with_an_agent_still_running_is_dropped_whole():
    session = EnvSession(FakeEnv())
    root = tree()
    child_of(root, "a")  # launched, never terminal
    finish(root)
    unfinished = RolloutTree(task=TaskSpec(id="t", goal="g"), root=root, index=0, session=session)

    collector = Collector(None, None, rollouts_per_task=1)
    collector.score(unfinished)

    assert root.terminal and not unfinished.complete
    assert unfinished.error == "incomplete"
    assert unfinished.scores == {}
    assert not unfinished.usable


def test_collect_scores_every_rollout_and_assigns_batch_credit():
    envs = [FakeEnv(rewards={"wood": 1.0, "plank": 0.5}) for _ in range(2)]

    flow, trees = run_collect(RecordingLLM(delegating()), envs, rollouts=2)

    assert len(trees) == 2
    assert all(tree_.usable for tree_ in trees)
    # Identical rollouts, so each root's leave-one-out baseline is the other's
    # reward and the roots wash out; the children do not.
    assert all(tree_.scores["root"].advantage == pytest.approx(0.0) for tree_ in trees)
    assert trees[0].scores["root.w"].advantage == pytest.approx(1.0 - 0.9)
    # Two roots and two children: every depth holds the same count.
    assert all(score.weight == pytest.approx(1.0) for score in trees[0].scores.values())


# -- Export --------------------------------------------------------------


def test_one_trajectory_per_agent_round_trips_through_jsonl(tmp_path):
    flow, trees = run_collect(RecordingLLM(delegating()), [FakeEnv(rewards={"wood": 1.0})])

    items = trajectories(trees, flow)
    path = write_jsonl(items, tmp_path / "examples.jsonl")
    rows = read_jsonl(path)

    assert [item.agent_id for item in items] == ["root", "root.w"]
    assert len(rows) == 2
    assert rows[0]["rollout_id"] == "t0/g0"
    assert rows[0]["score"]["reward"] == pytest.approx(items[0].score.reward)
    assert rows[1]["goal"] == "chop wood"
    assert rows[1]["metadata"]["result"] == "chopped"


def test_stats_report_what_was_dropped_and_why():
    flow, trees = run_collect(RecordingLLM(delegating()), [FakeEnv(rewards={"wood": 1.0})])
    trees.append(
        RolloutTree(
            task=trees[0].task,
            root=tree(),
            index=1,
            session=trees[0].session,
            error="incomplete",
        )
    )

    report = stats(trajectories(trees, flow), trees)

    assert report["trajectories"] == 2
    assert report["rollouts"] == 2
    assert report["usable_rollouts"] == 1
    assert report["dropped"] == {"incomplete": 1}
    assert report["by_depth"] == {0: 1, 1: 1}
    assert report["sampled_tokens"] == 6  # three turns of two tokens


# -- Datums --------------------------------------------------------------


def test_arrays_shift_targets_and_train_only_the_sampled_tokens():
    turn = TurnSample(
        prompt_tokens=[10, 11, 12],
        sampled_tokens=[20, 21],
        logprobs=[-0.5, -1.5],
    )

    data = arrays(turn, credit=2.0)

    assert data["input_tokens"] == [10, 11, 12, 20]
    assert data["target_tokens"] == [11, 12, 20, 21]
    # Position 2 predicts the first sampled token, so that is where training starts.
    assert data["weights"] == [0.0, 0.0, 1.0, 1.0]
    assert data["logprobs"] == [0.0, 0.0, -0.5, -1.5]
    assert data["advantages"] == [0.0, 0.0, 2.0, 2.0]
    assert len(data["target_tokens"]) == len(data["weights"]) == len(data["logprobs"])


def test_every_turn_of_a_trajectory_carries_the_same_credit():
    flow, trees = run_collect(RecordingLLM(delegating()), [FakeEnv(rewards={"wood": 1.0})])
    items = trajectories(trees, flow)
    root = next(item for item in items if item.agent_id == "root")

    datums = build_datums([root], FakeTypes)

    assert len(datums) == 2  # one per sampled turn
    credits = {tuple(datum.loss_fn_inputs["advantages"]) for datum in datums}
    assert len(credits) == 1
    assert datums[0].model_input.tokens == [1, 2, 3, 4]


def test_a_turn_with_nothing_sampled_produces_no_datum():
    empty = TurnSample(prompt_tokens=[1, 2], sampled_tokens=[], logprobs=[])
    scored = score_tree(finish(tree()), {"root": 1.0})
    from rlmflow.rao.export import Trajectory

    item = Trajectory(
        task_id="t",
        rollout_id="t/g0",
        agent_id="root",
        depth=0,
        goal="g",
        turns=[empty],
        score=scored["root"],
    )

    assert build_datums([item], FakeTypes) == []


# -- The loop ------------------------------------------------------------


def test_train_writes_examples_and_metrics_per_iteration(tmp_path):
    envs = [FakeEnv(rewards={"wood": 1.0, "plank": 0.5}) for _ in range(2)]
    llm = RecordingLLM(delegating())

    async def open_env(_task):
        return envs.pop(0)

    metrics = asyncio.run(
        train(
            [TaskSpec(id="t0", goal="build a house")],
            open_env,
            JsonlTrainer(sampler=llm, types=FakeTypes),
            RunConfig(
                out_dir=tmp_path,
                iterations=1,
                tasks_per_iteration=1,
                rollouts_per_task=2,
                budget=Budget(max_depth=1, max_iters=5),
            ),
        )
    )

    update = tmp_path / "updates" / "000000"
    record = json.loads((update / "metrics.json").read_text())

    assert len(metrics) == 1
    assert metrics[0]["usable_rollouts"] == 2
    assert metrics[0]["trajectories"] == 4  # two rollouts, two agents each
    assert metrics[0]["datums"] == 6  # three model turns per rollout
    assert record["mean_root_reward"] == pytest.approx(0.5 + 0.4 * 1.0)
    assert len(read_jsonl(update / "examples.jsonl")) == 4
