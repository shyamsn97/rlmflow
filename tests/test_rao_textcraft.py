import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.research.textcraft.dataset import generate_task, load_dataset
from examples.research.textcraft.env import (
    TextCraftEnv,
    open_textcraft,
    textcraft_graph_rewards,
    textcraft_local_scores,
)
from examples.research.textcraft.recipes import Recipe
from rlmflow import CODE, DONE, Graph, LLMUsage, Node
from rlmflow.llm import LLMClient
from rlmflow.rao import (
    Budget,
    EnvSession,
    JsonlTrainer,
    RolloutFlow,
    RolloutTree,
    RunConfig,
    TurnSample,
    read_jsonl,
    train,
)
from rlmflow.rao.trainer import (
    TinkerConfig,
    TinkerTrainer,
    arrays,
    build_datums,
    tinker_loss_fn,
)


def test_minecraft_shaped_recipe_counts_symbols():
    recipe = Recipe.from_minecraft(
        {
            "type": "minecraft:crafting_shaped",
            "pattern": ["# #", "###"],
            "key": {"#": {"item": "minecraft:acacia_planks"}},
            "result": {"item": "minecraft:acacia_boat"},
        },
        "acacia_boat.json",
    )

    assert recipe.result == "acacia_boat"
    assert recipe.ingredients == {"acacia_planks": 5}


def test_generated_dataset_is_deterministic_and_held_out():
    first = generate_task(7, depth=3)
    again = generate_task(7, depth=3)
    validation = load_dataset("validation", limit=2)

    assert first == again
    assert first.metadata["depth"] == 3
    assert all(task.metadata["seed"] >= 1000 for task in validation)


def test_gym_crafting_conserves_inventory_under_concurrent_steps():
    task = generate_task(3, depth=1, branching=1)
    target = next(iter(task.target_items))
    leaf = next(iter(task.initial_inventory))
    env = TextCraftEnv(task)
    session = EnvSession(env)

    async def exercise():
        await session.reset()
        return await asyncio.gather(
            session.act("root.a", op="craft", item=target, quantity=1),
            session.act("root.b", op="craft", item=target, quantity=1),
        )

    steps = asyncio.run(exercise())

    assert env.inventory[target] == 1
    assert env.inventory[leaf] == 0
    assert sum(bool(step.observation.get("produced")) for step in steps) == 1


def test_invalid_craft_does_not_consume_inventory():
    task = generate_task(4, depth=1, branching=2)
    target = next(iter(task.target_items))
    env = TextCraftEnv(task)
    env.reset()
    one_leaf = next(iter(env.inventory))
    env.inventory[one_leaf] = 0
    before = env.inventory.copy()

    observation, reward, terminated, truncated, _info = env.step(
        {"op": "craft", "item": target, "quantity": 1}
    )

    assert env.inventory == before
    assert "missing ingredients" in observation["message"]
    assert reward == 0.0
    assert not terminated
    assert not truncated


def test_child_finish_does_not_terminate_parent_episode():
    task = generate_task(12, depth=2, branching=1)
    target = next(iter(task.target_items))
    target_recipe = task.recipe_book.options(target)[0]
    intermediate, count = next(iter(target_recipe.ingredients.items()))
    root = EnvSession(TextCraftEnv(task), max_steps=4)

    async def exercise():
        await root.reset()
        child, observation = root.fork(
            f"Craft {count}x {intermediate}",
            {"target_items": {intermediate: count}},
        )
        await child.act("root.worker", op="craft", item=intermediate, quantity=count)
        child_finish = await child.act("root.worker", op="finish")
        assert child_finish.reward == 1.0
        assert child.done
        assert not root.done
        await root.act("root", op="craft", item=target, quantity=1)
        root_finish = await root.act("root", op="finish")
        return observation, child, root_finish

    observation, child, root_finish = asyncio.run(exercise())

    assert observation["target_items"] == {intermediate: count}
    assert child.env.inventory is root.env.inventory
    assert root_finish.reward == 1.0
    assert root.done
    assert [step.index for step in root.steps] == [0, 1, 2, 3]


def _tree(task_metadata: dict, child_targets: object):
    """A finished root with one attached ``worker`` child, in a new graph."""
    g = Graph()
    root = g.create_agent("root")
    root.annotate("rollout.task", task_metadata)
    child = g.spawn("child", name="worker", target_items=child_targets)
    g.attach(root, child)
    return g, root, child


def _record_steps(node, steps) -> None:
    for step in steps:
        node.annotate("environment.steps", step, append=True)


def test_restored_child_lazily_binds_an_independent_session():
    task = generate_task(13, depth=2, branching=1)
    target = next(iter(task.target_items))
    intermediate, count = next(iter(task.recipe_book.options(target)[0].ingredients.items()))
    flow = RolloutFlow(SimpleNamespace(), budget=Budget(max_depth=1))
    g = Graph()
    root = g.create_agent("root")
    child = g.spawn("restored child", name="worker", target_items={intermediate: count})
    g.attach(root, child)
    root_session = EnvSession(TextCraftEnv(task))

    async def exercise():
        await root_session.reset()
        flow.bind(root, root_session)
        try:
            first = flow.ensure_session(g, child)
            second = flow.ensure_session(g, child)
            return first, second
        finally:
            await flow.aclose()

    first, second = asyncio.run(exercise())

    assert first is second
    assert first is not root_session
    assert first.env.inventory is root_session.env.inventory


def test_textcraft_scores_child_from_subtree_provenance():
    task = generate_task(5, depth=1, branching=2)
    target = next(iter(task.target_items))
    g, root, child = _tree(
        {
            "id": task.id,
            "metadata": {
                "target_items": task.target_items,
                "initial_inventory": task.initial_inventory,
            },
        },
        {target: 1},
    )
    env = TextCraftEnv(task)
    session = EnvSession(env)

    async def exercise():
        await session.reset()
        await session.act("root.worker", op="craft", item=target, quantity=1)
        await session.act("root", op="finish")

    asyncio.run(exercise())
    _record_steps(
        g.frontier(root),
        [
            {
                "agent_id": step.agent_id,
                "action": step.action,
                "observation": step.observation,
                "reward": step.reward,
                "done": step.done,
                "index": step.index,
            }
            for step in session.steps
        ],
    )
    g.append(child, Node(DONE, result="crafted"))
    g.append(root, Node(DONE, result="done"))
    tree = RolloutTree(task.to_spec(), g, root, 0, session)

    assert textcraft_graph_rewards(g, root) == {"root": 1.0, "root.worker": 1.0}
    assert textcraft_local_scores(tree) == {"root": 1.0, "root.worker": 1.0}


def test_textcraft_rewards_follow_environment_index_not_graph_walk_order():
    g, root, child = _tree(
        {"metadata": {"target_items": {"final": 1}, "initial_inventory": {}}},
        {"middle": 1},
    )
    g.append(child, Node(DONE, result="crafted middle"))
    g.append(root, Node(DONE, result="crafted final"))
    # Root nodes precede child nodes in graph order, but the child acted first.
    # Reward reconstruction must honor the global environment index.
    _record_steps(
        g.frontier(root),
        [
            {
                "agent_id": "root",
                "action": {"op": "finish"},
                "observation": {"inventory": {"final": 1}},
                "reward": 1.0,
                "done": True,
                "index": 1,
            }
        ],
    )
    _record_steps(
        g.frontier(child),
        [
            {
                "agent_id": "root.worker",
                "action": {"op": "craft"},
                "observation": {"inventory": {"middle": 1}, "produced": {"middle": 1}},
                "reward": 0.0,
                "done": False,
                "index": 0,
            }
        ],
    )

    assert textcraft_graph_rewards(g, root) == {"root": 1.0, "root.worker": 1.0}


def test_textcraft_malformed_child_targets_score_zero_instead_of_crashing():
    g, root, child = _tree(
        {"metadata": {"target_items": {"target": 1}, "initial_inventory": {}}},
        {"item": "target"},
    )
    g.append(child, Node(DONE, result="invalid assignment"))
    g.append(root, Node(DONE, result="done"))

    assert textcraft_graph_rewards(g, root) == {"root": 0.0, "root.worker": 0.0}


def test_arrays_mask_prompts_and_validate_logprobs():
    data = arrays(TurnSample([10, 11], [12, 13], [-0.1, -0.2]), credit=0.75)

    assert data["input_tokens"] == [10, 11, 12]
    assert data["target_tokens"] == [11, 12, 13]
    assert data["mask"] == [0.0, 1.0, 1.0]
    assert data["advantages"] == [0.0, 0.75, 0.75]
    with pytest.raises(ValueError, match="mismatch"):
        arrays(TurnSample([10], [11], []), credit=1.0)


def test_build_datums_sends_policy_gradient_inputs():
    class ModelInput:
        @staticmethod
        def from_ints(values):
            return values

    class Datum:
        def __init__(self, *, model_input, loss_fn_inputs):
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs

    types = SimpleNamespace(ModelInput=ModelInput, Datum=Datum)
    item = SimpleNamespace(
        turns=[TurnSample([1], [2], [-0.1])],
        credit=1.0,
    )

    datum = build_datums([item], types, advantage_scale=0.25)[0]

    assert set(datum.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
    assert "weights" not in datum.loss_fn_inputs
    assert datum.loss_fn_inputs["advantages"] == [0.25]


def test_tinker_trainer_skips_zero_signal_before_importing_tinker():
    trainer = object.__new__(TinkerTrainer)
    trainer.config = TinkerConfig()
    trainer.steps = 3
    item = SimpleNamespace(
        turns=[TurnSample([1], [2], [-0.1])],
        credit=0.0,
    )

    result = trainer.update([item])

    assert result == {
        "algorithm": "rloo",
        "active_tokens": 1,
        "datums": 0,
        "skipped": "zero policy-gradient signal",
        "step": 3,
    }


def test_tinker_trainer_normalizes_advantages_by_action_tokens(monkeypatch):
    class ModelInput:
        @staticmethod
        def from_ints(values):
            return values

    class Datum:
        def __init__(self, *, model_input, loss_fn_inputs):
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs

    class Future:
        def result(self):
            return None

    class Training:
        def __init__(self):
            self.datums = None
            self.loss_fn = None

        def forward_backward(self, datums, *, loss_fn):
            self.datums = datums
            self.loss_fn = loss_fn
            return Future()

        def optim_step(self, _params):
            return Future()

        def save_weights_and_get_sampling_client(self, **_kwargs):
            return None

    fake_types = SimpleNamespace(
        ModelInput=ModelInput,
        Datum=Datum,
        AdamParams=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, "tinker", SimpleNamespace(types=fake_types))
    trainer = object.__new__(TinkerTrainer)
    trainer.config = TinkerConfig()
    trainer.training = Training()
    trainer.steps = 0
    trainer._sampler = None
    trainer.build_sampler = lambda _factory: "updated sampler"
    items = [
        SimpleNamespace(
            turns=[TurnSample([1], [2, 3], [-0.1, -0.2])],
            credit=1.0,
        ),
        SimpleNamespace(
            turns=[TurnSample([1], [4, 5], [-0.1, -0.2])],
            credit=-1.0,
        ),
    ]

    result = trainer.update(items)

    assert result["active_tokens"] == 4
    assert result["advantage_scale"] == 0.25
    assert trainer.training.datums[0].loss_fn_inputs["advantages"] == [0.25, 0.25]
    assert trainer.training.datums[1].loss_fn_inputs["advantages"] == [-0.25, -0.25]
    assert trainer.training.loss_fn == "importance_sampling"
    assert trainer.steps == 1


def test_rloo_is_the_default_tinker_objective():
    assert TinkerConfig().loss_fn == "rloo"
    assert tinker_loss_fn("rloo") == "importance_sampling"
    assert tinker_loss_fn("cispo") == "cispo"
    with pytest.raises(ValueError, match="loss_fn"):
        tinker_loss_fn("unknown")


class RecordingCraftLLM(LLMClient):
    records_tokens = True
    thread_safe = True

    def __init__(self, target: str):
        self.target = target
        self.last_usage = LLMUsage(input_tokens=4, output_tokens=2)

    def chat(self, messages, *args, **kwargs):
        text = "\n".join(message["content"] for message in messages)
        if "recipe(s) for" not in text:
            reply = f'```repl\nprint(await env_step(op="inspect", item="{self.target}"))\n```'
        else:
            reply = (
                "```repl\n"
                f'print(await env_step(op="craft", item="{self.target}", quantity=1))\n'
                'print(await env_step(op="finish"))\n'
                'submit("crafted")\n'
                "```"
            )
        sink = kwargs.get("sample_sink")
        if sink is not None:
            sink.append(TurnSample([1, 2], [3, 4], [-0.1, -0.1]))
        return reply


class RecordingRecursiveCraftLLM(LLMClient):
    records_tokens = True
    thread_safe = True

    def __init__(self, target: str, intermediate: str, count: int):
        self.target = target
        self.intermediate = intermediate
        self.count = count
        self.last_usage = LLMUsage(input_tokens=4, output_tokens=2)

    def chat(self, messages, *args, **kwargs):
        text = "\n".join(message["content"] for message in messages)
        first_user = next(message["content"] for message in messages if message["role"] == "user")
        if "Delegated intermediate" in first_user:
            reply = (
                "```repl\n"
                f'print(await env_step(op="craft", item="{self.intermediate}", '
                f"quantity={self.count}))\n"
                'print(await env_step(op="finish"))\n'
                'submit("crafted intermediate")\n'
                "```"
            )
        elif "crafted intermediate" not in text:
            reply = (
                "```repl\n"
                'worker = launch_agent("Delegated intermediate", '
                f'target_items={{"{self.intermediate}": {self.count}}})\n'
                "print(await worker)\n"
                "```"
            )
        else:
            reply = (
                "```repl\n"
                f'print(await env_step(op="craft", item="{self.target}", quantity=1))\n'
                'print(await env_step(op="finish"))\n'
                'submit("crafted final")\n'
                "```"
            )
        sink = kwargs.get("sample_sink")
        if sink is not None:
            sink.append(TurnSample([1, 2], [3, 4], [-0.1, -0.1]))
        return reply


def test_recursive_training_child_finishes_then_parent_continues(tmp_path: Path):
    task = generate_task(100, depth=2, branching=1)
    target = next(iter(task.target_items))
    intermediate, count = next(iter(task.recipe_book.options(target)[0].ingredients.items()))

    records = asyncio.run(
        train(
            [task.to_spec()],
            open_textcraft,
            JsonlTrainer(RecordingRecursiveCraftLLM(target, intermediate, count)),
            RunConfig(
                out_dir=tmp_path,
                iterations=1,
                tasks_per_iteration=1,
                rollouts_per_task=2,
                budget=Budget(max_depth=1, max_iters=4, max_env_steps=4),
                checkpoint_every=None,
            ),
            collector_kwargs={
                "delegation_bonus": 0.0,
                "reward_fn": textcraft_graph_rewards,
            },
        )
    )

    assert records[0]["usable_rollouts"] == 2
    assert records[0]["mean_root_reward"] == 1.0
    assert records[0]["success_by_depth"] == {"0": 1.0, "1": 1.0}
    assert records[0]["delegations"] == 2
    examples = read_jsonl(tmp_path / "updates" / "000000" / "examples.jsonl")
    assert all(example["metadata"]["env_steps"] == 2 for example in examples)
    assert all(example["metadata"]["episode_reward"] == 1.0 for example in examples)


def test_offline_grouped_training_smoke(tmp_path: Path):
    task = generate_task(99, depth=1, branching=2)
    target = next(iter(task.target_items))

    records = asyncio.run(
        train(
            [task.to_spec()],
            open_textcraft,
            JsonlTrainer(RecordingCraftLLM(target)),
            RunConfig(
                out_dir=tmp_path,
                iterations=1,
                tasks_per_iteration=1,
                rollouts_per_task=2,
                budget=Budget(max_depth=0, max_iters=4, max_env_steps=8),
                checkpoint_every=None,
            ),
            collector_kwargs={
                "delegation_bonus": 0.0,
                "reward_fn": textcraft_graph_rewards,
            },
        )
    )

    assert records[0]["usable_rollouts"] == 2
    assert records[0]["mean_root_reward"] == 1.0
    assert records[0]["trajectories"] == 2
    assert (tmp_path / "updates" / "000000" / "examples.jsonl").exists()


def test_delegation_bonus_credits_immediate_children_on_the_launcher_tree():
    from rlmflow.rao.credit import score_tree

    g = Graph()
    root = g.spawn("root", name="root")
    launch = g.append(root, Node(CODE, code="launch", output=""))
    good = g.spawn("good", by=launch, name="good")
    bad = g.spawn("bad", by=launch, name="bad")
    relaunch = g.append(good, Node(CODE, code="launch", output=""))
    leaf = g.spawn("leaf", by=relaunch, name="leaf")
    for agent in (leaf, good, bad, root):
        g.append(agent, Node(DONE, result="done"))

    local = {"root": 0.0, "root.good": 1.0, "root.bad": 0.0, "root.good.leaf": 1.0}
    scores = score_tree(g, root, local, delegation_bonus=0.4)

    assert scores["root"].delegation == 0.5
    assert scores["root"].reward == pytest.approx(0.2)
    assert scores["root.good"].reward == pytest.approx(1.4)
    assert scores["root.good.leaf"].depth == 2
    assert scores["root.bad"].reward == 0.0
