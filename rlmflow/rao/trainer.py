"""Turning scored trajectories into Tinker updates, and the RAO training loop.

The only module that needs the ``tinker`` extra. Tinker imports stay inside the
functions that use them, so importing this module costs nothing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rlmflow.graph.nodes import LLMUsage
from rlmflow.llm import retry_transient
from rlmflow.llm.tinker import TinkerClient
from rlmflow.rao.env import Env
from rlmflow.rao.export import Trajectory, stats, trajectories, write_jsonl
from rlmflow.rao.rollout import Budget, Collector, RolloutFlow, TaskSpec, TurnSample


class Optimizer(Protocol):
    """Policy-update boundary shared by CISPO, RLOO, PPO, and other objectives."""

    @property
    def sampler(self) -> Any:
        """The LLM client rollouts are sampled from."""

    def update(self, items: Sequence[Trajectory]) -> dict[str, Any]:
        """Apply one policy update and return metrics."""


# Backward-compatible name retained for the experimental API.
Trainer = Optimizer


# -- Sampling -------------------------------------------------------------


class TinkerSampler(TinkerClient):
    """A ``TinkerClient`` that hands back the tokens and logprobs it sampled.

    ``records_tokens`` is declared rather than sniffed from the signature:
    every client takes ``**kwargs``, so introspection would call all of them
    capable and silently produce empty trajectories.
    """

    records_tokens = True

    @retry_transient
    def completion(self, messages: list[dict[str, str]], *args, **kwargs) -> tuple[str, LLMUsage]:
        from tinker import types  # type: ignore[import-not-found]

        sink = kwargs.pop("sample_sink", None)
        prompt, renderer_stop = self._render_prompt(messages)
        stop = self.stop
        if stop is None:
            stop = renderer_stop
        params_kwargs = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stop": kwargs.get("stop", stop),
        }
        params = types.SamplingParams(
            **{key: value for key, value in params_kwargs.items() if value is not None}
        )
        future = self.sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=params,
        )
        output = self._future_result(future, timeout=kwargs.get("timeout"))
        sequence = first_sequence(output)
        tokens = as_list(getattr(sequence, "tokens", None))
        logprobs = [float(value) for value in as_list(getattr(sequence, "logprobs", None))]
        text = self._parse_tokens(tokens)
        usage = LLMUsage(input_tokens=len(prompt_ints(prompt)), output_tokens=len(tokens))
        self.last_usage = usage
        if sink is not None:
            # Appended last, so a retried attempt cannot leave a partial record.
            sink.append(
                TurnSample(
                    prompt_tokens=prompt_ints(prompt),
                    sampled_tokens=tokens,
                    logprobs=logprobs,
                    stop_reason=str(getattr(sequence, "stop_reason", "") or ""),
                )
            )
        return text, usage


def prompt_ints(prompt: Any) -> list[int]:
    """The prompt as token ids, whichever shape the renderer handed back."""
    for name in ("to_ints", "tokens"):
        value = getattr(prompt, name, None)
        if callable(value):
            return as_list(value())
        if value is not None:
            return as_list(value)
    return as_list(prompt)


def first_sequence(output: Any) -> Any:
    sequences = getattr(output, "sequences", None)
    if sequences is None and isinstance(output, dict):
        sequences = output.get("sequences")
    if isinstance(sequences, (list, tuple)):
        return sequences[0]
    return sequences if sequences is not None else output


def as_list(value: Any) -> list:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    return list(tolist() if callable(tolist) else value)


# -- Datums ---------------------------------------------------------------


def arrays(turn: TurnSample, credit: float) -> dict[str, list]:
    """One turn as next-token-prediction arrays, masked to the sampled tokens.

    Targets are the input shifted by one, so position ``i`` predicts
    ``tokens[i + 1]``. The sampled tokens begin at ``len(prompt_tokens)``, which
    puts the first trainable position at ``len(prompt_tokens) - 1``.
    """
    if not turn.prompt_tokens:
        raise ValueError("a sampled turn must include at least one prompt token")
    if len(turn.sampled_tokens) != len(turn.logprobs):
        raise ValueError(
            "sampled token/logprob mismatch: "
            f"{len(turn.sampled_tokens)} tokens, {len(turn.logprobs)} logprobs"
        )
    tokens = [*turn.prompt_tokens, *turn.sampled_tokens]
    start = len(turn.prompt_tokens) - 1
    inputs, targets = tokens[:-1], tokens[1:]
    mask = [1.0 if index >= start else 0.0 for index in range(len(targets))]
    logprobs = [
        turn.logprobs[index - start] if index >= start else 0.0 for index in range(len(targets))
    ]
    return {
        "input_tokens": inputs,
        "target_tokens": targets,
        "mask": mask,
        "logprobs": logprobs,
        "advantages": [credit if weight else 0.0 for weight in mask],
    }


def build_datums(
    items: Sequence[Trajectory],
    types: Any,
    *,
    advantage_scale: float = 1.0,
) -> list[Any]:
    """One datum per sampled turn, carrying that trajectory's credit.

    Every turn of one trajectory gets the same advantage: RAO's credit is
    trajectory-level, not an attribution to a particular token. Turns are not
    merged onto a shared prefix, which would save tokens but is an optimization,
    not a correctness matter.
    """
    datums = []
    for item in items:
        for turn in item.turns:
            data = arrays(turn, item.credit * advantage_scale)
            if not any(data["mask"]):
                continue
            data.pop("mask")
            inputs = data.pop("input_tokens")
            loss_inputs = {
                name: _tensor_data(types, values, integral=name == "target_tokens")
                for name, values in data.items()
            }
            datums.append(
                types.Datum(
                    model_input=types.ModelInput.from_ints(inputs),
                    loss_fn_inputs=loss_inputs,
                )
            )
    return datums


def _tensor_data(types: Any, values: list, *, integral: bool) -> Any:
    """Build Tinker's required tensor wrapper while keeping test doubles tiny."""
    tensor_type = getattr(types, "TensorData", None)
    if tensor_type is None:
        return values
    import torch

    dtype = torch.long if integral else torch.float32
    return tensor_type.from_torch(torch.tensor(values, dtype=dtype))


# -- Trainers -------------------------------------------------------------


@dataclass
class TinkerConfig:
    base_model: str = "Qwen/Qwen3-8B"
    state_path: str | None = None
    renderer: str = "qwen3"
    lora_rank: int = 32
    learning_rate: float = 1e-5
    max_tokens: int = 2048
    temperature: float = 1.0
    #: RLOO is the default critic-free objective. Tinker implements its policy
    #: gradient update as importance sampling over our leave-one-out advantages.
    loss_fn: str = "rloo"


def tinker_loss_fn(name: str) -> str:
    """Resolve public optimizer names to Tinker's token-loss identifiers."""
    normalized = name.lower().replace("-", "_")
    aliases = {
        "rloo": "importance_sampling",
        "reinforce_leave_one_out": "importance_sampling",
        "is": "importance_sampling",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in {"importance_sampling", "cispo", "ppo"}:
        raise ValueError("loss_fn must be one of rloo, importance_sampling, cispo, or ppo")
    return resolved


class TinkerTrainer:
    """Sample from Tinker, update through Tinker, then resample from new weights."""

    def __init__(self, config: TinkerConfig | None = None, *, service_client: Any = None) -> None:
        import tinker  # type: ignore[import-not-found]

        self.config = config or TinkerConfig()
        tinker_loss_fn(self.config.loss_fn)
        self.service = service_client or tinker.ServiceClient()
        if self.config.state_path:
            self.training = self.service.create_training_client_from_state_with_optimizer(
                path=self.config.state_path,
                base_model=self.config.base_model,
            )
        else:
            self.training = self.service.create_lora_training_client(
                base_model=self.config.base_model,
                rank=self.config.lora_rank,
            )
        self.steps = 0
        self._sampler: TinkerSampler | None = None

    @property
    def sampler(self) -> TinkerSampler:
        if self._sampler is None:
            self._sampler = self.build_sampler(self.training.save_weights_and_get_sampling_client)
        return self._sampler

    def build_sampler(self, factory: Callable[..., Any]) -> TinkerSampler:
        client = factory(name=f"rao-{self.steps:04d}")
        return TinkerSampler(
            sampling_client=client,
            renderer=self.config.renderer,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

    def update(self, items: Sequence[Trajectory]) -> dict[str, Any]:
        active_tokens = sum(
            len(turn.sampled_tokens) for item in items for turn in item.turns if turn.trainable
        )
        if not any(item.credit != 0.0 for item in items):
            return {
                "algorithm": self.config.loss_fn,
                "active_tokens": active_tokens,
                "datums": 0,
                "skipped": "zero policy-gradient signal",
                "step": self.steps,
            }
        if active_tokens == 0:
            return {
                "algorithm": self.config.loss_fn,
                "active_tokens": 0,
                "datums": 0,
                "skipped": "no trainable turns",
                "step": self.steps,
            }

        from tinker import types  # type: ignore[import-not-found]

        # Tinker policy objectives use sum reduction. Normalize by active action
        # tokens so update magnitude does not depend on response length or batch size.
        advantage_scale = 1.0 / active_tokens
        datums = build_datums(items, types, advantage_scale=advantage_scale)
        if not datums:
            return {
                "algorithm": self.config.loss_fn,
                "active_tokens": active_tokens,
                "datums": 0,
                "skipped": "no trainable turns",
                "step": self.steps,
            }
        objective = tinker_loss_fn(self.config.loss_fn)
        self.training.forward_backward(datums, loss_fn=objective).result()
        self.training.optim_step(types.AdamParams(learning_rate=self.config.learning_rate)).result()
        self.steps += 1
        # Resampling from the updated weights is what keeps the next batch
        # on-policy; the recorded logprobs are what keep the ratio correct if a
        # batch drifts anyway.
        self._sampler = self.build_sampler(self.training.save_weights_and_get_sampling_client)
        return {
            "algorithm": self.config.loss_fn,
            "active_tokens": active_tokens,
            "advantage_scale": advantage_scale,
            "datums": len(datums),
            "loss_fn": objective,
            "step": self.steps,
        }

    def save_checkpoint(self, name: str) -> dict[str, str]:
        """Persist inference weights and optimizer state under one logical name."""
        sampler = self.training.save_weights_for_sampler(f"{name}-sampler").result()
        state = self.training.save_state(f"{name}-state").result()
        return {"sampler": sampler.path, "state": state.path}


@dataclass
class JsonlTrainer:
    """Write examples and train nothing.

    Lets the whole pipeline — env sharing, the barrier, scoring, advantages,
    depth weights, datum shapes — be validated before spending a GPU on it.
    """

    sampler: Any
    types: Any = None

    def update(self, items: Sequence[Trajectory]) -> dict[str, Any]:
        if self.types is None:
            return {"datums": 0}
        return {"datums": len(build_datums(items, self.types))}


# -- Loop -----------------------------------------------------------------


@dataclass
class RunConfig:
    out_dir: Path = field(default_factory=lambda: Path("runs/rao"))
    iterations: int = 1
    tasks_per_iteration: int = 4
    rollouts_per_task: int = 4
    budget: Budget = field(default_factory=Budget)
    checkpoint_every: int | None = 1


async def train(
    tasks: Sequence[TaskSpec],
    open_env: Callable[[TaskSpec], Awaitable[Env]],
    trainer: Trainer,
    config: RunConfig | None = None,
    collector_kwargs: Mapping[str, Any] | None = None,
    **flow_kwargs: Any,
) -> list[dict[str, Any]]:
    """Run the RAO loop and return one metrics dict per iteration."""
    config = config or RunConfig()
    metrics: list[dict[str, Any]] = []
    for iteration in range(config.iterations):
        # Tinker's public trainer lifecycle is synchronous. Keep those calls off
        # the event loop; the SDK explicitly warns that calling them from async
        # code can deadlock.
        sampler = await asyncio.to_thread(lambda: trainer.sampler)
        flow = RolloutFlow(sampler, budget=config.budget, **flow_kwargs)
        collector = Collector(
            flow,
            open_env,
            rollouts_per_task=config.rollouts_per_task,
            **dict(collector_kwargs or {}),
        )
        start = (iteration * config.tasks_per_iteration) % max(len(tasks), 1)
        batch = [
            tasks[(start + offset) % len(tasks)] for offset in range(config.tasks_per_iteration)
        ]
        try:
            trees = await collector.collect(batch)
            items = trajectories(trees, flow)
            record = {
                "iteration": iteration,
                **stats(items, trees),
                "refusals": dict(flow.refusals),
            }
            record.update(await asyncio.to_thread(trainer.update, items))
            if (
                config.checkpoint_every
                and (iteration + 1) % config.checkpoint_every == 0
                and callable(save := getattr(trainer, "save_checkpoint", None))
            ):
                record["checkpoint"] = await asyncio.to_thread(save, f"rao-{iteration + 1:06d}")
            metrics.append(record)
            write_artifacts(config.out_dir, iteration, items, record)
        finally:
            await flow.aclose()
    return metrics


def write_artifacts(
    out_dir: str | Path,
    iteration: int,
    items: Sequence[Trajectory],
    record: dict[str, Any],
) -> Path:
    target = Path(out_dir) / "updates" / f"{iteration:06d}"
    write_jsonl(items, target / "examples.jsonl")
    (target / "metrics.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    return target


__all__ = [
    "JsonlTrainer",
    "Optimizer",
    "RunConfig",
    "TinkerConfig",
    "TinkerSampler",
    "TinkerTrainer",
    "Trainer",
    "arrays",
    "build_datums",
    "train",
    "tinker_loss_fn",
    "write_artifacts",
]
