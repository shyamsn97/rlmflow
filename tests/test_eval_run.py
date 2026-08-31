from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from benchmarks.eval.models.openai import OpenAIModel
from benchmarks.eval.types import (
    ComponentSpec,
    Example,
    ModelSpec,
    Prediction,
    Row,
    Score,
    SuiteConfig,
)

eval_run = importlib.import_module("benchmarks.eval.run")


class ExactDataset:
    def score(self, example: Example, prediction: Prediction) -> Score:
        correct = prediction.answer == example.expected
        return Score(value=float(correct), correct=correct)


def result(job_index: int, answer: str, delay: float) -> dict:
    time.sleep(delay)
    return {
        "run_id": "streaming",
        "dataset": "exact",
        "example_id": f"example-{job_index}",
        "runner": "fake",
        "model": "fake",
        "seed": 0,
        "prediction": asdict(Prediction(answer=answer)),
        "metadata": {},
        "job_index": job_index,
        "attempt": 0,
        "best_of_n": 1,
    }


def test_parallel_jobs_yield_rows_as_each_job_completes(monkeypatch, tmp_path):
    dataset = ExactDataset()
    jobs = [
        {
            "dataset": dataset,
            "example": Example(id="example-0", prompt="slow", expected="slow"),
            "payloads": [{"job_index": 0, "answer": "slow", "delay": 0.1}],
        },
        {
            "dataset": dataset,
            "example": Example(id="example-1", prompt="fast", expected="fast"),
            "payloads": [{"job_index": 1, "answer": "fast", "delay": 0.0}],
        },
    ]
    config = SuiteConfig(
        run_id="streaming",
        datasets=[],
        runners=[],
        model=ModelSpec(provider="fake", name="fake"),
        loggers=[],
        seeds=[0],
        output_root=tmp_path,
        parallelism=2,
    )
    monkeypatch.setattr(
        eval_run,
        "_run_job_payload",
        lambda payload: result(payload["job_index"], payload["answer"], payload["delay"]),
    )

    rows = iter(eval_run._run_jobs(config, jobs))

    assert next(rows).example_id == "example-1"
    assert next(rows).example_id == "example-0"


def test_interrupted_suite_keeps_completed_rows_and_marks_aborted(monkeypatch, tmp_path):
    config = SuiteConfig(
        run_id="interrupted",
        datasets=[],
        runners=[],
        model=ModelSpec(provider="fake", name="fake"),
        loggers=[ComponentSpec(name="jsonl")],
        seeds=[0],
        output_root=tmp_path,
    )
    example = Example(id="saved", prompt="save me")
    row = Row(
        run_id=config.run_id,
        dataset="exact",
        example_id=example.id,
        runner="fake",
        model="fake",
        seed=0,
        prediction=Prediction(answer="done"),
        score=Score(value=1.0, correct=True),
    )
    job = {"example": example, "payloads": [{"runner": {"name": "fake"}}]}
    monkeypatch.setattr(eval_run, "_build_jobs", lambda _config, seen: [job])

    def interrupted(_config, _jobs):
        yield row
        raise KeyboardInterrupt("stop")

    monkeypatch.setattr(eval_run, "_run_jobs", interrupted)

    with pytest.raises(KeyboardInterrupt, match="stop"):
        eval_run.run_suite(config)

    saved = eval_run.load_rows(config.root / "rows.jsonl")
    status = json.loads((config.root / "status.json").read_text())

    assert [item.example_id for item in saved] == ["saved"]
    assert status["state"] == "aborted"
    assert status["completed_rows"] == 1
    assert status["error"] == "KeyboardInterrupt: stop"


def test_openai_benchmark_model_uses_one_sync_client(monkeypatch):
    requests = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create),
            )

        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeClient))
    model = OpenAIModel(
        name="gpt-5-mini",
        api_key="test",
        reasoning_effort="medium",
    )

    assert model.complete([{"role": "user", "content": "first"}]) == "answer"
    assert model.complete([{"role": "user", "content": "second"}]) == "answer"
    assert model.usage() == {"input_tokens": 11, "output_tokens": 7}
    assert len(requests) == 2
    assert all(request["stream"] is False for request in requests)
    assert all(request["reasoning_effort"] == "medium" for request in requests)
    assert requests[0]["model"] == "gpt-5-mini"

    client = model._client
    model.close()

    assert client.closed
