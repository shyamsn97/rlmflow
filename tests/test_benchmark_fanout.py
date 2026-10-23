import hashlib
import json

from benchmarks.eval.runners.prime_agent import _with_spawn_tasks
from benchmarks.eval.tasks.swarm_fanout import (
    SwarmFanoutDataset,
    fanout_payloads_for_example,
)


def test_full_fanout_matrix_is_frozen_and_varied():
    examples = SwarmFanoutDataset().examples(split="full", limit=None, seed=0)
    repeated = SwarmFanoutDataset().examples(split="full", limit=None, seed=999)

    assert len(examples) == 16
    assert [example.id for example in examples] == [example.id for example in repeated]
    assert {len(example.metadata["shards"]) for example in examples} == {4, 6, 8, 12, 16}

    first_hashes = [
        hashlib.sha256(
            json.dumps(
                fanout_payloads_for_example(example),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for example in examples
    ]
    repeated_hashes = [
        hashlib.sha256(
            json.dumps(
                fanout_payloads_for_example(example),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        for example in repeated
    ]
    assert first_hashes == repeated_hashes
    assert len(set(first_hashes)) == len(examples)


def test_prime_spawn_tasks_are_recovered_from_child_sessions(tmp_path):
    task = "exact child task"
    child = tmp_path / "child.jsonl"
    child.write_text(
        "\n".join(
            [
                json.dumps({"type": "session"}),
                json.dumps(
                    {
                        "type": "custom_message",
                        "customType": "agent_message",
                        "details": {"id": "spawn:child-1", "message": task},
                    }
                ),
            ]
        )
    )

    records = _with_spawn_tasks([{"op": "spawn", "childId": "child-1", "child": str(child)}])

    assert records[0]["task"] == task
