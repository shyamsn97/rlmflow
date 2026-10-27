"""Deterministic fake model for smoke tests."""

from __future__ import annotations

import re

from benchmarks.eval import model
from benchmarks.eval.types import Model


@model("fake")
class FakeModel(Model):
    provider = "fake"

    def __init__(self, name: str = "fake", **_: object) -> None:
        self.name = name
        self._usage = {"input_tokens": 0, "output_tokens": 0}

    def complete(self, messages: list[dict[str, str]], **kwargs) -> str:
        del kwargs
        prompt = "\n\n".join(message.get("content", "") for message in messages)
        self._usage = {
            "input_tokens": max(1, len(prompt) // 4),
            "output_tokens": 32,
        }
        answer = _extract_synthetic_secret(prompt)
        wants_repl = "```repl" in prompt or "submit(" in prompt or "Python REPL" in prompt
        # The needle haystack sits in the `observation` JSON argument; print
        # only the record after the marker so output truncation keeps it.
        marker = re.search(r"needle-marker-[0-9]{4}-[0-9]{8}", prompt)
        if wants_repl and answer == "UNKNOWN" and marker is not None:
            return (
                "```repl\nimport json\n"
                'haystack = json.loads(observation)["haystack"]\n'
                f'start = haystack.find("marker: {marker.group(0)}")\n'
                "print(haystack[start:start + 300])\n```"
            )
        if wants_repl:
            return f'```repl\nsubmit("{answer}")\n```'
        return answer

    def usage(self) -> dict[str, int]:
        return dict(self._usage)


def _extract_synthetic_secret(prompt: str) -> str:
    marker_match = re.search(r"needle-marker-[0-9]{4}-[0-9]{8}", prompt)
    if not marker_match:
        return "UNKNOWN"
    marker = marker_match.group(0)
    serialized = re.search(
        rf"marker: {re.escape(marker)}(?:\\n|\n).*?secret: (SECRET-[0-9-]+)",
        prompt,
    )
    if serialized is not None:
        return serialized.group(1)
    lines = prompt.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == f"marker: {marker}":
            for candidate in lines[index + 1 : index + 5]:
                if candidate.startswith("secret: "):
                    return candidate.split("secret: ", 1)[1].strip()
    return "UNKNOWN"


__all__ = ["FakeModel"]
