"""Small helpers shared across rlmflow: output truncation and host-value encoding."""

from __future__ import annotations

import base64
from typing import Any

import cloudpickle


def encode_host_value(value: Any) -> str:
    """Copy one host-controlled value into a worker payload."""
    return base64.b64encode(cloudpickle.dumps(value)).decode("ascii")


def decode_host_value(blob: str) -> Any:
    """Decode a host-created payload inside an isolated worker."""
    return cloudpickle.loads(base64.b64decode(blob))


def truncate_output(output: str, limit: int) -> str:
    """Keep the start and end of a long observation (0 disables truncation)."""
    if limit and len(output) > limit:
        omitted = len(output) - limit
        head = limit // 2
        tail = limit - head
        return (
            output[:head]
            + f"\n...<truncated {omitted} chars; keep full data in variables>...\n"
            + output[-tail:]
        )
    return output


__all__ = [
    "decode_host_value",
    "encode_host_value",
    "truncate_output",
]
