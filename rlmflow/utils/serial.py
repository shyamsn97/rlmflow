"""Trusted host-to-worker value serialization.

The host may send arbitrary Python objects into an isolated worker. Those values
are copied with ``cloudpickle`` and are safe to deserialize only because the
host created them. Worker-to-host messages use the data-only JSON protocol and
must never call this module's decoder.
"""

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


__all__ = ["decode_host_value", "encode_host_value"]
