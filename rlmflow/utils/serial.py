"""Value serialization across worker and host-tool RPC boundaries.

Plain data crosses the wire as JSON (readable, no dependency). Arbitrary live
Python objects — custom class instances, closures, lambdas — cannot, so they are
shipped **by value** as a base64-encoded ``cloudpickle`` blob and rebuilt in the
sandbox as an independent copy (mutations there do NOT reflect back on the host
object unless it is retrieved again). ``cloudpickle`` is a core dependency
because every runtime uses this path for agent bindings and host-tool RPC.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import cloudpickle

#: Wire encodings for an injected value / retrieved result.
JSON = "json"
CLOUDPICKLE = "cloudpickle"


def cloudpickle_available() -> bool:
    """Whether the required serializer is available."""
    return True


def is_json_safe(value: Any) -> bool:
    """True when ``value`` round-trips through JSON as-is (no custom objects)."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def encode_object(value: Any) -> str:
    """Serialize any picklable object to a base64 ``cloudpickle`` string."""
    return base64.b64encode(cloudpickle.dumps(value)).decode("ascii")


def decode_object(blob: str) -> Any:
    """Inverse of :func:`encode_object` (rebuilds an independent copy)."""
    return cloudpickle.loads(base64.b64decode(blob))


__all__ = [
    "CLOUDPICKLE",
    "JSON",
    "cloudpickle_available",
    "decode_object",
    "encode_object",
    "is_json_safe",
]
