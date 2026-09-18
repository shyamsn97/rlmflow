"""Bounded, portable asset handling for the offline visual report."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MAX_EMBEDDED_PAYLOAD_BYTES = 4 * 1024 * 1024
MAX_EMBEDDED_REPLAY_BYTES = 512 * 1024


def write_static_assets(
    output_dir: Path,
    *,
    css: str,
    javascript: str,
) -> dict[str, str]:
    assets = output_dir / "viewer-assets"
    assets.mkdir(parents=True, exist_ok=True)
    css_path = assets / "report.css"
    javascript_path = assets / "report.js"
    css_path.write_text(css, encoding="utf-8")
    javascript_path.write_text(javascript, encoding="utf-8")
    return {
        "css": css_path.relative_to(output_dir).as_posix(),
        "javascript": javascript_path.relative_to(output_dir).as_posix(),
    }


def externalize_large_replays(
    payload: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Move large replay streams into bounded relative JSON files in-place."""
    replay_dir = output_dir / "replays"
    for task in payload.get("tasks", []):
        if not isinstance(task, dict):
            continue
        for lane in task.get("lanes", []):
            if not isinstance(lane, dict):
                continue
            replay = lane.get("replay")
            if not isinstance(replay, Mapping):
                continue
            encoded = _encoded(replay)
            if len(encoded) <= MAX_EMBEDDED_REPLAY_BYTES:
                continue
            replay_dir.mkdir(parents=True, exist_ok=True)
            identity = "\0".join(
                (
                    str(task.get("example_id", "")),
                    str(lane.get("runner", "")),
                    str(lane.get("seed", "")),
                )
            )
            digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
            path = replay_dir / f"{digest}.json"
            path.write_bytes(encoded + b"\n")
            lane["replay"] = {
                "kind": str(replay.get("kind", "replay")),
                "status": "external",
                "external_href": path.relative_to(output_dir).as_posix(),
                "bytes": len(encoded),
                "summary": _replay_summary(replay),
            }
    size = len(_encoded(payload))
    if size > MAX_EMBEDDED_PAYLOAD_BYTES:
        raise ValueError(
            f"visual report payload is {size} bytes; cap is {MAX_EMBEDDED_PAYLOAD_BYTES} bytes"
        )
    return payload


def _replay_summary(replay: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": replay.get("kind"),
        "instances": _length(replay.get("instances")),
        "events": _length(replay.get("events")),
        "frames": _length(replay.get("frames")),
        "steps": _length(replay.get("steps")),
    }


def _length(value: Any) -> int:
    return len(value) if isinstance(value, (list, dict)) else 0


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


__all__ = [
    "MAX_EMBEDDED_PAYLOAD_BYTES",
    "MAX_EMBEDDED_REPLAY_BYTES",
    "externalize_large_replays",
    "write_static_assets",
]
