"""What ``rlmflow run`` needs to build a Flow, and where each value came from.

Flags beat environment beats ``./rlmflow.toml`` beats
``~/.config/rlmflow/config.toml`` beats the defaults below. Which model your
terminal reaches for is a per-machine preference, not library configuration, so
it is worth a file; anything richer than "which model, which runtime, which
tools" stays in Python where a ``Flow(...)`` can express it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_NAME = "rlmflow.toml"
USER_CONFIG = Path("~/.config/rlmflow/config.toml")
ENV_PREFIX = "RLMFLOW_"


class CliError(Exception):
    """A failure the user can act on. The entry point prints it and exits 1."""


@dataclass(frozen=True)
class RunOptions:
    """One resolved configuration for a ``run``."""

    model: str = "gpt-5"
    fast_model: str = "gpt-5-mini"
    reasoning_effort: str | None = None
    workdir: str = "."
    docker_image: str | None = None
    max_depth: int = 3
    max_iters: int = 30
    workers: int = 8
    tools: str = "files"

    def __post_init__(self) -> None:
        if self.tools not in ("files", "none"):
            raise CliError(f"tools must be 'files' or 'none', not {self.tools!r}")


# Keyed by field name; the value is the default, which is also how ``_coerce``
# learns whether a string from a file or the environment needs to become an int.
DEFAULTS = {field.name: field.default for field in fields(RunOptions)}


def env_var(name: str) -> str:
    """``max_iters`` -> ``RLMFLOW_MAX_ITERS``."""
    return f"{ENV_PREFIX}{name.upper()}"


def resolve(
    overrides: dict[str, Any] | None = None,
    *,
    cwd: Path | None = None,
    home: Path | None = None,
    environ: dict[str, str] | None = None,
) -> tuple[RunOptions, dict[str, str]]:
    """Merge every source into one ``RunOptions``, plus a field -> source map.

    The map is what ``rlmflow config`` prints: with four places a model name can
    come from, "which one won" is the only question worth answering.
    """
    environ = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else cwd
    home = Path.home() if home is None else home

    values: dict[str, Any] = {}
    origins: dict[str, str] = dict.fromkeys(DEFAULTS, "default")

    layers = [
        ("user config", read_toml(user_config(home))),
        ("project config", read_toml(cwd / CONFIG_NAME)),
        ("env", _from_env(environ)),
        ("flag", {k: v for k, v in (overrides or {}).items() if v is not None}),
    ]
    for source, layer in layers:
        for name, value in layer.items():
            if name not in DEFAULTS:
                raise CliError(f"unknown {source} setting {name!r} (see `rlmflow config`)")
            values[name] = _coerce(name, value)
            origins[name] = source

    return RunOptions(**values), origins


def read_toml(path: Path) -> dict[str, Any]:
    """The ``[run]`` table of ``path``, or nothing if it isn't there."""
    import tomllib

    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc
    table = data.get("run", {})
    if not isinstance(table, dict):
        raise CliError(f"{path}: [run] must be a table")
    return table


def user_config(home: Path | None = None) -> Path:
    """The per-machine config file, or whatever ``RLMFLOW_CONFIG`` points at."""
    override = os.environ.get("RLMFLOW_CONFIG")
    if override:
        return Path(override).expanduser()
    home = Path.home() if home is None else home
    return Path(str(USER_CONFIG).replace("~", str(home), 1))


def _from_env(environ: dict[str, str]) -> dict[str, Any]:
    found = {}
    for name in DEFAULTS:
        value = environ.get(env_var(name))
        if value:
            found[name] = value
    return found


def _coerce(name: str, value: Any) -> Any:
    """Files and the environment hand over strings; the dataclass wants types."""
    if isinstance(DEFAULTS[name], int):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise CliError(f"{name} must be a whole number, not {value!r}") from exc
    return value if value is None else str(value)


__all__ = [
    "CONFIG_NAME",
    "CliError",
    "RunOptions",
    "env_var",
    "read_toml",
    "resolve",
    "user_config",
]
