"""Docker transport for the minimal remote REPL protocol."""

from __future__ import annotations

from pathlib import Path


def build_docker_argv(
    image: str,
    *,
    mounts: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    network: str | None = None,
    cpus: float | None = None,
    memory: str | None = None,
    user: str | None = None,
    workdir: str | None = None,
    extra_args: list[str] | None = None,
    docker_bin: str = "docker",
) -> list[str]:
    argv: list[str] = [docker_bin, "run", "-i", "--rm"]
    for host, container in (mounts or {}).items():
        argv += ["-v", f"{Path(host).resolve()}:{container}"]
    for key, value in (env or {}).items():
        argv += ["-e", f"{key}={value}"]
    if network is not None:
        argv += ["--network", network]
    if cpus is not None:
        argv += ["--cpus", str(cpus)]
    if memory is not None:
        argv += ["--memory", memory]
    if user is not None:
        argv += ["--user", user]
    if workdir is not None:
        argv += ["--workdir", workdir]
    argv += list(extra_args or [])
    argv += [image, "python", "-u", "-m", "rflow.minimal.remote_server"]
    if workdir is not None:
        argv += ["--workdir", workdir]
    return argv


__all__ = ["build_docker_argv"]
