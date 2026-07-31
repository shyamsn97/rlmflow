"""Docker-backed runtime for the minimal remote REPL protocol."""

from __future__ import annotations

from pathlib import Path

from rlmflow.graph import Node
from rlmflow.runtime.connections import PopenConnection
from rlmflow.runtime.repl import Repl
from rlmflow.runtime.repl_client import RemoteRepl
from rlmflow.runtime.runtime import Runtime


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
    argv += [image, "python", "-u", "-m", "rlmflow.runtime.repl_server"]
    if workdir is not None:
        argv += ["--workdir", workdir]
    return argv


class DockerRuntime(Runtime):
    """Run each agent in a Docker container using the minimal remote protocol."""

    def __init__(
        self,
        image: str,
        *,
        working_directory: str | Path | None = None,
        mounts: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        network: str | None = None,
        cpus: float | None = None,
        memory: str | None = None,
        user: str | None = None,
        workdir: str | None = None,
        extra_args: list[str] | None = None,
        docker_bin: str = "docker",
        repl_timeout: float | None = None,
        **options: object,
    ) -> None:
        super().__init__(working_directory=working_directory)
        self.image = image
        if self.working_directory is not None:
            host = str(self.working_directory.resolve())
            if mounts is None:
                mounts = {host: "/workspace"}
            if workdir is None:
                workdir = "/workspace"
        self.options = dict(
            mounts=mounts,
            env=env,
            network=network,
            cpus=cpus,
            memory=memory,
            user=user,
            workdir=workdir,
            extra_args=extra_args,
            docker_bin=docker_bin,
            repl_timeout=repl_timeout,
            **options,
        )

    def deploy_repl_server(self, agent: Node) -> RemoteRepl:
        cwd = str(self.working_directory) if self.working_directory else None
        options = dict(self.options)
        repl_timeout = options.pop("repl_timeout", None)
        argv = build_docker_argv(self.image, **options)
        return RemoteRepl(
            PopenConnection(
                argv,
                cwd=cwd,
                label="minimal Docker REPL",
                repl_timeout=repl_timeout,
            )
        )

    def open(self, agent: Node) -> Repl:
        return self.deploy_repl_server(agent)


__all__ = ["DockerRuntime", "build_docker_argv"]
