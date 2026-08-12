"""Docker provisioner for the lightweight Python worker."""

from __future__ import annotations

from pathlib import Path

from rlmflow.graph.nodes import AgentStart
from rlmflow.runtime.connections import DEFAULT_REPL_TIMEOUT, PopenConnection
from rlmflow.runtime.repl import Repl
from rlmflow.runtime.repl_client import WorkerRepl, WorkerSession
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
    """Build a stdio-attached, automatically removed worker container."""
    argv = [docker_bin, "run", "-i", "--rm"]
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
    return argv


class DockerRuntime(Runtime):
    """Run isolated or explicitly shared agents in Docker workers."""

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
        repl_timeout: float | None = DEFAULT_REPL_TIMEOUT,
        execution_timeout: float | None = None,
        **options: object,
    ) -> None:
        super().__init__(working_directory=working_directory)
        self.image = image
        if self.working_directory is not None:
            host = str(self.working_directory.resolve())
            mounts = mounts or {host: "/workspace"}
            workdir = workdir or "/workspace"
        self.repl_timeout = repl_timeout or DEFAULT_REPL_TIMEOUT
        self.execution_timeout = execution_timeout
        self.options = {
            "mounts": mounts,
            "env": env,
            "network": network,
            "cpus": cpus,
            "memory": memory,
            "user": user,
            "workdir": workdir,
            "extra_args": extra_args,
            "docker_bin": docker_bin,
            **options,
        }

    def open(self, agent: AgentStart) -> Repl:
        if agent.config.reuse_repl and agent.parent is not None:
            parent = agent.parent.parent_agent
            parent_repl = self.repls.get(parent.id)
            if not isinstance(parent_repl, WorkerRepl):
                raise RuntimeError("reuse_repl requires a live parent worker")
            return WorkerRepl(parent_repl.session, tenant_id=agent.id)
        connection = PopenConnection(
            build_docker_argv(self.image, **self.options),
            cwd=self.working_directory or Path.cwd(),
            label="Docker REPL worker",
        )
        session = WorkerSession(
            connection,
            timeout=self.repl_timeout,
            execution_timeout=self.execution_timeout,
        )
        return WorkerRepl(session, tenant_id=agent.id)


__all__ = ["DockerRuntime", "build_docker_argv"]
