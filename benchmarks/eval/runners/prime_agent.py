"""Pinned PrimeAgent runner for runner-neutral fanout comparisons."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmarks.eval import runner
from benchmarks.eval.delegation.fanout import prime_trace
from benchmarks.eval.delegation.parity import comparison_contract
from benchmarks.eval.runners.shared import (
    fanout_payloads_from_example,
    fanout_shards_from_payloads,
    prepare_run,
    runtime_prompt,
)
from benchmarks.eval.tasks.swarm_fanout import fanout_payloads
from benchmarks.eval.types import Example, Model, Prediction, RunContext, Runner


@runner("prime-agent", tags=["external-reference", "delegation-arm"])
class PrimeAgentRunner(Runner):
    def __init__(
        self,
        agent_bin: str = "prime-agent",
        timeout_seconds: int = 1800,
    ) -> None:
        self.agent_bin = agent_bin
        self.timeout_seconds = timeout_seconds

    def run(self, example: Example, model: Model, ctx: RunContext) -> Prediction:
        payloads = fanout_payloads_from_example(example)
        shards = _fanout_shards(example)
        shards = shards or fanout_shards_from_payloads(payloads)
        if payloads is None and shards is None:
            return Prediction(
                answer="",
                error="PrimeAgentRunner currently supports only swarm-fanout fixtures",
            )
        prepared = prepare_run(example, ctx, runtime="prime-agent")
        workdir = prepared.work_dir
        if payloads is None:
            payloads = fanout_payloads(shards, workdir / "shards")
        payloads_path = workdir / "fanout-payloads.json"
        payloads_path.write_text(
            json.dumps(
                payloads,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        _init_repo(workdir)
        sessions_dir = ctx.artifact_dir / "prime-sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir = sessions_dir.resolve()
        agent_home = ctx.artifact_dir / "prime-home"
        agent_home.mkdir(parents=True, exist_ok=True)
        # Unix-domain socket paths are length-limited; artifact paths are intentionally long.
        socket_path = Path(tempfile.gettempdir()) / f"prime-agent-{os.getpid()}.sock"
        prompt = _runtime_prompt(example)
        started = time.perf_counter()
        launch_error = None
        stdout = ""
        stderr = ""
        returncode = None
        try:
            completed = subprocess.run(
                [
                    self.agent_bin,
                    "--mode",
                    "json",
                    "--daemon-socket",
                    str(socket_path),
                    "--cwd",
                    str(workdir),
                    "--session-dir",
                    str(sessions_dir),
                    "--model",
                    _model_selector(model),
                    "--",
                    prompt,
                ],
                cwd=workdir,
                env=_agent_env(agent_home, sessions_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = _captured(exc.stdout)
            stderr = _captured(exc.stderr)
            launch_error = f"TimeoutError: PrimeAgent exceeded {self.timeout_seconds}s"
        except OSError as exc:
            launch_error = f"{type(exc).__name__}: {exc}"
        finally:
            _shutdown_daemon(socket_path)
        elapsed = time.perf_counter() - started

        ledger_text = _read_ledger(agent_home, sessions_dir)
        records = _parse_records(ledger_text)
        transcript = _parent_transcript(sessions_dir, records)
        persisted = {
            str(record["childId"])
            for record in records
            if record.get("op") == "spawn"
            and isinstance(record.get("child"), str)
            and Path(record["child"]).is_file()
        }
        replies, notices = _receipt_sets(transcript)
        evidence = prime_trace(
            shards,
            records,
            persisted_child_ids=persisted,
            reply_session_ids=replies,
            noticed_child_ids=notices,
        )
        usage = _usage(stdout)
        answer_path = workdir / "combined-index.md"
        answer = _final_answer(stdout)
        if not answer and answer_path.is_file():
            answer = answer_path.read_text()
        children = evidence["children"]
        retrieved = sum(bool(child.get("receipt")) for child in children)
        metrics = {
            "time_seconds": elapsed,
            "iterations": usage["turns"],
            "comparison_contract": comparison_contract(
                runtime="prime-agent",
                task_prompt=example.prompt,
                delivered_prompt=prompt,
                inputs={"fanout_payloads": payloads},
                root_model=model.name,
                worker_model=model.name,
                recursive_levels=1,
                max_iterations=usage["turns"],
                max_tokens=None,
                attempt_timeout_seconds=int(ctx.params.get("attempt_timeout", 0)),
                environment=None,
            ),
            "graph": {
                "agents": 1 + len(children),
                "nodes": len(children),
                "llm_turns": usage["turns"],
                "max_depth": 1 if children else 0,
                "max_branching": len(children),
            },
            "delegation": {
                "arm": "prime-agent",
                "children": len(children),
                "launch_events": len(children),
                "retrieved_child_results": retrieved,
                "retrieval_ratio": retrieved / len(children) if children else 0.0,
                "max_depth": 1 if children else 0,
                "runtime": "prime-agent",
            },
            "fanout": evidence,
            "prime_agent": {
                "returncode": returncode,
                "stderr": stderr[-4000:],
            },
        }
        artifacts = {
            "workdir": str(workdir),
            "fanout_payloads": str(payloads_path),
            "sessions": str(sessions_dir),
            "ledger": str(_ledger_path(agent_home, sessions_dir)),
            "stdout": str(ctx.artifact_dir / "prime-agent.stdout.jsonl"),
            "stderr": str(ctx.artifact_dir / "prime-agent.stderr.txt"),
        }
        Path(artifacts["stdout"]).write_text(stdout)
        Path(artifacts["stderr"]).write_text(stderr)
        if answer_path.is_file():
            artifacts["combined_index"] = str(answer_path)
        return Prediction(
            answer=answer,
            usage={
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
            },
            metrics=metrics,
            artifacts=artifacts,
            error=launch_error,
        )


def _fanout_shards(example: Example) -> list[dict[str, Any]] | None:
    shards = example.metadata.get("shards")
    return list(shards) if example.metadata.get("fixture") and isinstance(shards, list) else None


def _runtime_prompt(example: Example) -> str:
    from benchmarks.eval.tasks.swarm_fanout import fanout_runtime_guidance

    return runtime_prompt(example.prompt, fanout_runtime_guidance("prime-agent"))


def _model_selector(model: Model) -> str:
    provider = str(getattr(model, "provider", "") or "openai")
    return f"{provider}/{model.name}"


def _init_repo(workdir: Path) -> None:
    if (workdir / ".git").is_dir():
        return
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "eval",
        "GIT_AUTHOR_EMAIL": "eval@eval",
        "GIT_COMMITTER_NAME": "eval",
        "GIT_COMMITTER_EMAIL": "eval@eval",
    }
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"],
    ):
        subprocess.run(command, cwd=workdir, env=env, check=True, capture_output=True)


def _agent_env(agent_home: Path, sessions_dir: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PRIME_AGENT_INTERNAL_") and key not in {"RLM_DEPTH", "RLM_MAX_DEPTH"}
    }
    for prefix in ("PRIME_AGENT", "PI"):
        env[f"{prefix}_CODING_AGENT_DIR"] = str(agent_home)
        env[f"{prefix}_SESSION_DIR"] = str(sessions_dir)
        env[f"{prefix}_CODING_AGENT_SESSION_DIR"] = str(sessions_dir)
    return env


def _ledger_path(agent_home: Path, sessions_dir: Path) -> Path:
    canonical = os.path.realpath(sessions_dir)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return agent_home / "rlm-ledger" / f"{digest}.jsonl"


def _read_ledger(agent_home: Path, sessions_dir: Path) -> str:
    path = _ledger_path(agent_home, sessions_dir)
    return path.read_text() if path.is_file() else ""


def _parse_records(text: str) -> list[dict[str, Any]]:
    records = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _parent_transcript(sessions_dir: Path, records: list[dict[str, Any]]) -> str:
    parent_names = {
        Path(record["parent"]).name
        for record in records
        if record.get("op") == "spawn" and isinstance(record.get("parent"), str)
    }
    candidates = [path for path in sessions_dir.glob("*.jsonl") if path.name in parent_names]
    if not candidates:
        candidates = list(sessions_dir.glob("*.jsonl"))
    if not candidates:
        return ""
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    return path.read_text()


def _receipt_sets(transcript: str) -> tuple[set[str], set[str]]:
    replies: set[str] = set()
    notices: set[str] = set()
    for record in _parse_records(transcript):
        details = record.get("details") or {}
        if record.get("customType") == "agent_message":
            sender = details.get("from") or {}
            if details.get("fromRelationship") == "child" and sender.get("sessionId"):
                replies.add(str(sender["sessionId"]))
        elif record.get("customType") in {"rlm_child_terminal_notice", "rlm_child_failure"}:
            if details.get("childId"):
                notices.add(str(details["childId"]))
    return replies, notices


def _usage(stdout: str) -> dict[str, int]:
    input_tokens = output_tokens = turns = 0
    for record in _parse_records(stdout):
        message = record.get("message")
        if record.get("type") != "message_end" or not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage") or {}
        input_tokens += int(usage.get("input") or usage.get("inputTokens") or 0)
        output_tokens += int(usage.get("output") or usage.get("outputTokens") or 0)
        turns += 1
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turns": turns,
    }


def _final_answer(stdout: str) -> str:
    for record in reversed(_parse_records(stdout)):
        message = record.get("message")
        if record.get("type") != "message_end" or not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text = "\n".join(
                str(item.get("text"))
                for item in content
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ).strip()
            if text:
                return text
    return ""


def _shutdown_daemon(socket_path: Path) -> None:
    if not socket_path.exists():
        return
    envelope = json.dumps(
        {
            "type": "command",
            "id": "eval-shutdown",
            "protocol": {"name": "prime-agent.daemon", "version": 7},
            "command": {"type": "shutdown"},
        }
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5)
            client.connect(str(socket_path))
            client.sendall(envelope.encode() + b"\n")
            while client.recv(4096):
                pass
    except OSError:
        pass


def _captured(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


__all__ = ["PrimeAgentRunner"]
