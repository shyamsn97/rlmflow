"""Modal launcher for autoresearch trial directories."""

from __future__ import annotations

import atexit
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ``modal.enable_output()`` drives a single *process-global* output manager.
# Concurrent trials each run ``submit()`` on their own thread, so entering the
# context from several threads at once races on that global and can raise
# ``AttributeError: 'NoneType' object has no attribute 'label'``. Enter it once
# for the whole process (under a lock) and leave it open, so concurrent
# ``app.run()`` calls share the already-initialized manager.
_OUTPUT_LOCK = threading.Lock()
_OUTPUT_ENABLED = False


SUMMARY_KEYS = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
)
SUMMARY_RE = re.compile(
    rf"^({'|'.join(re.escape(key) for key in SUMMARY_KEYS)}):\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ModalConfig:
    app_name: str = "rlmflow-autoresearch"
    gpu: str = "L4"
    parallel: int = 4
    timeout_s: int = 1200
    volume_name: str = "rlmflow-autoresearch-cache"
    python_version: str | None = None


def validate_gpu(config: ModalConfig) -> None:
    if not str(config.gpu).strip():
        raise ValueError("--gpu must be a non-empty Modal GPU spec, e.g. L4 or H100")


def parse_autoresearch_stdout(
    stdout: str,
    stderr: str = "",
    *,
    returncode: int = 0,
) -> dict[str, Any]:
    """Parse autoresearch's final summary block."""

    metrics: dict[str, float | int] = {}
    for key, raw in SUMMARY_RE.findall(stdout or ""):
        value = float(raw)
        if key in {"num_steps", "depth"}:
            metrics[key] = int(value)
        else:
            metrics[key] = value

    val_bpb = metrics.get("val_bpb")
    status = classify_result(returncode, stdout, stderr, val_bpb)
    return {
        **metrics,
        "val_bpb": float(val_bpb) if val_bpb is not None else None,
        "status": status,
    }


def classify_result(
    returncode: int,
    stdout: str = "",
    stderr: str = "",
    val_bpb: float | int | None = None,
) -> str:
    lower = f"{stdout}\n{stderr}".lower()
    if "out of memory" in lower or "cuda oom" in lower:
        return "oom"
    if returncode != 0:
        return "crashed"
    if val_bpb is None:
        return "crashed"
    return "succeeded"


def preflight(config: ModalConfig) -> dict[str, Any]:
    """Fail before agent startup if Modal is not importable/authenticated."""

    validate_gpu(config)
    _modal()
    return {
        "status": "ok",
        "app_name": config.app_name,
        "gpu": config.gpu,
        "parallel": config.parallel,
    }


def submit(
    config: ModalConfig,
    *,
    path: str | Path,
    slug: str,
    n: int,
    run_id: str,
) -> dict[str, Any]:
    """Submit one trial directory to Modal and wait for the result."""

    modal = _modal()
    app = modal.App(config.app_name)
    cache_volume = modal.Volume.from_name(config.volume_name, create_if_missing=True)

    @app.function(
        image=_image(modal, config),
        gpu=config.gpu,
        timeout=config.timeout_s,
        max_containers=max(1, config.parallel),
        volumes={"/root/.cache/autoresearch": cache_volume},
        secrets=_secrets(modal, config),
        serialized=True,
    )
    def run_trial(payload: dict[str, Any]) -> dict[str, Any]:
        import os
        import re
        import subprocess
        import tempfile
        from pathlib import Path
        from time import monotonic

        def needs_prepare() -> bool:
            data_dir = Path("/root/.cache/autoresearch/data")
            return not ((data_dir / "train.pt").exists() and (data_dir / "val.pt").exists())

        def coerce_output(value: str | bytes | None) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode(errors="replace")
            return value

        def classify_result(returncode: int, stdout: str, stderr: str, val_bpb) -> str:
            lower = f"{stdout}\n{stderr}".lower()
            if "out of memory" in lower or "cuda oom" in lower:
                return "oom"
            if returncode != 0:
                return "crashed"
            if val_bpb is None:
                return "crashed"
            return "succeeded"

        def parse_summary(stdout: str, stderr: str, returncode: int) -> dict[str, Any]:
            keys = (
                "val_bpb",
                "training_seconds",
                "total_seconds",
                "peak_vram_mb",
                "total_tokens_M",
                "num_steps",
                "num_params_M",
                "depth",
            )
            pattern = re.compile(
                rf"^({'|'.join(re.escape(key) for key in keys)}):\s*"
                r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
                re.MULTILINE,
            )
            metrics = {}
            for key, raw in pattern.findall(stdout or ""):
                value = float(raw)
                metrics[key] = int(value) if key in {"num_steps", "depth"} else value
            val_bpb = metrics.get("val_bpb")
            return {
                **metrics,
                "val_bpb": float(val_bpb) if val_bpb is not None else None,
                "status": classify_result(returncode, stdout, stderr, val_bpb),
            }

        t0 = monotonic()
        trial = payload["n"]
        trial_slug = payload["slug"]
        print(f"[autoresearch] start trial={trial} slug={trial_slug}", flush=True)

        with tempfile.TemporaryDirectory(prefix="autoresearch-") as tmp:
            root = Path(tmp)
            for relpath, text in payload["files"].items():
                target = root / relpath
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)

            env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "AUTORESEARCH_RUN_ID": payload["run_id"],
                "AUTORESEARCH_TRIAL": str(trial),
                "AUTORESEARCH_SLUG": trial_slug,
            }

            prep_stdout = ""
            prep_stderr = ""
            if needs_prepare():
                prep = subprocess.run(
                    ["uv", "run", "prepare.py"],
                    cwd=str(root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=int(payload["timeout_s"]),
                )
                prep_stdout = prep.stdout or ""
                prep_stderr = prep.stderr or ""
                if prep.returncode != 0:
                    return {
                        "status": "infra_error",
                        "returncode": prep.returncode,
                        "elapsed_s": monotonic() - t0,
                        "stdout": prep_stdout,
                        "stderr": prep_stderr,
                    }
                try:
                    cache_volume.commit()
                except Exception:
                    pass

            try:
                train = subprocess.run(
                    ["uv", "run", "train.py"],
                    cwd=str(root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=int(payload["timeout_s"]),
                )
            except subprocess.TimeoutExpired as exc:
                stdout = coerce_output(exc.stdout)
                stderr = coerce_output(exc.stderr)
                return {
                    "status": "timeout",
                    "returncode": -1,
                    "elapsed_s": monotonic() - t0,
                    "stdout": prep_stdout + stdout,
                    "stderr": prep_stderr + stderr,
                }

            stdout = prep_stdout + (train.stdout or "")
            stderr = prep_stderr + (train.stderr or "")
            parsed = parse_summary(stdout, stderr, train.returncode)
            print(
                f"[autoresearch] done trial={trial} slug={trial_slug} "
                f"status={parsed['status']} val_bpb={parsed.get('val_bpb')}",
                flush=True,
            )
            return {
                **parsed,
                "returncode": train.returncode,
                "elapsed_s": monotonic() - t0,
                "stdout": stdout,
                "stderr": stderr,
            }

    payload = {
        "files": _pack_trial_dir(Path(path)),
        "slug": slug,
        "n": n,
        "run_id": run_id,
        "timeout_s": config.timeout_s,
    }
    # Enable Modal's output once for the process (thread-safe), then keep the
    # ephemeral app alive until the remote call finishes. Each rlmflow child blocks
    # here, so several children block on independent Modal calls concurrently.
    _enable_output_once(modal)
    with app.run():
        call = run_trial.spawn(payload)
        job_id = (
            getattr(call, "object_id", None)
            or getattr(call, "function_call_id", None)
            or f"{run_id}:{n}:{slug}"
        )
        result = call.get(timeout=config.timeout_s)

    return {
        **(result or {}),
        "job_id": job_id,
        "gpu": config.gpu,
    }


def _enable_output_once(modal) -> None:
    """Enter Modal's process-global output manager exactly once, thread-safely.

    Entering ``modal.enable_output()`` concurrently from multiple trial threads
    races on a shared global; do it once and leave it open for the process so
    every ``app.run()`` shares the initialized manager. Closed at interpreter
    exit.
    """
    global _OUTPUT_ENABLED
    with _OUTPUT_LOCK:
        if _OUTPUT_ENABLED:
            return
        cm = modal.enable_output()
        cm.__enter__()
        atexit.register(lambda: cm.__exit__(None, None, None))
        _OUTPUT_ENABLED = True


def _pack_trial_dir(path: Path) -> dict[str, str]:
    allowed = {"README.md", "prepare.py", "train.py", "program.md", "pyproject.toml", "uv.lock"}
    files: dict[str, str] = {}
    for name in sorted(allowed):
        candidate = path / name
        if candidate.exists():
            files[name] = candidate.read_text()
    missing = {"prepare.py", "train.py", "pyproject.toml"} - set(files)
    if missing:
        raise FileNotFoundError(f"trial directory is missing required files: {sorted(missing)}")
    return files


def _modal():
    try:
        import modal  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "autoresearch requires the modal extra: pip install -e '.[modal]'"
        ) from exc
    return modal


def _secrets(modal, _config: ModalConfig) -> list[Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return []
    return [
        modal.Secret.from_dict(
            {
                "HF_TOKEN": token,
                "HUGGING_FACE_HUB_TOKEN": token,
            }
        )
    ]


def _image(modal, config: ModalConfig):
    python_version = config.python_version or (f"{sys.version_info.major}.{sys.version_info.minor}")
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
            add_python=python_version,
        )
        .apt_install("git", "curl", "build-essential")
        .run_commands(
            "python -m pip install --upgrade pip",
            "python -m pip install uv",
        )
        .env(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": "/root",
                "CC": "gcc",
                "CXX": "g++",
            }
        )
        .add_local_file(Path(__file__), "/root/modal_runner.py")
    )
