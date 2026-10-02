"""Capture a generated Boids app as a continuous browser GIF and MP4."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from validate import find_entrypoint
from websocket import WebSocket, create_connection

CHROME_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
)


def _chrome() -> str:
    for executable in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(executable)
        if found:
            return found
    found = next((str(path) for path in CHROME_CANDIDATES if path.is_file()), None)
    if found is None:
        raise SystemExit("Chrome or Chromium is required to record the simulation")
    return found


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found is None:
        raise SystemExit("ffmpeg is required to encode the simulation assets")
    return found


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _debug_page(port: int, target: str, timeout: float = 10) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                pages = json.load(response)
            matching = [
                page
                for page in pages
                if page.get("type") == "page"
                and (page.get("url") == target or page.get("url", "").startswith("file:"))
            ]
            page = (matching or [page for page in pages if page.get("type") == "page"])[0]
            return str(page["webSocketDebuggerUrl"])
        except (IndexError, KeyError, OSError, json.JSONDecodeError):
            time.sleep(0.1)
    raise TimeoutError("Chrome DevTools endpoint did not become ready")


class _CDP:
    def __init__(self, socket: WebSocket) -> None:
        self.socket = socket
        self.next_id = 1

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        call_id = self.next_id
        self.next_id += 1
        self.socket.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") != call_id:
                continue
            if "error" in message:
                raise RuntimeError(f"{method}: {message['error']}")
            return dict(message.get("result", {}))


def _capture_frames(
    run_dir: Path,
    frames_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    warmup_seconds: float,
) -> int:
    chrome = _chrome()
    port = _free_port()
    target = find_entrypoint(run_dir).resolve().as_uri()
    frame_count = max(2, round(fps * seconds))

    with tempfile.TemporaryDirectory(prefix="rlmflow-boids-chrome-") as profile:
        process = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--allow-file-access-from-files",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--run-all-compositor-stages-before-draw",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                f"--window-size={width},{height}",
                target,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        socket: WebSocket | None = None
        try:
            endpoint = _debug_page(port, target)
            socket = create_connection(endpoint, timeout=10)
            cdp = _CDP(socket)
            cdp.call("Page.enable")
            cdp.call(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            time.sleep(warmup_seconds)
            interval = 1 / fps
            for index in range(frame_count):
                started = time.monotonic()
                result = cdp.call(
                    "Page.captureScreenshot",
                    {
                        "format": "png",
                        "fromSurface": True,
                        "captureBeyondViewport": False,
                    },
                )
                (frames_dir / f"frame-{index:04d}.png").write_bytes(
                    base64.b64decode(result["data"])
                )
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            if socket is not None:
                socket.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return frame_count


def _encode(
    frames_dir: Path,
    *,
    fps: int,
    gif: Path | None,
    mp4: Path | None,
) -> None:
    ffmpeg = _ffmpeg()
    source = str(frames_dir / "frame-%04d.png")
    if mp4 is not None:
        mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                source,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "26",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(mp4),
            ],
            check=True,
        )
    if gif is not None:
        gif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                source,
                "-vf",
                (
                    "fps=8,scale=640:-1:flags=lanczos,split[s0][s1];"
                    "[s0]palettegen=max_colors=64[p];"
                    "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
                ),
                str(gif),
            ],
            check=True,
        )


def _assert_nonblank(frame: Path) -> None:
    from PIL import Image, ImageStat

    with Image.open(frame) as image:
        extrema = ImageStat.Stat(image.convert("RGB")).extrema
    if not any(high - low >= 8 for low, high in extrema):
        raise RuntimeError(f"browser rendered a blank frame: {frame}")


def render(
    run_dir: Path,
    *,
    gif: Path | None,
    mp4: Path | None,
    width: int = 800,
    height: int = 450,
    fps: int = 10,
    seconds: float = 5,
    warmup_seconds: float = 1.5,
) -> tuple[Path | None, Path | None]:
    if gif is None and mp4 is None:
        raise ValueError("at least one of gif or mp4 is required")
    find_entrypoint(run_dir)
    with tempfile.TemporaryDirectory(prefix="rlmflow-boids-frames-") as temp:
        frames_dir = Path(temp)
        count = _capture_frames(
            run_dir,
            frames_dir,
            width=width,
            height=height,
            fps=fps,
            seconds=seconds,
            warmup_seconds=warmup_seconds,
        )
        _assert_nonblank(frames_dir / f"frame-{count - 1:04d}.png")
        _encode(frames_dir, fps=fps, gif=gif, mp4=mp4)
    print(f"captured {count} continuous browser frames")
    return gif, mp4


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the generated Boids canvas.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/_runs/coding/boids-rlmflow"),
    )
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--mp4", type=Path)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--warmup-seconds", type=float, default=1.5)
    args = parser.parse_args()
    gif, mp4 = render(
        args.run_dir,
        gif=args.gif,
        mp4=args.mp4,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
        warmup_seconds=args.warmup_seconds,
    )
    if gif:
        print(f"wrote {gif}")
    if mp4:
        print(f"wrote {mp4}")


if __name__ == "__main__":
    main()
