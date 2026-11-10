"""Drive headless Chrome over the DevTools protocol: open a page, watch it, screenshot it."""

from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from websocket import create_connection

CHROME_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
)
MAX_CONSOLE_LINES = 100


def chrome() -> str:
    for executable in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(executable)
        if found:
            return found
    found = next((str(path) for path in CHROME_CANDIDATES if path.is_file()), None)
    if found is None:
        raise RuntimeError("Chrome or Chromium is required")
    return found


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _page_endpoint(port: int, timeout: float = 10) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json/list", timeout=1
            ) as response:
                pages = [page for page in json.load(response) if page.get("type") == "page"]
            return str(pages[0]["webSocketDebuggerUrl"])
        except (IndexError, KeyError, OSError, json.JSONDecodeError):
            time.sleep(0.1)
    raise TimeoutError("Chrome DevTools endpoint did not become ready")


class CDP:
    """One DevTools session. Events that arrive while waiting for a reply are kept in ``events``."""

    def __init__(self, url: str, timeout: float) -> None:
        self.socket = create_connection(url, timeout=timeout)
        self.next_id = 1
        self.events: list[dict[str, Any]] = []

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        call_id = self.next_id
        self.next_id += 1
        self.socket.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if "method" in message:
                self.events.append(message)
            if message.get("id") != call_id:
                continue
            if "error" in message:
                raise RuntimeError(f"{method}: {message['error']}")
            return dict(message.get("result", {}))


@contextmanager
def chrome_page(page: Path, *, width: int, height: int, timeout: float = 60) -> Iterator[CDP]:
    """``page`` loaded in headless Chrome, with console events recorded from the first script."""
    port = _free_port()
    with tempfile.TemporaryDirectory(prefix="rlmflow-chrome-") as profile:
        process = subprocess.Popen(
            [
                chrome(),
                "--headless=new",
                "--hide-scrollbars",
                "--allow-file-access-from-files",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={profile}",
                f"--window-size={width},{height}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cdp: CDP | None = None
        try:
            cdp = CDP(_page_endpoint(port), timeout)
            cdp.call(
                "Emulation.setDeviceMetricsOverride",
                {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
            )
            cdp.call("Runtime.enable")
            cdp.call("Page.enable")
            cdp.call("Page.navigate", {"url": page.resolve().as_uri()})
            yield cdp
        finally:
            if cdp is not None:
                cdp.socket.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def console_lines(events: list[dict[str, Any]]) -> list[str]:
    lines = []
    for event in events:
        params = event.get("params", {})
        if event["method"] == "Runtime.consoleAPICalled":
            text = " ".join(
                str(arg.get("value", arg.get("description", ""))) for arg in params.get("args", [])
            )
            lines.append(f"{params.get('type', 'log')}: {text}")
        elif event["method"] == "Runtime.exceptionThrown":
            details = params.get("exceptionDetails", {})
            text = details.get("exception", {}).get("description") or details.get("text", "")
            lines.append(f"uncaught: {text}")
    if len(lines) > MAX_CONSOLE_LINES:
        omitted = len(lines) - MAX_CONSOLE_LINES
        lines = [*lines[:MAX_CONSOLE_LINES], f"... {omitted} more console lines"]
    return lines


def look(
    page: Path,
    out_dir: Path,
    *,
    seconds: float = 3,
    screenshots: int = 2,
    width: int = 1280,
    height: int = 720,
    scale: float = 0.625,
) -> tuple[list[str], list[Path]]:
    """Open ``page`` for ``seconds``; return its console output and evenly spaced screenshots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = uuid.uuid4().hex[:8]
    paths: list[Path] = []
    with chrome_page(page, width=width, height=height) as cdp:
        started = time.monotonic()
        for index in range(screenshots):
            time.sleep(max(0.0, started + seconds * (index + 1) / screenshots - time.monotonic()))
            shot = cdp.call(
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "clip": {"x": 0, "y": 0, "width": width, "height": height, "scale": scale},
                },
            )
            path = out_dir / f"{page.stem}-{stamp}-{index + 1}.png"
            path.write_bytes(base64.b64decode(shot["data"]))
            paths.append(path)
        if not screenshots:
            time.sleep(seconds)
        cdp.call("Runtime.evaluate", {"expression": "0"})
        return console_lines(cdp.events), paths
