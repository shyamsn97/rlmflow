"""Validate and browser-smoke a generated Boids application."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

CHROME_CANDIDATES = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
)


class _Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.sources: list[str] = []
        self.stylesheets: list[str] = []
        self.canvas_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        tag = tag.lower()
        if tag == "script" and attributes.get("src"):
            self.sources.append(str(attributes["src"]))
        elif tag == "link" and attributes.get("rel") == "stylesheet" and attributes.get("href"):
            self.stylesheets.append(str(attributes["href"]))
        elif tag == "canvas":
            self.canvas_count += 1


def find_entrypoint(run_dir: Path) -> Path:
    """Return the generated HTML entrypoint without prescribing its name."""
    pages = sorted(run_dir.glob("*.html"))
    if not pages:
        raise FileNotFoundError("no HTML entrypoint found")
    conventional = run_dir / "index.html"
    return conventional if conventional in pages else pages[0]


def _chrome() -> str | None:
    for executable in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        found = shutil.which(executable)
        if found:
            return found
    return next((str(path) for path in CHROME_CANDIDATES if path.is_file()), None)


def _check_js(run_dir: Path, files: list[Path]) -> tuple[bool, str]:
    node = shutil.which("node")
    if node is None:
        return False, "node executable not found"
    if not files:
        return True, "no external JavaScript files; browser smoke checks inline code"
    failures = []
    for path in files:
        proc = subprocess.run(
            [node, "--check", str(path.relative_to(run_dir))],
            cwd=run_dir,
            text=True,
            capture_output=True,
        )
        if proc.returncode:
            failures.append(f"{path.relative_to(run_dir)}: {(proc.stderr or proc.stdout).strip()}")
    return not failures, "\n".join(failures) if failures else "all JavaScript parsed"


def _browser_smoke(entrypoint: Path, screenshot: Path | None) -> tuple[bool, str]:
    chrome = _chrome()
    if chrome is None:
        return False, "Chrome or Chromium executable not found"
    target = screenshot or entrypoint.parent / "boids-smoke.png"
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            "--enable-logging=stderr",
            "--v=1",
            "--virtual-time-budget=3000",
            "--window-size=1280,720",
            f"--screenshot={target}",
            entrypoint.resolve().as_uri(),
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    console_errors = [
        line
        for line in proc.stderr.splitlines()
        if "INFO:CONSOLE" in line and ("Uncaught" in line or "Error" in line)
    ]
    passed = (
        proc.returncode == 0
        and target.is_file()
        and target.stat().st_size > 0
        and not console_errors
    )
    detail = (
        f"loaded {entrypoint.name}; screenshot={target}"
        if passed
        else "\n".join(console_errors) or (proc.stderr or proc.stdout)
    )
    return passed, detail.strip()


def validate_project(
    run_dir: Path,
    *,
    browser: bool = False,
    screenshot: Path | None = None,
) -> dict[str, object]:
    """Return a machine-readable validation report for one generated workspace."""
    run_dir = run_dir.resolve()
    checks: dict[str, dict[str, object]] = {}

    try:
        entrypoint = find_entrypoint(run_dir)
    except FileNotFoundError:
        entrypoint = None
    checks["entrypoint"] = {
        "passed": entrypoint is not None,
        "detail": entrypoint.name if entrypoint is not None else "no HTML entrypoint found",
    }

    document = _Document()
    if entrypoint is not None:
        document.feed(entrypoint.read_text(encoding="utf-8"))
    references = [*document.sources, *document.stylesheets]
    parsed_references = [(reference, urlparse(reference)) for reference in references]
    external = [
        reference
        for reference, parsed in parsed_references
        if parsed.scheme or parsed.netloc
    ]
    missing = [
        reference
        for reference, parsed in parsed_references
        if not parsed.scheme
        and not parsed.netloc
        and entrypoint is not None
        and not (entrypoint.parent / parsed.path).is_file()
    ]
    checks["local_assets"] = {
        "passed": not external and not missing,
        "detail": (
            f"{len(references)} local references resolved"
            if not external and not missing
            else f"external: {external or 'none'}; missing: {missing or 'none'}"
        ),
    }
    checks["canvas"] = {
        "passed": document.canvas_count > 0,
        "detail": f"{document.canvas_count} canvas element(s)",
    }

    js_files = sorted(run_dir.rglob("*.js"))
    js_passed, js_detail = _check_js(run_dir, js_files)
    checks["javascript_syntax"] = {"passed": js_passed, "detail": js_detail}

    if browser and entrypoint is not None:
        browser_passed, browser_detail = _browser_smoke(entrypoint, screenshot)
        checks["browser_smoke"] = {"passed": browser_passed, "detail": browser_detail}
    elif browser:
        checks["browser_smoke"] = {"passed": False, "detail": "no HTML entrypoint to load"}

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def write_report(report: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generated Boids browser application.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--browser", action="store_true", help="Load index.html in headless Chrome.")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = validate_project(
        args.run_dir,
        browser=args.browser,
        screenshot=args.screenshot,
    )
    report_path = args.report or args.run_dir / "validation.json"
    write_report(report, report_path)
    for name, check in report["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status:4} {name}: {check['detail']}")
    print(f"report: {report_path}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
