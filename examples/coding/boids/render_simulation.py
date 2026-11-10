"""Capture a generated Boids app as a continuous browser GIF and MP4."""

from __future__ import annotations

import argparse
import base64
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from browser import chrome_page
from validate import find_entrypoint


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found is None:
        raise SystemExit("ffmpeg is required to encode the simulation assets")
    return found


RECORD_SCRIPT = """
(async () => {
  const canvas = document.querySelector("canvas");
  if (!canvas) throw new Error("no canvas on the page");
  const recorder = new MediaRecorder(canvas.captureStream(%(fps)d), {
    mimeType: "video/webm;codecs=vp9",
    videoBitsPerSecond: 8000000,
  });
  const chunks = [];
  recorder.ondataavailable = (event) => chunks.push(event.data);
  const stopped = new Promise((resolve) => (recorder.onstop = resolve));
  recorder.start();
  await new Promise((resolve) => setTimeout(resolve, %(ms)d));
  recorder.stop();
  await stopped;
  const bytes = new Uint8Array(await new Blob(chunks).arrayBuffer());
  let binary = "";
  for (let i = 0; i < bytes.length; i += 32768) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 32768));
  }
  return btoa(binary);
})()
"""


def _record(
    run_dir: Path,
    video: Path,
    *,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    warmup_seconds: float,
) -> None:
    """Record the page's canvas in real time with MediaRecorder, as WebM."""
    page = find_entrypoint(run_dir)
    with chrome_page(
        page, width=width, height=height, timeout=seconds + warmup_seconds + 30
    ) as cdp:
        time.sleep(warmup_seconds)
        result = cdp.call(
            "Runtime.evaluate",
            {
                "expression": RECORD_SCRIPT % {"fps": fps, "ms": round(seconds * 1000)},
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
    if "exceptionDetails" in result:
        raise RuntimeError(f"recording failed: {result['exceptionDetails']}")
    video.write_bytes(base64.b64decode(result["result"]["value"]))


def _encode(
    video: Path,
    *,
    fps: int,
    gif: Path | None,
    mp4: Path | None,
) -> None:
    ffmpeg = _ffmpeg()
    if mp4 is not None:
        mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-r",
                str(fps),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "24",
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
                "-i",
                str(video),
                "-vf",
                (
                    f"fps={min(fps, 20)},scale=480:-1:flags=lanczos,split[s0][s1];"
                    "[s0]palettegen=max_colors=32:stats_mode=diff[p];"
                    "[s1][p]paletteuse=dither=none:diff_mode=rectangle"
                ),
                str(gif),
            ],
            check=True,
        )


def _last_frame(video: Path, out: Path) -> Path:
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-sseof",
            "-0.5",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )
    return out


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
    fps: int = 30,
    seconds: float = 4,
    warmup_seconds: float = 6,
) -> tuple[Path | None, Path | None]:
    if gif is None and mp4 is None:
        raise ValueError("at least one of gif or mp4 is required")
    find_entrypoint(run_dir)
    with tempfile.TemporaryDirectory(prefix="rlmflow-boids-video-") as temp:
        video = Path(temp) / "canvas.webm"
        _record(
            run_dir,
            video,
            width=width,
            height=height,
            fps=fps,
            seconds=seconds,
            warmup_seconds=warmup_seconds,
        )
        _assert_nonblank(_last_frame(video, Path(temp) / "last.png"))
        _encode(video, fps=fps, gif=gif, mp4=mp4)
    print(f"recorded {seconds:g}s of the canvas at {fps} fps")
    return gif, mp4


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the generated Boids canvas.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/_runs/coding/boids"),
    )
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--mp4", type=Path)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--warmup-seconds", type=float, default=6)
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
