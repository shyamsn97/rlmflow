"""Single-row benchmark entrypoint for the Docker executor."""

from __future__ import annotations

import json
import sys

from benchmarks.eval.run import _DOCKER_RESULT_PREFIX, _run_job_payload_inner


def main() -> int:
    payload = json.load(sys.stdin)
    result = _run_job_payload_inner(payload)
    print(_DOCKER_RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
