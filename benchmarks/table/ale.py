"""Algorithm engineering: ALE-Bench-Lite, 10 AtCoder Heuristic Contest problems.

Every system gets the problem statement and an ``./evaluate`` command that runs
``main.cpp`` on the problem's public test cases, as contestants had. The final
``main.cpp`` is scored on the hidden cases and ranked against the contest's
participants. ALE-Bench runs in its own environment (``ALE_PYTHON``), since it
needs Python 3.11+ and Docker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.table.systems import Limits, Outcome

NAME = "ale"
HOME = Path.home() / ".cache" / "rlmflow-bench" / "ALE-Bench"
ALE_PYTHON = Path(os.environ.get("ALE_PYTHON", HOME / ".venv" / "bin" / "python"))
ENV = {**os.environ, "DYLD_FALLBACK_LIBRARY_PATH": "/opt/homebrew/opt/cairo/lib"}
LITE = [
    "ahc008",
    "ahc011",
    "ahc015",
    "ahc016",
    "ahc024",
    "ahc025",
    "ahc026",
    "ahc027",
    "ahc039",
    "ahc046",
]

TASK = """\
Solve the AtCoder Heuristic Contest problem in `problem.md` (figures are next to
it). Write your solution in C++23 (gcc 12.2, AC Library and Boost available) to
`main.cpp`.

`./evaluate` compiles `main.cpp` and runs it on the public test cases, printing
each case's score and status. When you finish, `main.cpp` is run on hidden test
cases and ranked against the contest's participants.

You have {minutes} minutes in total. Leave your best solution in `main.cpp`
before then."""

#: Runs inside ALE_PYTHON: ``helper.py statement|public|private PROBLEM [CODE]``.
HELPER = r"""
import json, sys
from pathlib import Path
import ale_bench

command, problem_id = sys.argv[1], sys.argv[2]
session = ale_bench.start(problem_id, lite_version=False, num_workers=8)
try:
    if command == "statement":
        out = Path(sys.argv[3])
        statement = session.problem.statement
        for name, image in session.problem.statement_images.items():
            image = image[0] if isinstance(image, list) else image
            png = Path(name).stem + ".png"
            image.convert("RGB").save(out / png)
            statement = statement.replace(name, png)
        (out / "problem.md").write_text(statement)
    else:
        code = Path(sys.argv[3]).read_text()
        if command == "public":
            result = session.public_eval(code, code_language="cpp23", judge_version="202301")
            print(f"overall score: {result.overall_absolute_score}")
            for i, case in enumerate(result.case_results):
                status = getattr(case.judge_result, "value", case.judge_result)
                print(f"case {i}: {status} score {case.absolute_score} time {case.execution_time:.2f}s {case.message}".rstrip())
        else:
            result, rank, performance = session.private_eval(code, code_language="cpp23", judge_version="202301")
            print(json.dumps({"score": result.overall_absolute_score, "rank": rank, "performance": performance}))
finally:
    session.close()
"""

EVALUATE = """\
#!/bin/sh
exec {python} {helper} public {problem} "${{1:-main.cpp}}"
"""


def helper(command: str, problem: str, *args: str) -> str:
    script = HOME / "rlmflow_helper.py"
    script.write_text(HELPER)
    done = subprocess.run(
        [str(ALE_PYTHON), str(script), command, problem, *args],
        env=ENV,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-2000:])
    return done.stdout


def ids() -> list[str]:
    return list(LITE)


def prepare(task_id: str, workdir: Path, limits: Limits) -> str:
    helper("statement", task_id, str(workdir))
    evaluate = workdir / "evaluate"
    evaluate.write_text(
        EVALUATE.format(python=ALE_PYTHON, helper=HOME / "rlmflow_helper.py", problem=task_id)
    )
    evaluate.chmod(0o755)
    return TASK.format(minutes=int(limits.minutes))


def score(task_id: str, workdir: Path, outcome: Outcome) -> dict:
    code = workdir / "main.cpp"
    if not code.exists():
        return {"score": None, "rank": None, "performance": 0, "error": "no main.cpp"}
    try:
        return json.loads(helper("private", task_id, str(code)).strip().splitlines()[-1])
    except (RuntimeError, json.JSONDecodeError, IndexError) as exc:
        print(f"{task_id}: private eval failed: {exc}", file=sys.stderr)
        return {"score": None, "rank": None, "performance": 0, "error": str(exc)[-500:]}
