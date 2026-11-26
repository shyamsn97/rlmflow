#!/usr/bin/env bash
# Needle in a haystack with your own Claude Code session as the root agent,
# typed the way you would in a terminal: make a project, load the plugin, run
# /rlmflow:run, then read the run it saved. The README's plugin walkthrough as
# a script; claude_plugin.py is the same run driven from Python.
#
#   examples/needle/claude_plugin.sh                 # headless (claude -p), then checks the answer
#   examples/needle/claude_plugin.sh --interactive   # opens claude; you paste the /rlmflow:run line
#
# Needs `claude` (signed in), `rlmflow` on PATH (a source checkout's .venv works),
# and OPENAI_API_KEY for the workers. Override with NUM_FILES, WORKER_MODEL,
# BUDGET (the root session's dollar cap) and PROJECT (default: a new temp directory).
set -euo pipefail

repo="$(cd "$(dirname "$0")/../.." && pwd)"
# The plugin's hooks run whichever `rlmflow` is first on PATH: in a checkout, this one.
[[ -x "$repo/.venv/bin/rlmflow" ]] && export PATH="$repo/.venv/bin:$PATH"
command -v rlmflow >/dev/null || { echo "rlmflow is not on PATH (pip install rlmflow)" >&2; exit 1; }
command -v claude >/dev/null || { echo "claude is not on PATH" >&2; exit 1; }
: "${OPENAI_API_KEY:?the workers need OPENAI_API_KEY}"

num_files="${NUM_FILES:-100}"
project="${PROJECT:-$(mktemp -d)}"
mkdir -p "$project/haystack"
cd "$project"

# 1. A project with a haystack in it: one line in one file holds the number.
answer="$(python3 - "$num_files" <<'EOF'
import random, string, sys
n = int(sys.argv[1])
words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
answer = "".join(random.choices(string.digits, k=7))
hit, line = random.randrange(n), random.randrange(200)
for i in range(n):
    rows = [" ".join(random.choices(words, k=random.randint(3, 8))) for _ in range(200)]
    if i == hit:
        rows[line] = f"The magic number is {answer}"
    open(f"haystack/file_{i:04d}.txt", "w").write("\n".join(rows))
print(answer)
EOF
)"

# 2. The workers: rlmflow's default worker agent, on a small model.
cat > rlmflow.toml <<EOF
[run]
fast_model = "${WORKER_MODEL:-gpt-6.1-terra}"
EOF

task="There are $num_files text files in haystack/. Exactly one line in one file matches \
the pattern \`The magic number is <number>\`. Find the number and the file it's in. \
Use rlmflow agents to search the files in parallel."
plugin="$(rlmflow plugin path)"

if [[ "${1:-}" == "--interactive" ]]; then
    echo "project: $project"
    echo "In Claude Code, type:"
    echo
    echo "/rlmflow:run $task"
    echo
    exec claude --plugin-dir "$plugin"
fi

# 3. The same command, headless.
echo ">>> claude -p \"/rlmflow:run ...\" --plugin-dir \"\$(rlmflow plugin path)\"  in $project"
start=$SECONDS
reply="$(claude -p "/rlmflow:run $task" --plugin-dir "$plugin" \
    --dangerously-skip-permissions --max-budget-usd "${BUDGET:-2}")"
echo "$reply"

# 4. The run the plugin saved, as a tree and a timeline.
run="$(ls -d .rlmflow/runs/* 2>/dev/null | tail -n 1)"
[[ -n "$run" ]] || { echo "the plugin saved no run; see $project/.rlmflow/serve.log" >&2; exit 1; }
rlmflow view show "$run" | tee view.txt

workers="$(grep -c -- '── root\.' view.txt || true)"
echo
echo "needle   $answer"
echo "workers  $workers"
echo "seconds  $((SECONDS - start))"
echo "saved    $project/$run"
# The root can grep the haystack itself, so a right answer alone doesn't show the workers ran.
if grep -q "$answer" <<<"$reply" && [[ "$workers" -gt 0 ]]; then
    echo "correct"
else
    echo "wrong: expected $answer from a run with workers" >&2
    exit 1
fi
