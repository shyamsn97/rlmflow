"""A stand-in for ``claude``: runs scripted shell commands as Claude's turns.

``FAKE_CLAUDE_SCRIPT`` is a JSON list of ``[needle, commands, reply]``; the first
entry whose needle is in the turn's input runs. Every call's argv is appended to
``FAKE_CLAUDE_LOG`` as one JSON line.

``claude --input-format stream-json ...`` is the Agent SDK's headless turn: it
answers the SDK's ``initialize`` request, reads the prompt from stdin and writes
the turn's result. The log line is that argv with ``--name=value`` split in two
and ``-p PROMPT`` appended, so tests can read flags uniformly.

``claude --interactive --plugin-dir DIR [--resume ID [--fork-session]] PROMPT...``
plays your own session with the plugin loaded: per prompt it runs the plugin's
``UserPromptSubmit`` hooks (their output joins the turn's input), the scripted
commands, then the ``Stop`` hooks, taking a ``block`` reason as the next turn's
input; ``SessionEnd`` fires at the end. Each reply goes to a transcript JSONL.
"""

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def scripted(text):
    for needle, commands, reply in json.loads(os.environ["FAKE_CLAUDE_SCRIPT"]):
        if needle in text:
            return commands, reply
    raise SystemExit(f"no scripted turn for {text!r}")


def run_commands(commands):
    outputs = []
    for command in commands:
        done = subprocess.run(command, shell=True, capture_output=True, text=True)
        outputs.append(done.stdout + done.stderr)
    return outputs


def reply_text(reply, outputs):
    return reply.replace("{outputs}", " | ".join(o.strip() for o in outputs))


def write(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def normalized(argv):
    out = []
    for arg in argv:
        out += arg.split("=", 1) if arg.startswith("--") and "=" in arg else [arg]
    return out


def headless(argv):
    prompt = None
    for line in sys.stdin:
        message = json.loads(line)
        if message.get("type") == "control_request":
            request_id = message["request_id"]
            write(
                {
                    "type": "control_response",
                    "response": {"subtype": "success", "request_id": request_id, "response": {}},
                }
            )
        elif message.get("type") == "user":
            prompt = message["message"]["content"]
            break
    log(normalized(argv) + ["-p", prompt])
    commands, reply = scripted(prompt)
    outputs = run_commands(commands)
    session = flag(argv, "--session-id") or flag(argv, "--resume")
    if "--fork-session" in argv:
        session = str(uuid.uuid4())
    text = reply_text(reply, outputs)
    result = {
        "type": "result",
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": 1,
        "session_id": session,
        "result": text,
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    if "--json-schema" in argv:
        result["structured_output"] = json.loads(text)
    write(result)
    for _ in sys.stdin:
        pass
    return 0


def log(argv):
    with open(os.environ["FAKE_CLAUDE_LOG"], "a") as handle:
        handle.write(json.dumps(argv) + "\n")


class Plugin:
    def __init__(self, plugin_dir):
        self.hooks = json.loads((Path(plugin_dir) / "hooks" / "hooks.json").read_text())["hooks"]

    def fire(self, event, data):
        outputs = []
        for group in self.hooks.get(event, []):
            for hook in group["hooks"]:
                done = subprocess.run(
                    hook["command"],
                    shell=True,
                    input=json.dumps({**data, "hook_event_name": event}),
                    capture_output=True,
                    text=True,
                    timeout=hook.get("timeout", 60),
                )
                if done.stderr:
                    sys.stderr.write(done.stderr)
                outputs.append(done.stdout.strip())
        return [o for o in outputs if o]


def interactive(argv):
    plugin = Plugin(flag(argv, "--plugin-dir"))
    cwd = os.getcwd()
    transcripts = Path(cwd) / ".fake-claude"
    transcripts.mkdir(exist_ok=True)
    session = flag(argv, "--resume") or flag(argv, "--session-id") or str(uuid.uuid4())
    transcript = transcripts / f"{session}.jsonl"
    if "--fork-session" in argv:
        forked = str(uuid.uuid4())
        if transcript.exists():
            shutil.copy(transcript, transcripts / f"{forked}.jsonl")
        session, transcript = forked, transcripts / f"{forked}.jsonl"
    transcript.touch()
    base = {"session_id": session, "transcript_path": str(transcript), "cwd": cwd}

    skip = {"--interactive", "--fork-session"}
    valued = {"--plugin-dir", "--resume", "--session-id"}
    prompts, i = [], 0
    while i < len(argv):
        if argv[i] in valued:
            i += 2
            continue
        if argv[i] not in skip:
            prompts.append(argv[i])
        i += 1

    for prompt in prompts:
        context = plugin.fire("UserPromptSubmit", {**base, "prompt": prompt})
        turn_input, active = "\n\n".join([prompt, *context]), False
        while True:
            commands, reply = scripted(turn_input)
            text = reply_text(reply, run_commands(commands))
            with transcript.open("a") as handle:
                entry = {
                    "type": "assistant",
                    "requestId": str(uuid.uuid4()),
                    "message": {
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [{"type": "text", "text": text}],
                    },
                }
                handle.write(json.dumps(entry) + "\n")
            decisions = plugin.fire(
                "Stop", {**base, "stop_hook_active": active, "last_assistant_message": text}
            )
            blocks = [json.loads(d) for d in decisions if d.startswith("{")]
            blocks = [d for d in blocks if d.get("decision") == "block"]
            if not blocks:
                break
            turn_input, active = blocks[0]["reason"], True
    plugin.fire("SessionEnd", {**base, "reason": "other"})
    print(json.dumps({"session_id": session}))
    return 0


def main():
    argv = normalized(sys.argv[1:])
    if argv == ["-v"]:
        print("2.1.310 (Claude Code)")
        return 0
    if "--interactive" in argv:
        log(argv)
        return interactive(argv)
    return headless(argv)


if __name__ == "__main__":
    sys.exit(main())
