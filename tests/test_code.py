"""Phase 0 — code-block parsing + ``await`` pre-flight validation.

Ported from the legacy code-block coverage to the new ``rflow.runtime.code``
module. Covers fence extraction edge
cases (glued fences, EOF, bare labels), first-repl-block replacement, and the
full ``check_wait_syntax`` matrix that turns unsupported ``await``/``yield`` into
a single recoverable error string before the block ever runs.
"""

from __future__ import annotations

from rflow.runtime.code import (
    check_wait_syntax,
    find_code_blocks,
    replace_code_block,
)

# ── find_code_blocks ──────────────────────────────────────────────────


def test_standard_fence():
    text = "before\n```repl\nx = 1\nprint(x)\n```\nafter"
    assert find_code_blocks(text) == ["x = 1\nprint(x)"]


def test_python_and_bare_fences():
    assert find_code_blocks("```python\nx = 1\n```") == ["x = 1"]
    assert find_code_blocks("```\ny = 2\n```") == ["y = 2"]


def test_glued_fence_then_text():
    # closing fence glued to the last line of code, trailing prose after.
    text = "```repl\ndone('ok')```\ntrailing"
    assert find_code_blocks(text) == ["done('ok')"]


def test_fence_at_eof():
    assert find_code_blocks("```repl\nx = 1\n```") == ["x = 1"]


def test_glued_fence_at_eof():
    assert find_code_blocks("```repl\ndone('x')```") == ["done('x')"]


def test_no_blocks():
    assert find_code_blocks("just some prose, no fences") == []


def test_bare_repl_label_is_not_a_code_block():
    # "repl" as a word, not an opening fence, yields nothing.
    assert find_code_blocks("use the repl to compute things") == []


def test_nested_backticks_do_not_truncate():
    text = '```repl\ns = """\n```bash\nls\n```\n"""\nprint(s)\n```'
    (block,) = find_code_blocks(text)
    assert "```bash" in block and "print(s)" in block


def test_two_openers_coalesce_under_greedy_last_close():
    # Documented greedy rule: the LAST closing fence after an opener wins, so
    # two ```repl blocks with prose between them merge into one captured block.
    text = "```repl\na = 1\n```\nmid\n```repl\nb = 2\n```"
    (block,) = find_code_blocks(text)
    assert block.startswith("a = 1") and block.endswith("b = 2")


# ── replace_code_block ────────────────────────────────────────────────


def test_replace_standard_targets_first_repl_block():
    text = "intro\n```repl\nold = 1\n```\ntail"
    assert replace_code_block(text, "new = 2") == "intro\n```repl\nnew = 2\n```"


def test_replace_glued_fence():
    text = "```repl\nold()```\nafter"
    assert replace_code_block(text, "new()") == "```repl\nnew()\n```"


def test_replace_no_block_is_identity():
    assert replace_code_block("no block here", "x = 1") == "no block here"


# ── check_wait_syntax ─────────────────────────────────────────────────


def test_wait_check_accepts_direct_await():
    assert check_wait_syntax("rs = await launch_subagents([{'query': 'q'}])") is None


def test_wait_check_accepts_conditional_await():
    assert (
        check_wait_syntax(
            "x = (await launch_subagents([{'query': 'q'}])) if cond else None"
        )
        is None
    )


def test_wait_check_leaves_yield_to_python_execution():
    assert check_wait_syntax("x = yield h") is None


def test_wait_check_allows_generator_functions():
    assert check_wait_syntax("def g():\n    yield 1\nprint(sum(g()))") is None


def test_wait_check_rejects_naked_wait():
    err = check_wait_syntax("flow_wait(h)")
    assert err is not None and "internal" in err


def test_wait_check_rejects_direct_flow_wait():
    err = check_wait_syntax("x = await flow_wait(h)")
    assert err is not None and "internal" in err


def test_wait_check_rejects_direct_flow_delegate():
    err = check_wait_syntax('h = flow_delegate(name="x", query="q")')
    assert err is not None and "internal" in err


def test_wait_check_rejects_naked_launch_subagents():
    err = check_wait_syntax("launch_subagents([{'query': 'q'}])")
    assert err is not None and "must be awaited" in err


def test_wait_check_rejects_internal_wait_in_comprehension():
    err = check_wait_syntax("[await flow_wait(h) for h in hs]")
    assert err is not None and "internal" in err


def test_wait_check_accepts_nested_async_launch():
    assert check_wait_syntax("async def f():\n    return await launch_subagents([])") is None


def test_wait_check_allows_unknown_await_for_runtime_driver():
    assert check_wait_syntax("x = await something_else()") is None


def test_wait_check_allows_regular_asyncio_await():
    assert check_wait_syntax("import asyncio\nawait asyncio.sleep(0)") is None


def test_wait_check_ignores_plain_code():
    assert check_wait_syntax("x = 1\nprint(x)\ndone('ok')") is None


def test_wait_check_returns_none_on_syntax_error():
    # a genuine SyntaxError is left for the normal execution path to report.
    assert check_wait_syntax("def (:") is None
