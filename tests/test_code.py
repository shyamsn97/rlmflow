
from rflow import (
    code_block,
    find_code_blocks,
)



def test_minimal_code_block_extraction_handles_runtime_edge_cases():
    assert code_block("```repl   \ndone('ok')```\ntrailing") == "done('ok')"
    assert code_block("```python\nx = 1\n```") == "x = 1"

    text = '```repl\ns = """\n```bash\nls\n```\n"""\nprint(s)\n```'
    assert "```bash" in code_block(text)
    assert find_code_blocks("no block") == []

