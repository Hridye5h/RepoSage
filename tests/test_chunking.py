"""The chunker's contract: it must keep whole functions/classes intact and tag
each chunk with its enclosing symbol — that's the whole point of AST chunking."""
from reposage.chunking import chunk_code

SAMPLE = '''\
import os


def add(a, b):
    """Add two numbers."""
    return a + b


class Calculator:
    def __init__(self, start=0):
        self.value = start

    def multiply(self, x):
        self.value *= x
        return self.value
'''


def test_small_functions_stay_whole():
    chunks = chunk_code(SAMPLE, "calc.py", "python", max_chars=1500)
    assert chunks, "should produce at least one chunk"
    # the add() function must appear intact inside a single chunk
    assert any("def add(a, b):" in c.text and "return a + b" in c.text for c in chunks)
    # every chunk carries its file + a valid line range
    for c in chunks:
        assert c.file_path == "calc.py"
        assert c.start_line <= c.end_line


def test_symbols_are_tagged():
    chunks = chunk_code(SAMPLE, "calc.py", "python", max_chars=1500)
    symbols = {c.symbol for c in chunks if c.symbol}
    # at least one chunk should be tagged with a top-level symbol name
    assert symbols & {"add", "Calculator"}


def test_oversize_class_splits_into_methods():
    # tiny cap forces the class to split; methods should become separate chunks
    chunks = chunk_code(SAMPLE, "calc.py", "python", max_chars=80)
    assert len(chunks) >= 3, "tight cap should split the class into multiple chunks"
