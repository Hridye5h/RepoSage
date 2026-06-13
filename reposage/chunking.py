"""AST-aware code chunking (cAST: split-then-merge), per arXiv 2506.15655.

WHY THIS EXISTS (the interview answer):
Fixed-size character splitting cuts code at arbitrary byte offsets — a chunk can
end mid-identifier ("subtotal += item.qu") so the embedding model never sees a
complete thought. AST chunking keeps whole syntactic units (functions, classes)
intact. On code-RAG benchmarks this measurably lifts retrieval (+4.3 Recall@5)
and downstream generation (+2.67 Pass@1).

THE ALGORITHM (cAST):
1. Parse the file into a syntax tree with tree-sitter.
2. Walk the tree's nodes. GREEDILY MERGE consecutive sibling nodes while their
   combined span stays under a size cap (so small statements group together).
3. If a single node is larger than the cap, SPLIT it by recursing into its
   children (e.g. a big class -> its methods become their own chunks).
4. If an oversize node is a leaf, fall back to line-window splits.
Each chunk carries metadata: file path, language, line range, enclosing symbol.

NOTE on the binding: tree-sitter-language-pack (1.8.x) exposes a method-style API
(parse(str); tree.root_node(); node.kind(); node.child(i); node.start_byte()),
so accessors are *called*. Line numbers are derived from UTF-8 byte offsets here,
which is binding-independent and robust.
"""
import bisect
import os
from dataclasses import dataclass
from typing import List, Optional

from tree_sitter_language_pack import get_parser

# File extension -> tree-sitter language name.
EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".java": "java", ".go": "go",
    ".rs": "rust", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".c": "c", ".h": "c", ".rb": "ruby", ".cs": "csharp", ".php": "php",
}

# Node kinds that introduce a named symbol, per language (best-effort).
SYMBOL_NODES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "impl_item", "trait_item"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
    "c": {"function_definition", "struct_specifier"},
    "ruby": {"method", "class", "module"},
    "csharp": {"method_declaration", "class_declaration"},
    "php": {"function_definition", "method_declaration", "class_declaration"},
}


@dataclass
class Chunk:
    text: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol: Optional[str] = None


def lang_for(path: str) -> Optional[str]:
    return EXT_LANG.get(os.path.splitext(path)[1].lower())


def _children(node) -> list:
    return [node.child(i) for i in range(node.child_count())]


def _symbol_name(src: bytes, node, lang: str) -> Optional[str]:
    """Pull the identifier out of a symbol-defining node (function/class name)."""
    if node.kind() not in SYMBOL_NODES.get(lang, set()):
        return None
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return src[name_node.start_byte():name_node.end_byte()].decode("utf-8", "ignore")
    return None


def chunk_code(source: str, file_path: str, language: str, max_chars: int) -> List[Chunk]:
    parser = get_parser(language)
    src = source.encode("utf-8")
    root = parser.parse(source).root_node()   # parse() takes str; root_node() is a method

    # byte offset -> 0-indexed line, via newline byte positions
    newline_pos = [i for i, b in enumerate(src) if b == 0x0A]

    def row_of(byte_off: int) -> int:
        return bisect.bisect_left(newline_pos, byte_off)

    chunks: List[Chunk] = []

    def add_text(text: str, start_row: int, end_row: int, symbol: Optional[str]):
        text = text.strip("\n")
        if text.strip():
            chunks.append(Chunk(text, file_path, language, start_row + 1, end_row + 1, symbol))

    def add_span(sb: int, eb: int, symbol: Optional[str]):
        end_row = row_of(eb - 1) if eb > sb else row_of(sb)
        add_text(src[sb:eb].decode("utf-8", "ignore"), row_of(sb), end_row, symbol)

    def line_split(node, symbol):
        """Last resort: a leaf bigger than the cap -> split it into line windows."""
        sb = node.start_byte()
        text = src[sb:node.end_byte()].decode("utf-8", "ignore")
        base = row_of(sb)
        lines = text.splitlines(keepends=True)
        cur, cur_len, seg_start = [], 0, 0
        for i, ln in enumerate(lines):
            ln_len = len(ln.encode("utf-8"))
            if cur and cur_len + ln_len > max_chars:
                add_text("".join(cur), base + seg_start, base + i - 1, symbol)
                cur, cur_len, seg_start = [], 0, i
            cur.append(ln)
            cur_len += ln_len
        if cur:
            add_text("".join(cur), base + seg_start, base + len(lines) - 1, symbol)

    def process(nodes, inherited_symbol):
        """Each definition (function/class) becomes its own chunk tagged with its
        name; other consecutive siblings (imports, statements) greedily merge."""
        buf = []  # buffered consecutive NON-definition siblings

        def flush():
            nonlocal buf
            if buf:
                add_span(buf[0].start_byte(), buf[-1].end_byte(), inherited_symbol)
                buf = []

        for n in nodes:
            sym = _symbol_name(src, n, language)
            # qualify nested names, e.g. methods become "Calculator.multiply"
            full_sym = f"{inherited_symbol}.{sym}" if (inherited_symbol and sym) else (sym or inherited_symbol)
            size = n.end_byte() - n.start_byte()

            if size > max_chars:
                flush()
                kids = _children(n)
                if kids:
                    process(kids, full_sym)     # SPLIT: recurse into the big node
                else:
                    line_split(n, full_sym)     # huge leaf -> line windows
                continue

            if sym is not None:
                # a definition that fits -> give it its own chunk
                flush()
                add_span(n.start_byte(), n.end_byte(), full_sym)
                continue

            # non-definition node -> greedily merge with neighbours under the cap
            if buf and (n.end_byte() - buf[0].start_byte()) > max_chars:
                flush()
            buf.append(n)
        flush()

    process(_children(root), None)
    return chunks


def fixed_size_chunks(source: str, file_path: str, language: str, max_chars: int) -> List[Chunk]:
    """Naive baseline chunker: fixed-size CHARACTER windows that ignore code
    structure (so they routinely cut functions and identifiers in half). This is
    the 'bad' baseline AST chunking is designed to beat — the eval uses it to
    prove the design choice with numbers."""
    out: List[Chunk] = []
    for i in range(0, len(source), max_chars):
        seg = source[i:i + max_chars]
        if not seg.strip():
            continue
        start_line = source.count("\n", 0, i) + 1
        end_line = source.count("\n", 0, i + len(seg)) + 1
        out.append(Chunk(seg, file_path, language, start_line, end_line, None))
    return out
