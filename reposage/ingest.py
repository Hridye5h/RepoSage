"""Walk a repository, skip the junk, and turn every text/code file into Chunks."""
import os
from typing import Iterator, List

from .chunking import Chunk, chunk_code, lang_for
from .config import config

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
             "dist", "build", ".next", "target", ".idea", ".vscode",
             "qdrant_data", ".pytest_cache", ".mypy_cache", "site-packages"}

SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
            ".pdf", ".zip", ".tar", ".gz", ".bin", ".exe", ".dll", ".so",
            ".pyc", ".class", ".o", ".a", ".lib", ".woff", ".woff2", ".ttf",
            ".eot", ".mp4", ".mp3", ".lock", ".map", ".min.js", ".min.css"}

MAX_FILE_BYTES = 1_000_000  # skip files larger than 1 MB (generated/vendored)


def iter_files(root: str) -> Iterator[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        # prune skip-dirs in place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in SKIP_EXT:
                continue
            yield os.path.join(dirpath, fn)


def _fallback_chunks(source: str, rel: str) -> List[Chunk]:
    """Line-window chunking for non-code / unsupported text (README, JSON, etc.)."""
    out: List[Chunk] = []
    lines = source.splitlines(keepends=True)
    cur, cur_len, start = [], 0, 0
    for i, ln in enumerate(lines):
        if cur and cur_len + len(ln) > config.max_chunk_chars:
            out.append(Chunk("".join(cur).strip("\n"), rel, "text", start + 1, i, None))
            cur, cur_len, start = [], 0, i
        cur.append(ln)
        cur_len += len(ln)
    if cur:
        out.append(Chunk("".join(cur).strip("\n"), rel, "text", start + 1, len(lines), None))
    return [c for c in out if c.text.strip()]


def ingest_repo(root: str) -> List[Chunk]:
    root = os.path.abspath(root)
    chunks: List[Chunk] = []
    for path in iter_files(root):
        try:
            if os.path.getsize(path) > MAX_FILE_BYTES:
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()
        except OSError:
            continue
        if not source.strip():
            continue
        rel = os.path.relpath(path, root).replace("\\", "/")
        lang = lang_for(path)
        if lang:
            try:
                chunks.extend(chunk_code(source, rel, lang, config.max_chunk_chars))
            except Exception:
                # any parser hiccup -> don't lose the file, fall back to line windows
                chunks.extend(_fallback_chunks(source, rel))
        else:
            chunks.extend(_fallback_chunks(source, rel))
    return chunks
