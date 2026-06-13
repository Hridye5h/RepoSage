"""Command-line entry point.

  python -m reposage.cli index <repo_path>     # ingest + chunk + index a repo
  python -m reposage.cli stats                  # show what's indexed
"""
import sys
import time
from collections import Counter

from .config import config
from .index import get_client, index_chunks
from .ingest import ingest_repo


def cmd_index(repo: str):
    t0 = time.time()
    print(f"[1/2] Ingesting + AST-chunking  {repo}")
    chunks = ingest_repo(repo)
    langs = Counter(c.language for c in chunks)
    print(f"      {len(chunks)} chunks  ({dict(langs)})")
    if not chunks:
        print("      nothing to index — is the path right?")
        return
    print(f"[2/2] Embedding (dense + BM25) + indexing into Qdrant ...")
    n = index_chunks(chunks)
    print(f"Done: indexed {n} chunks in {time.time() - t0:.1f}s")
    print(f"      collection='{config.collection}'  store='{config.qdrant_url or config.qdrant_path}'")


def cmd_stats():
    client = get_client()
    if not client.collection_exists(config.collection):
        print("No collection yet — run `index <repo>` first.")
        return
    info = client.get_collection(config.collection)
    print(f"collection='{config.collection}'  points={info.points_count}")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "index":
        cmd_index(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "stats":
        cmd_stats()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
