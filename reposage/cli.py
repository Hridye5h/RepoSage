"""Command-line entry point.

  python -m reposage.cli index <repo_path>     # ingest + chunk + index a repo
  python -m reposage.cli stats                  # show what's indexed
  python -m reposage.cli ask "<question>"            # hybrid-retrieve and answer
  python -m reposage.cli ask "<question>" --agentic  # route hard queries through self-RAG
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
    print("[2/2] Embedding (dense + BM25) + indexing into Qdrant ...")
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


def cmd_ask(question: str, agentic: bool = False):
    from .retrieve import Retriever

    r = Retriever()

    if agentic:
        if not config.llm_ready:
            print("agentic mode needs an LLM key in .env")
            return
        from .agentic import agentic_answer
        ans, path = agentic_answer(question, r)
        print(f"\n[routed: {path}]\n\n--- Answer ---\n{ans}")
        return

    hits = r.search(question)
    if not hits:
        print("No relevant code found — did you index a repo first?")
        return

    print(f"\nTop {len(hits)} sources (hybrid dense+BM25, RRF-fused):")
    for h in hits:
        sym = f"  ({h.symbol})" if h.symbol else ""
        print(f"  {h.score:.3f}  {h.file_path}:{h.start_line}-{h.end_line}{sym}")

    if not config.llm_ready:
        key = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(
            config.llm_provider, "an API key"
        )
        print(f"\n[retrieval only — set {key} in .env to generate an answer "
              f"(provider={config.llm_provider})]")
        return

    from .generate import answer

    print("\n--- Answer ---")
    print(answer(question, hits))


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "index":
        cmd_index(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "stats":
        cmd_stats()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        args = sys.argv[2:]
        agentic = "--agentic" in args
        args = [a for a in args if a != "--agentic"]
        cmd_ask(" ".join(args), agentic=agentic)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
