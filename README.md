# RepoSage 🧙

**Ask questions about any codebase in plain English and get answers grounded in the
actual source — with file-and-line citations.**

RepoSage is a retrieval-augmented (RAG) assistant for code. It indexes a repository
with **AST-aware chunking**, retrieves with **hybrid search** (dense embeddings +
BM25, fused with Reciprocal Rank Fusion), and answers with an LLM that **cites the
exact files and lines** it used. A built-in **evaluation harness** measures answer
quality (faithfulness, context precision/recall) so every design choice is backed by
numbers — not vibes.

> Status: **Phase 1 complete** — ingestion + AST chunking + hybrid indexing. See the roadmap.

## Why it's built the way it is

| Decision | Rationale |
|---|---|
| **AST chunking** (tree-sitter, cAST split-then-merge) | Fixed-size splitting cuts functions mid-token and wrecks retrieval. Keeping syntactic units intact lifts Recall@5 and downstream answer quality on code-RAG benchmarks. |
| **Hybrid retrieval** (dense + BM25, RRF) | Dense embeddings win natural-language→code questions; BM25 catches exact symbol names. RRF fuses them — the de-facto standard. |
| **Qdrant** (local on-disk → server) | Server-side hybrid Query API; runs with zero infra locally, scales to a cluster unchanged. |
| **Selective agentic loop** (later) | An adaptive self-RAG loop helps hard multi-hop questions but *hurts* simple ones via query drift. Applied only when a query needs it. |
| **Evaluation harness** (the centerpiece) | RAGAS-style metrics + a golden Q&A set prove each choice (AST vs fixed-size, hybrid vs dense-only, rerank on/off). |

## Architecture

```
        ┌─────────── INGEST ───────────┐      ┌──────────── ASK ────────────┐
repo ──▶ walk files ─▶ AST chunk ─▶ embed   query ─▶ embed ─▶ hybrid search
         (tree-sitter)  (dense+BM25)          (dense+BM25)  ─▶ RRF fuse
                              │                              ─▶ (rerank)
                              ▼                              ─▶ LLM answer + citations
                          Qdrant  ◀───────────────────────────────┘
                       (dense + sparse vectors)
```

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate     # (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env                                # add your ANTHROPIC_API_KEY (needed Phase 2)

python -m reposage.cli index <path-to-a-repo>       # index a codebase
python -m reposage.cli stats                        # how many chunks are indexed
pytest                                              # run the chunker tests
```

The first index downloads small embedding models (~100 MB) once, then runs on CPU.

## Roadmap

- [x] **Phase 1** — ingestion, AST-aware chunking, hybrid (dense + BM25) indexing
- [ ] **Phase 2** — hybrid retrieval + RRF + grounded, cited answers (Claude)
- [ ] **Phase 3** — RAGAS evaluation harness + golden question set + config comparison
- [ ] **Phase 4** — selective agentic loop (LangGraph self-RAG)
- [ ] **Phase 5** — FastAPI + minimal web UI + deployment

## License
MIT
