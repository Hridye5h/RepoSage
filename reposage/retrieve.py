"""Hybrid retrieval: dense + BM25, fused server-side with Reciprocal Rank Fusion.

WHY HYBRID (the interview answer):
- Dense embeddings capture meaning -> great for "where is auth handled?" (natural
  language -> code), which is the dominant "chat with codebase" case.
- BM25 sparse matches exact tokens -> catches a specific symbol name like
  `getSignedJwtToken` that a dense model might paraphrase away.
Fusing both with RRF (the de-facto standard) gets the best of each. Qdrant's
Query API does the prefetch + fusion in ONE server-side call.
"""
from dataclasses import dataclass
from typing import List, Optional

from qdrant_client import models

from .config import config
from .embeddings import Embedder
from .index import get_client


@dataclass
class Hit:
    text: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    symbol: Optional[str]
    score: float


class Retriever:
    def __init__(self):
        self.emb = Embedder()
        self.client = get_client()

    def search(self, query: str, top_k: int = 8, prefetch_k: int = 25) -> List[Hit]:
        dense = self.emb.embed_dense_one(query).tolist()
        sparse = self.emb.embed_sparse_one(query)

        result = self.client.query_points(
            collection_name=config.collection,
            prefetch=[
                # branch 1: dense semantic search
                models.Prefetch(query=dense, using="dense", limit=prefetch_k),
                # branch 2: BM25 keyword search
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                    using="sparse",
                    limit=prefetch_k,
                ),
            ],
            # fuse the two ranked lists with Reciprocal Rank Fusion
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        hits: List[Hit] = []
        for p in result.points:
            pl = p.payload or {}
            hits.append(
                Hit(
                    text=pl.get("text", ""),
                    file_path=pl.get("file_path", "?"),
                    language=pl.get("language", ""),
                    start_line=pl.get("start_line", 0),
                    end_line=pl.get("end_line", 0),
                    symbol=pl.get("symbol"),
                    score=p.score,
                )
            )
        return hits
