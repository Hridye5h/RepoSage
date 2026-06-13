"""Hybrid retrieval: dense + BM25, fused server-side with Reciprocal Rank Fusion.

WHY HYBRID (the interview answer):
- Dense embeddings capture meaning -> great for "where is auth handled?" (natural
  language -> code), the dominant "chat with codebase" case.
- BM25 sparse matches exact tokens -> catches a specific symbol name like
  `getSignedJwtToken` that a dense model might paraphrase away.
Fusing both with RRF (the de-facto standard) gets the best of each. Qdrant's
Query API does the prefetch + fusion in ONE server-side call.

`mode` ("hybrid"|"dense"|"sparse") exists so the eval harness can prove the
hybrid design beats either retriever alone — same index, different query.
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

    def _sparse_vector(self, query: str) -> models.SparseVector:
        s = self.emb.embed_sparse_one(query)
        return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())

    def search(self, query: str, top_k: int = 8, prefetch_k: int = 25, mode: str = "hybrid") -> List[Hit]:
        if mode == "dense":
            result = self.client.query_points(
                collection_name=config.collection,
                query=self.emb.embed_dense_one(query).tolist(),
                using="dense", limit=top_k, with_payload=True,
            )
        elif mode == "sparse":
            result = self.client.query_points(
                collection_name=config.collection,
                query=self._sparse_vector(query),
                using="sparse", limit=top_k, with_payload=True,
            )
        else:  # hybrid: dense + BM25 prefetch, fused with RRF
            result = self.client.query_points(
                collection_name=config.collection,
                prefetch=[
                    models.Prefetch(query=self.emb.embed_dense_one(query).tolist(),
                                    using="dense", limit=prefetch_k),
                    models.Prefetch(query=self._sparse_vector(query),
                                    using="sparse", limit=prefetch_k),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k, with_payload=True,
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
