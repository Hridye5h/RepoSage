"""Dense + sparse embeddings via fastembed (ONNX, CPU-friendly, no torch needed).

Dense vectors capture meaning (great for natural-language -> code questions).
Sparse BM25 vectors capture exact tokens (great for matching a specific symbol
name). We index BOTH and fuse them at query time — that's hybrid search.
"""
from typing import List

from fastembed import SparseTextEmbedding, TextEmbedding

from .config import config


class Embedder:
    def __init__(self):
        self.dense = TextEmbedding(model_name=config.dense_model)
        self.sparse = SparseTextEmbedding(model_name=config.sparse_model)
        # Probe the dense dimension once (needed to size the Qdrant collection).
        self.dense_dim = len(next(iter(self.dense.embed(["dimension probe"]))))

    def embed_dense(self, texts: List[str]):
        return list(self.dense.embed(texts))

    def embed_sparse(self, texts: List[str]):
        return list(self.sparse.embed(texts))

    def embed_dense_one(self, text: str):
        return next(iter(self.dense.embed([text])))

    def embed_sparse_one(self, text: str):
        return next(iter(self.sparse.embed([text])))
