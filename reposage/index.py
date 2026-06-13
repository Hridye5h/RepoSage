"""Index chunks into Qdrant with BOTH a dense vector and a BM25 sparse vector,
so we can do hybrid retrieval (fused with RRF) at query time.

Local mode (QdrantClient(path=...)) needs zero infrastructure — great for dev.
Set REPOSAGE_QDRANT_URL to point at a real server/cluster for production.

`collection` is parameterised so the eval can build separate indexes (AST vs
fixed-size chunking) and compare them.
"""
import uuid
from typing import List, Optional

from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .config import config
from .embeddings import Embedder


def get_client() -> QdrantClient:
    if config.qdrant_url:
        return QdrantClient(url=config.qdrant_url)
    return QdrantClient(path=config.qdrant_path)  # on-disk local store


def ensure_collection(client: QdrantClient, dim: int, collection: str, recreate: bool = False):
    exists = client.collection_exists(collection)
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )


def index_chunks(chunks: List[Chunk], collection: Optional[str] = None,
                 recreate: bool = True, batch: int = 64, client=None) -> int:
    collection = collection or config.collection
    emb = Embedder()
    client = client or get_client()
    ensure_collection(client, emb.dense_dim, collection, recreate=recreate)

    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        texts = [c.text for c in part]
        dense_vecs = emb.embed_dense(texts)
        sparse_vecs = emb.embed_sparse(texts)

        points = []
        for c, d, s in zip(part, dense_vecs, sparse_vecs):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": d.tolist(),
                        "sparse": models.SparseVector(
                            indices=s.indices.tolist(), values=s.values.tolist()
                        ),
                    },
                    payload={
                        "text": c.text,
                        "file_path": c.file_path,
                        "language": c.language,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "symbol": c.symbol,
                    },
                )
            )
        client.upsert(collection_name=collection, points=points)
    return len(chunks)
