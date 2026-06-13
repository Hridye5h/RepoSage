"""Prove AST chunking beats naive fixed-size splitting — with retrieval numbers,
no LLM needed.

Builds two indexes from the SAME repo (AST = ours, vs a fixed-size baseline) and
evaluates both on the golden set. Shares ONE Qdrant client + embedder, because
Qdrant local mode locks the on-disk store to a single client per process.

Run:  python -m reposage.eval.chunking_eval
"""
from reposage.config import config
from reposage.embeddings import Embedder
from reposage.eval.retrieval_eval import evaluate
from reposage.index import get_client, index_chunks
from reposage.ingest import ingest_repo
from reposage.retrieve import Retriever

REPO = "D:/Tracker_improve"          # the repo the golden set is written for
AST_COLL = config.collection          # already indexed (AST) from earlier runs
FIXED_COLL = "reposage_fixed"


def main():
    client = get_client()
    emb = Embedder()

    print(f"Building fixed-size baseline index from {REPO} ...")
    index_chunks(ingest_repo(REPO, chunker="fixed"), collection=FIXED_COLL, client=client)

    print("\nRetrieval eval (hybrid, top_k=8) — fixed-size vs AST chunking\n")
    print(f"{'chunker':<13}{'chunks':>8}{'Hit@8':>9}{'MRR':>9}{'Recall@8':>11}")
    print("-" * 50)
    for label, coll in [("fixed-size", FIXED_COLL), ("AST (ours)", AST_COLL)]:
        r = Retriever(collection=coll, client=client, emb=emb)
        h, m, rc = evaluate(r, mode="hybrid", k=8)
        n = client.get_collection(coll).points_count
        print(f"{label:<13}{n:>8}{h:>9.2f}{m:>9.3f}{rc:>11.2f}")
    print("\n(AST keeps functions/classes intact -> better ranking & recall)")


if __name__ == "__main__":
    main()
