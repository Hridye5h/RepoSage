"""Retrieval-quality eval (NO LLM — free + instant).

For each golden question we retrieve top-k chunks and check whether the
ground-truth file(s) show up, scoring three standard IR metrics:

- Hit@k    : did ANY correct file appear in the top-k?            (coverage)
- MRR      : 1 / rank of the first correct file (0 if missed)     (ranking)
- Recall@k : fraction of the question's correct files retrieved   (completeness)

We run it under three retrieval modes so the numbers PROVE that hybrid beats
either dense-only or sparse-only on this codebase.

Run:  python -m reposage.eval.retrieval_eval
"""
from reposage.eval.golden import GOLDEN
from reposage.retrieve import Retriever


def evaluate(retriever: Retriever, mode: str, k: int = 8):
    hit = mrr = recall = 0.0
    for item in GOLDEN:
        files = [h.file_path for h in retriever.search(item["q"], top_k=k, mode=mode)]
        expected = item["files"]
        first_rank = next((i + 1 for i, f in enumerate(files) if f in expected), None)
        if first_rank:
            hit += 1
            mrr += 1.0 / first_rank
        recall += len(expected & set(files)) / len(expected)
    n = len(GOLDEN)
    return hit / n, mrr / n, recall / n


def main():
    k = 8
    r = Retriever()
    print(f"Retrieval eval — {len(GOLDEN)} golden questions, top_k={k}\n")
    print(f"{'mode':<10}{'Hit@'+str(k):>9}{'MRR':>9}{'Recall@'+str(k):>11}")
    print("-" * 39)
    for mode in ("sparse", "dense", "hybrid"):
        h, m, rc = evaluate(r, mode, k)
        print(f"{mode:<10}{h:>9.2f}{m:>9.3f}{rc:>11.2f}")
    print("\n(higher is better; hybrid should match or beat the single-retriever rows)")


if __name__ == "__main__":
    main()
