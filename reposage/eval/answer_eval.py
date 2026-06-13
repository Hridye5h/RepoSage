"""LLM-judged answer-quality eval (RAGAS-style metrics), free-tier friendly.

For a small set of golden questions: retrieve -> generate -> have the LLM judge
two reference-free metrics in ONE call:
  - faithfulness     : are the answer's claims grounded in the retrieved context?
                       (the anti-hallucination metric)
  - answer_relevancy : does the answer actually address the question?

These are the same dimensions RAGAS measures; we call the judge directly (Gemini)
to keep it fast, free, and fully transparent. Throttled + retried for free-tier
rate limits. (Swapping in the `ragas` library later is a drop-in upgrade.)

Run:  python -m reposage.eval.answer_eval
"""
import re
import time

from reposage.config import config
from reposage.eval.golden import GOLDEN
from reposage.generate import answer, build_context, complete
from reposage.retrieve import Retriever

SUBSET = GOLDEN[:6]   # keep total LLM calls modest for the free tier
SLEEP = 4             # spacing between calls (free-tier RPM is limited)

JUDGE_SYS = (
    "You are a strict RAG evaluator. You receive a QUESTION, the CONTEXT (retrieved "
    "code the answer was based on), and the ANSWER. Output exactly two scores in [0,1]:\n"
    "FAITHFULNESS = fraction of the answer's claims supported by the context "
    "(1.0 = fully grounded, 0.0 = hallucinated).\n"
    "RELEVANCY = how directly the answer addresses the question (1.0 = fully, 0.0 = off-topic).\n"
    "Reply on ONE line, EXACTLY: faithfulness=<num> relevancy=<num>"
)


def _scores(text: str):
    f = re.search(r"faithfulness\s*=\s*([01](?:\.\d+)?)", text or "", re.I)
    r = re.search(r"relevancy\s*=\s*([01](?:\.\d+)?)", text or "", re.I)
    return (float(f.group(1)) if f else 0.0, float(r.group(1)) if r else 0.0)


_TRANSIENT = ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "500", "INTERNAL")


def _retry(fn, tries: int = 4):
    """Retry transient API errors (free-tier 429, 503 high-demand) with backoff.
    Returns '' after giving up so one flaky call can't crash the whole eval."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if any(s in str(e) for s in _TRANSIENT) and i < tries - 1:
                time.sleep(20 + 15 * i)
                continue
            return ""
    return ""


def main():
    r = Retriever()
    faiths, rels = [], []
    print(f"LLM-judged answer eval (RAGAS-style) — {len(SUBSET)} questions, "
          f"judge={config.llm_provider}/{config.gemini_model}\n")
    for item in SUBSET:
        q = item["q"]
        hits = r.search(q, top_k=6)
        ctx = build_context(hits)

        time.sleep(SLEEP)
        ans = _retry(lambda: answer(q, hits))
        if not ans:
            print(f"  (skipped — API unavailable)    {q[:48]}")
            continue

        time.sleep(SLEEP)
        judge = _retry(lambda: complete(
            JUDGE_SYS, f"QUESTION: {q}\n\nCONTEXT:\n{ctx}\n\nANSWER:\n{ans}", max_tokens=50
        ))
        if not judge:
            print(f"  (skipped — judge unavailable)  {q[:48]}")
            continue
        f, rl = _scores(judge)
        faiths.append(f)
        rels.append(rl)
        print(f"  faithfulness={f:.2f}  relevancy={rl:.2f}   {q[:48]}")

    n = len(faiths)
    if n == 0:
        print("\nNo questions scored (API unavailable). Try again in a minute.")
        return
    print("\n" + "-" * 40)
    print(f"questions scored      : {n}/{len(SUBSET)}")
    print(f"faithfulness     (avg): {sum(faiths)/n:.3f}")
    print(f"answer_relevancy (avg): {sum(rels)/n:.3f}")


if __name__ == "__main__":
    main()
