"""Phase 4 — selective agentic retrieval (LangGraph self-RAG).

Research finding (arXiv 2509.04820, 2501.09136): an adaptive self-RAG loop
(grade docs -> rewrite -> re-retrieve; grade generation -> regenerate/rewrite)
helps COMPLEX questions but DEGRADES simple ones via "query drift", at ~16-22x
the latency. So we apply it SELECTIVELY — a cheap router sends simple lookups
straight to the baseline pipeline; only complex/multi-part questions enter the loop.

The three graders are tiny yes/no LLM calls measuring the SAME dimensions RAGAS
scores offline:
  - doc relevance     (context precision)  -> drop irrelevant chunks (batched: 1 call)
  - groundedness      (faithfulness)        -> regenerate if hallucinating
  - answer usefulness (answer relevancy)    -> rewrite + retry if it misses
"""
import re
from typing import List, TypedDict

from .generate import SYSTEM, answer as baseline_answer, build_context, complete
from .retrieve import Hit, Retriever

MAX_REWRITES = 2
MAX_REGENS = 1


def _yesno(system: str, user: str) -> bool:
    return complete(system, user, max_tokens=5).strip().lower().startswith("y")


def is_complex(question: str) -> bool:
    """Router: does this query need the agentic loop? Cheap heuristics first, then
    one tiny LLM classification call."""
    q = question.lower()
    if len(question.split()) > 18 or " and " in q or " vs " in q or "compare" in q:
        return True
    sys = ("Classify a question about a codebase as SIMPLE (one direct lookup) or "
           "COMPLEX (multi-part, comparative, or spanning multiple files). "
           "Reply with one word: SIMPLE or COMPLEX.")
    return complete(sys, question, max_tokens=5).strip().upper().startswith("C")


class State(TypedDict):
    question: str        # current (possibly rewritten) query used for retrieval
    original: str        # the user's original question (what we answer/grade against)
    documents: List[Hit]
    generation: str
    rewrites: int
    regens: int


def build_app(retriever: Retriever):
    from langgraph.graph import END, StateGraph

    def retrieve(state: State):
        return {"documents": retriever.search(state["question"], top_k=8)}

    def grade_documents(state: State):
        # Batch-grade in ONE call (vs one per doc) to respect free-tier limits.
        docs = state["documents"]
        listing = "\n\n".join(f"[{i}] {h.file_path}: {h.text[:300]}" for i, h in enumerate(docs))
        sys = ("Given a QUESTION and numbered code SNIPPETS, reply with the comma-separated "
               "indices of the snippets relevant to answering it (e.g. '0,2,3'). If none, reply 'none'.")
        out = complete(sys, f"QUESTION: {state['original']}\n\nSNIPPETS:\n{listing}", max_tokens=40)
        idxs = [int(x) for x in re.findall(r"\d+", out)]
        kept = [docs[i] for i in idxs if i < len(docs)]
        return {"documents": kept or docs[:3]}  # never go fully empty

    def transform_query(state: State):
        sys = ("Rewrite this question to retrieve better code matches. Keep the user's intent; "
               "add likely symbol/keyword terms. Reply with ONLY the rewritten question.")
        rq = complete(sys, state["original"], max_tokens=60).strip()
        return {"question": rq or state["question"], "rewrites": state["rewrites"] + 1}

    def generate(state: State):
        ctx = build_context(state["documents"])
        user = f"# Retrieved code context\n\n{ctx}\n\n# Question\n{state['original']}"
        return {"generation": complete(SYSTEM, user), "regens": state["regens"] + 1}

    def decide_after_grade(state: State) -> str:
        if not state["documents"] and state["rewrites"] < MAX_REWRITES:
            return "transform_query"
        return "generate"

    def grade_generation(state: State) -> str:
        ctx = build_context(state["documents"])
        grounded = _yesno("Is the ANSWER fully supported by the CONTEXT? Reply yes or no.",
                          f"CONTEXT:\n{ctx}\n\nANSWER:\n{state['generation']}")
        if not grounded and state["regens"] <= MAX_REGENS:
            return "generate"          # hallucinating -> try again
        useful = _yesno("Does the ANSWER actually address the QUESTION? Reply yes or no.",
                        f"QUESTION: {state['original']}\n\nANSWER:\n{state['generation']}")
        if not useful and state["rewrites"] < MAX_REWRITES:
            return "transform_query"   # misses the point -> rewrite + re-retrieve
        return "end"

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("transform_query", transform_query)
    g.add_node("generate", generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges("grade_documents", decide_after_grade,
                            {"transform_query": "transform_query", "generate": "generate"})
    g.add_edge("transform_query", "retrieve")
    g.add_conditional_edges("generate", grade_generation,
                            {"generate": "generate", "transform_query": "transform_query", "end": END})
    return g.compile()


def agentic_answer(question: str, retriever: Retriever):
    """Returns (answer, path) where path is 'baseline' or 'agentic'."""
    if not is_complex(question):
        hits = retriever.search(question, top_k=8)
        return baseline_answer(question, hits), "baseline"
    app = build_app(retriever)
    out = app.invoke({
        "question": question, "original": question,
        "documents": [], "generation": "", "rewrites": 0, "regens": 0,
    })
    return out["generation"], "agentic"
