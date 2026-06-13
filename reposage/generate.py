"""Grounded, cited answer generation.

The prompt is engineered to fight hallucination: the model must answer ONLY from
the retrieved code, cite sources inline as [path:start-end], and admit when the
context doesn't contain the answer. That "say you don't know" instruction is the
single most important line for trustworthy RAG.
"""
from typing import List

from .config import config
from .retrieve import Hit

SYSTEM = """You are RepoSage, an assistant that answers questions about a codebase.

Rules:
- Answer ONLY using the provided code context below. Do not use outside knowledge
  about how similar projects usually work.
- If the context does not contain the answer, say so plainly. Never invent files,
  functions, or behavior.
- Cite the sources you used inline, as [file_path:start-end].
- Be concise and precise. Quote short code snippets when they help.
"""


def build_context(hits: List[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        head = f"[{h.file_path}:{h.start_line}-{h.end_line}]"
        if h.symbol:
            head += f"  symbol: {h.symbol}"
        blocks.append(f"### Source {i} {head}\n```{h.language}\n{h.text}\n```")
    return "\n\n".join(blocks)


def answer(query: str, hits: List[Hit]) -> str:
    context = build_context(hits)
    user = f"# Retrieved code context\n\n{context}\n\n# Question\n{query}"

    if config.llm_provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        msg = client.messages.create(
            model=config.anthropic_model,
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    raise ValueError(f"Unknown llm_provider: {config.llm_provider!r}")
