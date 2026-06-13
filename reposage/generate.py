"""Grounded, cited answer generation.

The prompt is engineered to fight hallucination: the model must answer ONLY from
the retrieved code, cite sources inline as [path:start-end], and admit when the
context doesn't contain the answer. That "say you don't know" instruction is the
single most important line for trustworthy RAG.

`complete()` is the provider-agnostic primitive (Gemini default, Anthropic
optional), reused by answer generation AND by the eval's LLM judge.
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


_TRANSIENT = ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "500", "INTERNAL")


def _with_retry(fn, tries: int = 4):
    """Retry transient LLM errors (free-tier 429 / 503 high-demand) with backoff.
    Essential for the agentic loop, which fires many rapid calls."""
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if any(s in str(e) for s in _TRANSIENT) and i < tries - 1:
                time.sleep(2 + 3 * i)
                continue
            raise


def complete(system: str, user: str, max_tokens: int = 1024) -> str:
    """Single-shot LLM completion for the configured provider (with retry)."""
    if config.llm_provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.gemini_api_key)

        def _call():
            resp = client.models.generate_content(
                model=config.gemini_model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return resp.text or ""

        return _with_retry(_call)

    if config.llm_provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

        def _call():
            msg = client.messages.create(
                model=config.anthropic_model, max_tokens=max_tokens,
                system=system, messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if b.type == "text")

        return _with_retry(_call)

    raise ValueError(f"Unknown llm_provider: {config.llm_provider!r}")


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
    return complete(SYSTEM, user)
