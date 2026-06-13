"""Central configuration. Everything is overridable via environment variables so
the same code runs locally (free, on-disk) and against a hosted Qdrant later."""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- Embeddings (fastembed model names) ---
    # bge-small is 384-dim and runs fast on CPU — a good free default.
    # Upgrade path (per research): BAAI/bge-code-v1 (SOTA code retrieval) or an API model.
    dense_model: str = os.getenv("REPOSAGE_DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    sparse_model: str = os.getenv("REPOSAGE_SPARSE_MODEL", "Qdrant/bm25")

    # --- Vector store (Qdrant) ---
    # If REPOSAGE_QDRANT_URL is set we talk to a server; otherwise we use a local
    # on-disk store (zero infra — perfect for development).
    qdrant_url: str = os.getenv("REPOSAGE_QDRANT_URL", "")
    qdrant_path: str = os.getenv("REPOSAGE_QDRANT_PATH", "./qdrant_data")
    collection: str = os.getenv("REPOSAGE_COLLECTION", "reposage")

    # --- Chunking ---
    # ~1500 chars ≈ ~400 tokens ≈ ~40 lines — the research sweet spot for typical
    # context budgets (32-64 line chunks).
    max_chunk_chars: int = int(os.getenv("REPOSAGE_MAX_CHUNK_CHARS", "1500"))

    # --- Generation (Phase 2+). Default provider is Gemini (free tier). ---
    llm_provider: str = os.getenv("REPOSAGE_LLM_PROVIDER", "gemini")
    # flash-lite: reliable on the free tier (avoids 2.5-flash's intermittent 503s),
    # fast, and proven at faithfulness/relevancy 1.00 in our eval.
    gemini_model: str = os.getenv("REPOSAGE_GEMINI_MODEL", "gemini-2.5-flash-lite")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    anthropic_model: str = os.getenv("REPOSAGE_ANTHROPIC_MODEL", "claude-haiku-4-5")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # --- Source citations (web UI) ---
    # If set, each cited source becomes a clickable link to the real code (e.g. a
    # GitHub blob URL). The UI appends "/<file_path>#Lstart-Lend".
    source_base_url: str = os.getenv("REPOSAGE_SOURCE_BASE_URL", "")

    @property
    def llm_ready(self) -> bool:
        """Is a key available for the selected provider? (CLI uses this to decide
        whether to generate an answer or stop at retrieval.)"""
        if self.llm_provider == "gemini":
            return bool(self.gemini_api_key)
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm_provider == "ollama":
            return True  # local, no key needed
        return False


config = Config()
