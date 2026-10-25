"""Low-memory, lazy embeddings backed by FastEmbed and ONNX Runtime."""

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Callable


logger = logging.getLogger(__name__)
EXPECTED_EMBEDDING_DIMENSION = 384


def normalize_hf_timeout(name: str, fallback_seconds: float) -> str:
    """Normalize Hub timeout variables to integer strings before Hub imports."""
    raw_value = os.getenv(name)
    try:
        seconds = int(float(raw_value)) if raw_value is not None else int(fallback_seconds)
        if seconds < 1:
            raise ValueError("timeout must be positive")
    except (TypeError, ValueError, OverflowError):
        seconds = max(1, int(fallback_seconds))
        if raw_value is not None:
            logger.warning(
                "invalid_environment_value setting=%s using_configured_default=true",
                name,
            )

    normalized = str(seconds)
    os.environ[name] = normalized
    return normalized


class FastEmbedAdapter:
    """LangChain-compatible adapter with small batches and one ONNX thread."""

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str,
        local_files_only: bool,
        model_factory: Callable | None = None,
    ):
        if local_files_only:
            # FastEmbed also receives local_files_only below. Set the Hub-wide
            # guard before importing it so no Hub metadata requests slip
            # through during a production cache-only load.
            os.environ["HF_HUB_OFFLINE"] = "1"

        if model_factory is None:
            from fastembed import TextEmbedding

            model_factory = TextEmbedding

        cache_path = Path(cache_dir).expanduser()
        if local_files_only and not cache_path.is_dir():
            raise RuntimeError(
                "The FastEmbed model cache is not present locally. Run "
                "scripts/download_embedding_model.py or set EMBEDDING_MODEL_PATH."
            )

        self._model = model_factory(
            model_name=model_name,
            cache_dir=str(cache_path),
            threads=1,
            local_files_only=local_files_only,
        )

        model_dimension = getattr(self._model, "embedding_size", None)
        if model_dimension != EXPECTED_EMBEDDING_DIMENSION:
            raise RuntimeError(
                "The configured FastEmbed model returned an unexpected dimension; "
                f"expected {EXPECTED_EMBEDDING_DIMENSION}."
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.embed(texts, batch_size=8, parallel=None)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = next(self._model.query_embed(text, batch_size=1, parallel=None))
        return vector.tolist()


class LazyInitializer:
    """Cache one successful initialization; serialize concurrent first callers."""

    def __init__(self):
        self._value = None
        self._lock = Lock()

    def get(self, factory: Callable):
        if self._value is None:
            with self._lock:
                if self._value is None:
                    # Keep failures uncached so a later request can retry initialization.
                    self._value = factory()
        return self._value
