"""Small, dependency-light helpers for safe lazy embedding initialization."""

from pathlib import Path
from threading import Lock
from typing import Callable, Type


def resolve_embedding_model_path(
    *,
    repository: str,
    configured_path: str,
    local_files_only: bool,
    token: str | None,
    etag_timeout: float,
    snapshot_download: Callable,
    cache_miss_error: Type[Exception],
) -> str:
    """Prefer a configured local model, then the Hub cache, then optional download."""
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_dir():
            raise RuntimeError(
                "EMBEDDING_MODEL_PATH is configured but does not point to a local model directory."
            )
        return str(path.resolve())

    try:
        return str(
            snapshot_download(
                repo_id=repository,
                token=token,
                local_files_only=True,
            )
        )
    except cache_miss_error as exc:
        if local_files_only:
            raise RuntimeError(
                "The embedding model is not present locally. Download it before starting "
                "the service, or configure EMBEDDING_MODEL_PATH."
            ) from exc

    # Network access is only used when explicitly allowed and the local cache misses.
    return str(
        snapshot_download(
            repo_id=repository,
            token=token,
            local_files_only=False,
            etag_timeout=etag_timeout,
        )
    )


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
