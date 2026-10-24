
import logging
import time
from threading import Lock

from app.core.config import get_settings
from app.rag.embedding_loader import (
    FastEmbedAdapter,
    LazyInitializer,
    normalize_hf_timeout,
)

settings = get_settings()
logger = logging.getLogger(__name__)

_embedding_initializer = LazyInitializer()
_vectorstore = None
_vectorstore_lock = Lock()

EMBEDDING_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")

    normalized = name.lower()

    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "text-embedding-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model '{model_name or settings.embedding_model}' "
        "for Pinecone. Add the matching dimension to EMBEDDING_DIMENSIONS."
    )


def get_embeddings(request_id: str = "unknown"):
    def initialize():
        started = time.perf_counter()

        logger.info(
            "integration_started request_id=%s integration=huggingface_embeddings",
            request_id,
        )

        try:
            # Canonicalize inherited Render values too; setdefault would leave
            # malformed strings such as "10.0" untouched.
            normalize_hf_timeout(
                "HF_HUB_ETAG_TIMEOUT", settings.hf_hub_etag_timeout_seconds
            )
            normalize_hf_timeout(
                "HF_HUB_DOWNLOAD_TIMEOUT", settings.hf_hub_download_timeout_seconds
            )

            embeddings = FastEmbedAdapter(
                model_name=EMBEDDING_REPOSITORY,
                cache_dir=settings.embedding_model_path,
                local_files_only=settings.embedding_local_files_only,
            )

        except Exception as exc:
            logger.exception(
                "integration_failed request_id=%s integration=huggingface_embeddings "
                "error_type=%s duration_ms=%d",
                request_id,
                type(exc).__name__,
                round((time.perf_counter() - started) * 1000),
            )
            raise

        logger.info(
            "integration_completed request_id=%s integration=huggingface_embeddings duration_ms=%d",
            request_id,
            round((time.perf_counter() - started) * 1000),
        )

        return embeddings

    return _embedding_initializer.get(initialize)


def ensure_index(request_id: str = "unknown"):
    if not settings.pinecone_api_key:
        logger.error(
            "integration_failed request_id=%s integration=pinecone reason=missing_api_key",
            request_id,
        )
        raise RuntimeError("PINECONE_API_KEY is missing")

    started = time.perf_counter()

    logger.info(
        "integration_started request_id=%s integration=pinecone_index",
        request_id,
    )

    try:
        from pinecone import Pinecone, ServerlessSpec

        desired_dimension = get_embedding_dimension()

        pc = Pinecone(
            api_key=settings.pinecone_api_key,
            timeout=settings.pinecone_control_timeout_seconds,
        )

        names = [item["name"] for item in pc.list_indexes()]

        if settings.pinecone_index_name in names:
            index_info = pc.describe_index(settings.pinecone_index_name)
            current_dimension = getattr(index_info, "dimension", None)

            if current_dimension is None and isinstance(index_info, dict):
                current_dimension = index_info.get("dimension")

            if (
                current_dimension is not None
                and current_dimension != desired_dimension
            ):
                raise RuntimeError(
                    f"Pinecone index has dimension {current_dimension}, "
                    f"but the selected embedding model requires "
                    f"{desired_dimension}. Configure a compatible Pinecone index."
                )

        if settings.pinecone_index_name not in names:
            logger.info(
                "pinecone_index_creation_started request_id=%s",
                request_id,
            )

            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=desired_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

            deadline = (
                time.monotonic()
                + settings.pinecone_index_ready_timeout_seconds
            )

            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Pinecone index did not become ready before timeout"
                    )

                index_info = pc.describe_index(settings.pinecone_index_name)
                status = getattr(index_info, "status", None)

                if status is not None and getattr(status, "ready", False):
                    break

                time.sleep(1)

        index = pc.Index(settings.pinecone_index_name)

    except Exception as exc:
        logger.exception(
            "integration_failed request_id=%s integration=pinecone_index "
            "error_type=%s duration_ms=%d",
            request_id,
            type(exc).__name__,
            round((time.perf_counter() - started) * 1000),
        )
        raise

    logger.info(
        "integration_completed request_id=%s integration=pinecone_index duration_ms=%d",
        request_id,
        round((time.perf_counter() - started) * 1000),
    )

    return index


def get_vectorstore(request_id: str = "unknown"):
    global _vectorstore

    if _vectorstore is None:
        with _vectorstore_lock:
            if _vectorstore is None:
                started = time.perf_counter()

                logger.info(
                    "integration_started request_id=%s integration=pinecone_vectorstore",
                    request_id,
                )

                try:
                    from langchain_pinecone import PineconeVectorStore

                    index = ensure_index(request_id=request_id)
                    embeddings = get_embeddings(request_id=request_id)

                    _vectorstore = PineconeVectorStore(
                        index=index,
                        embedding=embeddings,
                        namespace=settings.pinecone_namespace,
                    )

                except Exception as exc:
                    logger.exception(
                        "integration_failed request_id=%s integration=pinecone_vectorstore "
                        "error_type=%s duration_ms=%d",
                        request_id,
                        type(exc).__name__,
                        round((time.perf_counter() - started) * 1000),
                    )
                    raise

                logger.info(
                    "integration_completed request_id=%s integration=pinecone_vectorstore duration_ms=%d",
                    request_id,
                    round((time.perf_counter() - started) * 1000),
                )

    return _vectorstore


def get_retriever(request_id: str = "unknown"):
    return get_vectorstore(request_id=request_id).as_retriever(
        search_kwargs={
            "k": 2,
            "timeout": settings.pinecone_query_timeout_seconds,
        }
    )


def add_documents(chunks):
    store = get_vectorstore()

    ids = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        start = chunk.metadata.get("start_index", i)
        ids.append(f"{source}:{start}")

    return store.add_documents(chunks, ids=ids)
