import logging
import time
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_embeddings = None
_vectorstore = None

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
        f"Unsupported embedding model '{model_name or settings.embedding_model}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        started = time.perf_counter()
        logger.info("integration_started integration=huggingface_embeddings")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            _embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            logger.error(
                "integration_failed integration=huggingface_embeddings error_type=%s duration_ms=%d",
                type(exc).__name__,
                round((time.perf_counter() - started) * 1000),
            )
            raise

        logger.info(
            "integration_completed integration=huggingface_embeddings duration_ms=%d",
            round((time.perf_counter() - started) * 1000),
        )

    return _embeddings

def ensure_index():
    if not settings.pinecone_api_key:
        logger.error("integration_failed integration=pinecone reason=missing_api_key")
        raise RuntimeError("PINECONE_API_KEY is missing")

    started = time.perf_counter()
    logger.info("integration_started integration=pinecone_index")
    try:
        from pinecone import Pinecone, ServerlessSpec

        desired_dimension = get_embedding_dimension()
        pc = Pinecone(api_key=settings.pinecone_api_key)
        names = [x["name"] for x in pc.list_indexes()]

        if settings.pinecone_index_name in names:
            index_info = pc.describe_index(settings.pinecone_index_name)
            current_dimension = getattr(index_info, "dimension", None)
            if current_dimension is None and isinstance(index_info, dict):
                current_dimension = index_info.get("dimension")

            if current_dimension is not None and current_dimension != desired_dimension:
                raise RuntimeError(
                    f"Pinecone index has dimension {current_dimension}, "
                    f"but the selected embedding model requires {desired_dimension}. "
                    "Configure a new Pinecone index for the local model."
                )

        if settings.pinecone_index_name not in [x["name"] for x in pc.list_indexes()]:
            logger.info("pinecone_index_creation_started")
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=desired_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(settings.pinecone_index_name).status.ready:
                time.sleep(1)

        index = pc.Index(settings.pinecone_index_name)
    except Exception as exc:
        logger.error(
            "integration_failed integration=pinecone_index error_type=%s duration_ms=%d",
            type(exc).__name__,
            round((time.perf_counter() - started) * 1000),
        )
        raise

    logger.info(
        "integration_completed integration=pinecone_index duration_ms=%d",
        round((time.perf_counter() - started) * 1000),
    )
    return index


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        started = time.perf_counter()
        logger.info("integration_started integration=pinecone_vectorstore")
        try:
            from langchain_pinecone import PineconeVectorStore

            index = ensure_index()
            _vectorstore = PineconeVectorStore(
                index=index,
                embedding=get_embeddings(),
                namespace=settings.pinecone_namespace,
            )
        except Exception as exc:
            logger.error(
                "integration_failed integration=pinecone_vectorstore error_type=%s duration_ms=%d",
                type(exc).__name__,
                round((time.perf_counter() - started) * 1000),
            )
            raise

        logger.info(
            "integration_completed integration=pinecone_vectorstore duration_ms=%d",
            round((time.perf_counter() - started) * 1000),
        )
    return _vectorstore

def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": 2}
    )


def add_documents(chunks):
    store = get_vectorstore()

    ids = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        start = chunk.metadata.get("start_index", i)
        ids.append(f"{source}:{start}")

    return store.add_documents(chunks, ids=ids)
