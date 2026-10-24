
import logging
from pathlib import Path
import time
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Response
from pydantic import BaseModel, Field

from app.core.config import get_settings

from app.rag.workflow import ask

from app.rag.vectorstore import add_documents

from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")
settings = get_settings()
logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=3000,
        description="The user's IT support question",
    )


@router.get("/health")
def health():
    """Check whether the API service is running."""
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@router.post("/chat")
def chat(payload: ChatRequest, response: Response):
    """Process a question through the agentic RAG workflow."""
    question = payload.question.strip()
    request_id = uuid4().hex[:12]
    response.headers["X-Request-ID"] = request_id

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
            headers={"X-Request-ID": request_id},
        )

    try:
        logger.info("chat_request_started request_id=%s", request_id)
        result = ask(question, request_id=request_id)

        # Audit logging must not discard a successfully generated answer.
        audit_started = time.perf_counter()
        logger.info("workflow_stage_started request_id=%s stage=audit_write", request_id)
        try:
            write_audit(
                question,
                result.get("source_used", "unknown"),
                result.get("trace", []),
            )
        except Exception as exc:
            logger.error(
                "workflow_stage_failed request_id=%s stage=audit_write error_type=%s duration_ms=%d",
                request_id,
                type(exc).__name__,
                round((time.perf_counter() - audit_started) * 1000),
            )
        else:
            logger.info(
                "workflow_stage_completed request_id=%s stage=audit_write duration_ms=%d",
                request_id,
                round((time.perf_counter() - audit_started) * 1000),
            )

        logger.info(
            "chat_request_completed request_id=%s source=%s",
            request_id,
            result.get("source_used", "unknown"),
        )

        return {
            "answer": result.get("answer", ""),
            "source_used": result.get("source_used", "unknown"),
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", question),
        }

    except ValueError as exc:
        logger.warning(
            "chat_request_rejected request_id=%s error_type=%s",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=400,
            detail=str(exc),
            headers={"X-Request-ID": request_id},
        ) from exc

    except Exception as exc:
        logger.error(
            "chat_request_failed request_id=%s error_type=%s",
            request_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "The support assistant encountered an internal error. "
                f"Please try again. Reference ID: {request_id}"
            ),
            headers={"X-Request-ID": request_id},
        ) from exc


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default=""),
):
    """Validate, process, and index a company knowledge document."""

    # Verify the admin key before processing the uploaded file.
    if not settings.admin_api_key or x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key.",
        )

    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()

    if not filename or suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Supported formats: "
                f"{', '.join(sorted(SUPPORTED))}"
            ),
        )

    upload_dir = Path(settings.upload_dir)
    dest = upload_dir / filename

    try:
        # Read at most 10 MB plus one byte to detect oversized uploads.
        contents = await file.read(MAX_UPLOAD_SIZE + 1)

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 10 MB upload limit.",
            )

        upload_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(contents)

        documents = load_file(dest)
        chunks = chunk_documents(documents)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable content was found in the uploaded document.",
            )

        ids = add_documents(chunks)

        logger.info(
            "Document indexed successfully: filename=%s chunks=%d",
            filename,
            len(ids),
        )

        return {
            "message": "Document indexed successfully.",
            "file": filename,
            "chunks": len(chunks),
            "ids_created": len(ids),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Document ingestion failed: filename=%s",
            filename,
        )
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check the server logs.",
        ) from exc

    finally:
        await file.close()
