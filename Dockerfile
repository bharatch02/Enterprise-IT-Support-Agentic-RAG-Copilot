FROM python:3.11-slim

# Render's Docker build supports BuildKit secret mounts. The token, if provided,
# is available only during this RUN and is never passed through an ARG or image ENV.

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ARG EMBEDDING_MODEL_PATH=/opt/models/fastembed
ENV HF_HUB_ETAG_TIMEOUT=10
ENV HF_HUB_DOWNLOAD_TIMEOUT=30

COPY scripts/download_embedding_model.py ./scripts/download_embedding_model.py
RUN --mount=type=secret,id=hf_token,required=false \
    if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi && \
    python scripts/download_embedding_model.py --destination "$EMBEDDING_MODEL_PATH"

ENV EMBEDDING_MODEL_PATH=${EMBEDDING_MODEL_PATH}
ENV EMBEDDING_LOCAL_FILES_ONLY=true

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
