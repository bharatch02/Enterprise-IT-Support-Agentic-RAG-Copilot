"""Download the project's fixed 384-dimensional embedding model."""

import argparse
import os
from pathlib import Path

REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
FASTEMBED_HUB_REPOSITORY = "qdrant/all-MiniLM-L6-v2-onnx"


def normalize_hf_timeout(name: str, fallback_seconds: int) -> None:
    """Set Hub timeout variables to integer strings before importing the Hub."""
    raw_value = os.getenv(name)
    try:
        seconds = int(float(raw_value)) if raw_value is not None else fallback_seconds
        if seconds < 1:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        seconds = fallback_seconds
    os.environ[name] = str(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        default=str(Path.home() / ".cache" / "enterprise-it-support-fastembed"),
        help="FastEmbed cache directory (defaults to the user's local cache).",
    )
    args = parser.parse_args()

    normalize_hf_timeout("HF_HUB_ETAG_TIMEOUT", 10)
    normalize_hf_timeout("HF_HUB_DOWNLOAD_TIMEOUT", 30)
    from huggingface_hub import snapshot_download
    from fastembed import TextEmbedding

    # Fetch the ONNX model explicitly so HF_TOKEN is passed directly and the
    # runtime can remain offline. FastEmbed's registry uses the same cache dir.
    snapshot_download(
        repo_id=FASTEMBED_HUB_REPOSITORY,
        cache_dir=args.destination,
        token=os.getenv("HF_TOKEN") or None,
    )
    model = TextEmbedding(
        model_name=REPOSITORY,
        cache_dir=args.destination,
        threads=1,
        local_files_only=True,
    )
    if model.embedding_size != 384:
        raise RuntimeError("The downloaded embedding model is not 384-dimensional.")
    print(f"384-dimensional embedding model is available in: {args.destination}")


if __name__ == "__main__":
    main()
