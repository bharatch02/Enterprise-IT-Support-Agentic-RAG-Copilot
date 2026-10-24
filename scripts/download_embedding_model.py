"""Download the project's fixed 384-dimensional embedding model."""

import argparse
import os


REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        help="Optional local directory for the model; defaults to the Hugging Face cache.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=REPOSITORY,
        token=os.getenv("HF_TOKEN") or None,
        local_dir=args.destination,
        max_workers=4,
    )
    print(f"Embedding model is available locally at: {path}")


if __name__ == "__main__":
    main()
