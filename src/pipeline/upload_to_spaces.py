"""
upload_to_spaces.py

Upload pipeline outputs to DigitalOcean Spaces.

Using a .env file with:
    DO_SPACES_KEY=your-access-key
    DO_SPACES_SECRET=your-secret-key
    DO_SPACES_REGION=sfo3
    DO_SPACES_BUCKET=eecs767-reddit

The script checks these paths in order:
    <repo>/.env
    <repo>/src/pipeline/.env

Commands:
    python src/pipeline/upload_to_spaces.py
    python src/pipeline/upload_to_spaces.py --files src/data/reddit/filtered/all_subreddits_filtered.csv
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config

SRC_DIR = Path(__file__).resolve().parent.parent

DEFAULT_UPLOAD_DIRS = [
    SRC_DIR / "data" / "reddit" / "filtered",
    SRC_DIR / "data" / "reddit" / "processed",
    SRC_DIR / "data" / "reddit" / "predictions",
    SRC_DIR / "data" / "reddit" / "analysis",
    SRC_DIR / "data" / "reddit" / "final_outputs",
]


def infer_step_label(relative_path):
    """
    Map output paths to pipeline step labels for upload organization.
    """
    rel = Path(relative_path)
    rel_str = str(rel)
    parts = rel.parts

    if parts and parts[0] == "filtered":
        return "step07_filter_submissions"
    if parts and parts[0] == "processed":
        if "selected_fields" in rel_str:
            return "step08_select_fields"
        if "final_text" in rel_str:
            return "step09_build_final_text"
        if "cleaned_text" in rel_str:
            return "step10_preprocess_text"
        return "step08_10_processed"
    if parts and parts[0] == "predictions":
        return "step11_12_predict_sentiment"
    if parts and parts[0] == "analysis":
        if "time_grouped" in rel_str:
            return "step14_group_by_time"
        if "topic_tagged" in rel_str:
            return "step16_tag_topics"
        if "topic_buckets_definition" in rel_str:
            return "step15_topic_buckets"
        if len(parts) > 1 and parts[1] == "step17_topic_sentiment":
            return "step17_sentiment_by_topic"
        if len(parts) > 1 and parts[1] == "step18_topic_trends":
            return "step18_topic_trends"
        if len(parts) > 1 and parts[1] == "step19_hypothesis_tests":
            return "step19_hypothesis_tests"
        return "step14_19_analysis"
    if parts and parts[0] == "final_outputs":
        return "step20_final_outputs"
    return "other_outputs"


def load_env_file():
    """
    Load key=value pairs from the first available .env file.
    """
    repo_root = SRC_DIR.parent
    candidates = [
        repo_root / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def get_s3_client():
    key = os.environ.get("DO_SPACES_KEY")
    secret = os.environ.get("DO_SPACES_SECRET")
    region = os.environ.get("DO_SPACES_REGION", "nyc3")

    if not key or not secret:
        print("ERROR: Missing DO_SPACES_KEY or DO_SPACES_SECRET.")
        sys.exit(1)
    key_tail = key[-4:] if len(key) >= 4 else key
    print(f"Using Spaces key ending in ...{key_tail}")

    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )


def upload_file(client, bucket, local_path, remote_key):
    print(f"  Uploading {local_path.name} -> {remote_key}")
    client.upload_file(str(local_path), bucket, remote_key)


def main():
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Upload project outputs to DigitalOcean Spaces."
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Specific files to upload. If omitted, uploads all CSV/JSON/TXT/PNG "
             "files from the default output directories.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("DO_SPACES_BUCKET", "eecs767-reddit"),
        help="Spaces bucket name.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="pipeline-outputs",
        help="Key prefix inside the bucket.",
    )
    args = parser.parse_args()

    client = get_s3_client()
    bucket = args.bucket
    prefix = args.prefix

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        extensions = {".csv", ".json", ".txt", ".md", ".png"}
        paths = []
        for d in DEFAULT_UPLOAD_DIRS:
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.is_file() and f.suffix.lower() in extensions:
                        paths.append(f)

    if not paths:
        print("Nothing to upload.")
        return

    print(f"Uploading {len(paths)} file(s) to s3://{bucket}/{prefix}/")
    print()

    data_root = SRC_DIR / "data" / "reddit"
    for p in paths:
        try:
            relative = p.relative_to(data_root)
        except ValueError:
            relative = Path(p.name)
        step_label = infer_step_label(relative)
        remote_key = f"{prefix}/{step_label}/{relative}"
        upload_file(client, bucket, p, remote_key)

    print()
    print(f"Done. {len(paths)} file(s) uploaded.")


if __name__ == "__main__":
    main()
