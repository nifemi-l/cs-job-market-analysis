"""
download_from_spaces.py

Download pipeline outputs from DigitalOcean Spaces.

Using a .env file with:
    DO_SPACES_KEY=your-access-key
    DO_SPACES_SECRET=your-secret-key
    DO_SPACES_REGION=sfo3
    DO_SPACES_BUCKET=eecs767-reddit

The script checks these paths in order:
    <repo>/.env
    <repo>/src/pipeline/.env

Commands:
    python src/pipeline/download_from_spaces.py
    python src/pipeline/download_from_spaces.py --prefix pipeline-outputs --output-dir src/data/reddit
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config

SRC_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = SRC_DIR / "data" / "reddit"


def load_env_file():
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

    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
    )


def iter_object_keys(client, bucket, prefix):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key and not key.endswith("/"):
                yield key


def main():
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Download pipeline outputs from DigitalOcean Spaces."
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
        help="Prefix to download from bucket.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local files if they already exist.",
    )
    args = parser.parse_args()

    client = get_s3_client()
    bucket = args.bucket
    prefix = args.prefix.rstrip("/")
    output_dir = Path(args.output_dir)

    keys = list(iter_object_keys(client, bucket, prefix))
    if not keys:
        print(f"No files found at s3://{bucket}/{prefix}/")
        return

    print(f"Downloading {len(keys)} file(s) from s3://{bucket}/{prefix}/")
    downloaded = 0
    skipped = 0

    for key in keys:
        rel = key[len(prefix) :].lstrip("/")
        local_path = output_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and not args.overwrite:
            skipped += 1
            continue

        print(f"  {key} -> {local_path}")
        client.download_file(bucket, key, str(local_path))
        downloaded += 1

    print()
    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
