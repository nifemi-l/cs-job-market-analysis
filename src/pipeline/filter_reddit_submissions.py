"""
filter_reddit_submissions.py

Step 7 of the project:
Filter Reddit data to only the subreddits relevant to the project.

Target subreddits:
- cscareerquestions
- csMajors
- recruitinghell
- jobs

What this script does:
1. Reads a Reddit dump file (.zst, .jsonl, or .ndjson)
2. Parses one JSON object per line
3. Keeps only records from the target subreddits
4. Writes a smaller filtered CSV file for later steps

Run:
    python src/pipeline/filter_reddit_submissions.py --input src/data/reddit/RS_2023-02.zst
"""

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import zstandard as zstd

SRC_DIR = Path(__file__).resolve().parent.parent

TARGET_SUBREDDITS = {
    "cscareerquestions",
    "csmajors",
    "recruitinghell",
    "jobs",
}

DEFAULT_INPUT_CANDIDATES = [
    SRC_DIR / "data" / "reddit" / "RS_2023-02.zst",
    SRC_DIR / "data" / "reddit" / "RC_2023-02.zst",
]

DEFAULT_OUTPUT_DIR = SRC_DIR / "data" / "reddit" / "filtered"
PROGRESS_EVERY = 100_000


def normalize_text(value):
    """Convert None to empty string and strip surrounding whitespace."""
    if value is None:
        return ""
    return str(value).strip()


def detect_source_type(record):
    """
    Detect whether a record is a submission or comment.
    We use the presence of typical fields to infer the type.
    """
    if "title" in record or "selftext" in record or "num_comments" in record:
        return "submission"
    if "body" in record:
        return "comment"
    return "unknown"


def standardize_record(record):
    """
    Convert either a submission or comment into one consistent output schema.
    This keeps later steps easier.
    """
    source_type = detect_source_type(record)

    if source_type == "submission":
        title = normalize_text(record.get("title", ""))
        body = normalize_text(record.get("selftext", ""))
        num_comments = record.get("num_comments", "")
        url = normalize_text(record.get("url", ""))
    else:
        title = ""
        body = normalize_text(record.get("body", ""))
        num_comments = ""
        url = ""

    return {
        "source_type": source_type,
        "id": normalize_text(record.get("id", "")),
        "subreddit": normalize_text(record.get("subreddit", "")).lower(),
        "author": normalize_text(record.get("author", "")),
        "created_utc": record.get("created_utc", ""),
        "title": title,
        "body": body,
        "score": record.get("score", ""),
        "num_comments": num_comments,
        "permalink": normalize_text(record.get("permalink", "")),
        "url": url,
    }


def open_reddit_file(path):
    """
    Open a Reddit dump file for line-by-line text reading.
    Supports .zst, .jsonl, .ndjson, and plain text JSON lines files.
    """
    suffix = path.suffix.lower()

    if suffix == ".zst":
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        return fh, text_stream

    fh = open(path, "r", encoding="utf-8")
    return fh, fh


def find_default_input():
    """Pick the first existing default file candidate."""
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    reddit_dir = SRC_DIR / "data" / "reddit"
    if reddit_dir.exists():
        for pattern in ("*.zst", "*.jsonl", "*.ndjson", "*.json"):
            matches = sorted(reddit_dir.glob(pattern))
            if matches:
                return matches[0]

    return None


def filter_reddit_file(input_path, output_path):
    print("=" * 70)
    print("STEP 7: FILTER REDDIT DATA TO PROJECT SUBREDDITS")
    print("=" * 70)
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print(f"Target subreddits: {sorted(TARGET_SUBREDDITS)}")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    parsed_records = 0
    kept_records = 0
    json_errors = 0
    missing_subreddit = 0
    kept_by_subreddit = Counter()
    source_type_counts = Counter()

    fieldnames = [
        "source_type",
        "id",
        "subreddit",
        "author",
        "created_utc",
        "title",
        "body",
        "score",
        "num_comments",
        "permalink",
        "url",
    ]

    raw_fh = None
    text_stream = None

    try:
        print("Opening input file...")
        raw_fh, text_stream = open_reddit_file(input_path)
        print("Input opened successfully.")
        print()

        print("Starting line-by-line filtering...")
        with open(output_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            for line in text_stream:
                total_lines += 1

                if total_lines % PROGRESS_EVERY == 0:
                    print(
                        f"[Progress] lines read: {total_lines:,} | "
                        f"records kept: {kept_records:,} | "
                        f"json errors: {json_errors:,}"
                    )

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    parsed_records += 1
                except json.JSONDecodeError:
                    json_errors += 1
                    continue

                subreddit = normalize_text(record.get("subreddit", "")).lower()
                if not subreddit:
                    missing_subreddit += 1
                    continue

                if subreddit not in TARGET_SUBREDDITS:
                    continue

                standardized = standardize_record(record)
                writer.writerow(standardized)

                kept_records += 1
                kept_by_subreddit[subreddit] += 1
                source_type_counts[standardized["source_type"]] += 1

        print()
        print("Filtering finished successfully.")
        print("-" * 70)
        print(f"Total lines read:         {total_lines:,}")
        print(f"JSON records parsed:      {parsed_records:,}")
        print(f"JSON parse errors:        {json_errors:,}")
        print(f"Missing subreddit field:  {missing_subreddit:,}")
        print(f"Records kept:             {kept_records:,}")
        print()

        print("Kept records by subreddit:")
        if kept_by_subreddit:
            for sub, count in kept_by_subreddit.most_common():
                print(f"  {sub}: {count:,}")
        else:
            print("  No matching records were found.")

        print()
        print("Kept records by source type:")
        if source_type_counts:
            for source_type, count in source_type_counts.most_common():
                print(f"  {source_type}: {count:,}")
        else:
            print("  No records were written.")

        print()
        print(f"Saved filtered file to: {output_path}")
        print("=" * 70)

    finally:
        if text_stream is not None:
            try:
                text_stream.close()
            except Exception:
                pass

        if raw_fh is not None:
            try:
                raw_fh.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Filter Reddit dump to only project-relevant subreddits."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to Reddit input file. If omitted, the script tries common defaults.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file. If omitted, a default path is generated.",
    )
    args = parser.parse_args()

    if args.input is not None:
        input_path = Path(args.input)
    else:
        input_path = find_default_input()

    if input_path is None:
        print("ERROR: No Reddit input file was found.")
        print("Put your file inside src/data/reddit/ and rerun.")
        print("Example:")
        print(r"  python src/pipeline/filter_reddit_submissions.py --input src/data/reddit/RS_2023-02.zst")
        sys.exit(1)

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}")
        sys.exit(1)

    if args.output is not None:
        output_path = Path(args.output)
    else:
        safe_name = input_path.stem.replace(".", "_")
        output_path = DEFAULT_OUTPUT_DIR / f"{safe_name}_filtered.csv"

    filter_reddit_file(input_path, output_path)


if __name__ == "__main__":
    main()
