"""
topic_buckets.py

Step 15 of the project:
Create topic buckets for the main themes in the proposal.

What this file does:
1. Defines the project topic buckets and keyword rules
2. Provides helper functions that Step 16 can import later
3. Can be run directly to print and save the bucket definitions

Run from src/:
    python .\topic_buckets.py

Optional:
    python .\topic_buckets.py --output .\data\reddit\analysis\topic_buckets_definition.json
"""

import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT = Path("data/reddit/analysis/topic_buckets_definition.json")

# ---------------------------------------------------------------------
# STEP 15: TOPIC BUCKET DEFINITIONS
# These are based on the project proposal, with a few practical variants
# added so Step 16 can match real Reddit wording more reliably.
# ---------------------------------------------------------------------
TOPIC_BUCKETS = {
    "layoffs_market": {
        "description": "Layoffs, market contraction, hiring freezes, severance, and related market stress.",
        "keywords": [
            "layoff",
            "layoffs",
            "laid off",
            "laid-off",
            "rif",
            "reduction in force",
            "severance",
            "hiring freeze",
            "hiring freezes",
            "freeze hiring",
            "job market",
            "market contraction",
            "headcount reduction",
            "downsizing",
            "recession",
        ],
    },
    "recruiting_pipeline": {
        "description": "Recruiting process, interviews, offers, rejections, applications, and internship/new-grad pipeline.",
        "keywords": [
            "recruiter",
            "recruiters",
            "ghosted",
            "ghosting",
            "rejection",
            "rejections",
            "reject",
            "interview",
            "interviews",
            "phone screen",
            "onsite",
            "on-site",
            "offer",
            "offers",
            "oa",
            "online assessment",
            "leetcode",
            "new grad",
            "newgrad",
            "internship",
            "internships",
            "intern",
            "application",
            "applications",
            "resume",
            "referral",
            "referrals",
        ],
    },
    "compensation": {
        "description": "Salary, total compensation, equity, RSUs, negotiation, and pay discussions.",
        "keywords": [
            "tc",
            "salary",
            "salaries",
            "compensation",
            "comp",
            "negotiate",
            "negotiation",
            "equity",
            "rsu",
            "rsus",
            "stock grant",
            "base pay",
            "base salary",
            "pay",
            "hourly",
            "bonus",
            "bonuses",
        ],
    },
    "work_mode": {
        "description": "Remote work, hybrid work, return-to-office, and work-location mode.",
        "keywords": [
            "remote",
            "wfh",
            "work from home",
            "hybrid",
            "rto",
            "return to office",
            "return-to-office",
            "on site",
            "onsite",
            "on-site",
            "in office",
            "in-office",
        ],
    },
    "companies": {
        "description": "Specific companies frequently discussed in CS job-market conversations.",
        "keywords": [
            "amazon",
            "google",
            "meta",
            "microsoft",
            "apple",
            "qualcomm",
            "netflix",
            "tesla",
            "nvidia",
            "openai",
        ],
    },
    "ai_llm": {
        "description": "AI, LLMs, ChatGPT, Copilot, and generative AI topics.",
        "keywords": [
            "chatgpt",
            "gpt",
            "llm",
            "llms",
            "ai",
            "artificial intelligence",
            "copilot",
            "generative ai",
            "gen ai",
            "machine learning",
            "ml",
            "prompt engineering",
            "automation",
        ],
    },
}


def get_topic_buckets():
    """
    Return the topic bucket definitions.
    Step 16 can import and use this directly.
    """
    return TOPIC_BUCKETS


def print_topic_buckets():
    """
    Print all topic buckets in a clear format so we can verify Step 15.
    """
    print("=" * 70)
    print("STEP 15: TOPIC BUCKET DEFINITIONS")
    print("=" * 70)
    print(f"Total topic buckets: {len(TOPIC_BUCKETS)}")
    print()

    for bucket_name, info in TOPIC_BUCKETS.items():
        print(f"Bucket: {bucket_name}")
        print(f"Description: {info['description']}")
        print(f"Keyword count: {len(info['keywords'])}")
        print("Keywords:")
        for kw in info["keywords"]:
            print(f"  - {kw}")
        print("-" * 70)


def save_topic_buckets(output_path: Path):
    """
    Save the topic bucket definitions to a JSON file for reference.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(TOPIC_BUCKETS, f, indent=2, ensure_ascii=False)

    print()
    print(f"Topic bucket definitions saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 15: define topic buckets for the Reddit job-market sentiment project."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Optional JSON output path for saving the topic bucket definitions.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    print_topic_buckets()
    save_topic_buckets(output_path)

    print()
    print("Step 15 output is ready.")
    print("This file will be imported in Step 16 to label posts into topic buckets.")
    print("=" * 70)


if __name__ == "__main__":
    main()