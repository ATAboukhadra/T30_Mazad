"""Fetch soccer/football player names from FIFA video game datasets (GitHub).

This is the easiest and most comprehensive free source for footballer names.
No API rate limits - just downloads CSVs directly from GitHub.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from typing import Set


# FIFA player dataset maintained on GitHub (soccer/football only)
FIFA_URLS = {
    "fifa24": "https://raw.githubusercontent.com/stefanoleone992/fifa-players-ratings/master/data/players_24.csv",
    "fifa23": "https://raw.githubusercontent.com/stefanoleone992/fifa-players-ratings/master/data/players_23.csv",
    "fifa22": "https://raw.githubusercontent.com/stefanoleone992/fifa-players-ratings/master/data/players_22.csv",
    "fifa21": "https://raw.githubusercontent.com/stefanoleone992/fifa-players-ratings/master/data/players_21.csv",
    "fifa20": "https://raw.githubusercontent.com/stefanoleone992/fifa-players-ratings/master/data/players_20.csv",
}


def fetch_fifa_csv(url: str, timeout: int = 120) -> Set[str]:
    """Download FIFA CSV and extract player names."""
    names: Set[str] = set()
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read().decode("utf-8", errors="replace")
    
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        # short_name = "L. Messi", long_name = "Lionel Andrés Messi"
        for col in ["short_name", "long_name"]:
            if col in row and row[col]:
                name = row[col].strip()
                if name and len(name) > 2:
                    names.add(name)
    
    return names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download soccer player names from FIFA video game datasets"
    )
    parser.add_argument("--output", "-o", default="fifa_players.jsonl")
    parser.add_argument(
        "--version",
        choices=["fifa24", "fifa23", "fifa22", "fifa21", "fifa20", "all"],
        default="fifa24",
        help="Which FIFA version(s) to fetch",
    )
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    all_names: Set[str] = set()
    
    versions = list(FIFA_URLS.keys()) if args.version == "all" else [args.version]
    
    for version in versions:
        url = FIFA_URLS[version]
        print(f"Fetching {version.upper()}...", file=sys.stderr)
        try:
            names = fetch_fifa_csv(url, args.timeout)
            print(f"  → {len(names)} names", file=sys.stderr)
            all_names.update(names)
        except Exception as e:
            print(f"  ✗ Failed: {e}", file=sys.stderr)

    # Sort and write
    final = sorted(all_names)
    with open(args.output, "w", encoding="utf-8") as f:
        for name in final:
            f.write(json.dumps({"name": name}, ensure_ascii=False) + "\n")

    print(f"\n✓ Wrote {len(final)} unique player names to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
