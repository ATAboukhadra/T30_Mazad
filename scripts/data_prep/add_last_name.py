"""Add last_name field to a JSONL player dataset."""

from __future__ import annotations

import argparse
import json


def last_name(full: str) -> str:
    parts = full.strip().split()
    return parts[-1] if parts else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/all_players.jsonl")
    parser.add_argument("--output", default="data/all_players_with_last.jsonl")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as src, open(
        args.output, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("name") or obj.get("full_name") or ""
            obj["last_name"] = last_name(name)
            dst.write(json.dumps(obj, ensure_ascii=True) + "\n")

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
