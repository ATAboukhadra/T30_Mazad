"""Reduce player database to top-N by fame_score."""

from __future__ import annotations

import argparse
import json
from typing import List


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/all_players.jsonl")
    parser.add_argument("--output", default="data/top_10k_players.jsonl")
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    players: List[dict] = []
    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fame = obj.get("fame_score")
            if fame is None:
                fame = 0.0
            obj["_fame"] = float(fame)
            players.append(obj)

    players.sort(key=lambda p: p.get("_fame", 0.0), reverse=True)
    top = players[: args.limit]

    with open(args.output, "w", encoding="utf-8") as out:
        for obj in top:
            obj.pop("_fame", None)
            out.write(json.dumps(obj, ensure_ascii=True) + "\n")

    print(f"Wrote {len(top)} players to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
