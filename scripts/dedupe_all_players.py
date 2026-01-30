"""Deduplicate all_players.jsonl by normalized name, keeping highest fame_score."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict


_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower())).strip()


def fame_value(value: object) -> float:
    try:
        num = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(num):
        return 0.0
    return num


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/all_players.jsonl")
    parser.add_argument("--output", default="data/all_players.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    best_by_name: Dict[str, dict] = {}
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("full_name")
        if not name:
            continue
        key = normalize(name)
        if not key:
            continue
        current = best_by_name.get(key)
        if current is None:
            best_by_name[key] = obj
            continue
        if fame_value(obj.get("fame_score")) > fame_value(current.get("fame_score")):
            best_by_name[key] = obj

    with output_path.open("w", encoding="utf-8") as handle:
        for obj in best_by_name.values():
            handle.write(json.dumps(obj, ensure_ascii=True) + "\n")

    print(f"Wrote {len(best_by_name)} unique names to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
