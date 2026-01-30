"""Clean all_players.jsonl by removing fixture rows and normalizing fields."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict


_FIXTURE_RE = re.compile(r"\b(vs\.?|v)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    name = name.replace("\r", " ").replace("\n", " ")
    name = _WHITESPACE_RE.sub(" ", name).strip()
    return name


def is_fixture_name(name: str) -> bool:
    if "\n" in name or "\r" in name:
        return True
    if _FIXTURE_RE.search(name):
        return True
    return False


def clean_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_value(v) for v in value]
    if isinstance(value, str):
        return value.strip()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/all_players.jsonl")
    parser.add_argument("--output", default="data/all_players.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    kept = 0
    removed = 0

    cleaned: Dict[str, dict] = {}

    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            removed += 1
            continue

        name = obj.get("name") or obj.get("full_name")
        if not name:
            removed += 1
            continue
        name = normalize_name(str(name))
        if not name or is_fixture_name(name):
            removed += 1
            continue

        obj["name"] = name
        if "full_name" in obj and obj["full_name"]:
            obj["full_name"] = normalize_name(str(obj["full_name"]))
        obj = clean_value(obj)

        key = name.lower()
        cleaned[key] = obj
        kept += 1

    with output_path.open("w", encoding="utf-8") as handle:
        for obj in cleaned.values():
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Kept {kept} rows, removed {removed} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
