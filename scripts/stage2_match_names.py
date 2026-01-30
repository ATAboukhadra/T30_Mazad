"""Stage 2: Match token n-grams to player names and rank by fame."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = text.lower()
    parts = _WORD_RE.findall(text)
    return " ".join(parts).strip()


def iter_text_tokens(asr_payload: dict) -> List[dict]:
    tokens: List[dict] = []
    for segment in asr_payload.get("segments", []):
        for token in segment.get("tokens", []):
            if token.get("is_timestamp"):
                continue
            text = (token.get("text") or "").strip()
            if not text:
                continue
            tokens.append(
                {
                    "text": text,
                    "start": token.get("start"),
                    "end": token.get("end"),
                }
            )
    return tokens


def load_players(path: Path) -> Dict[str, List[dict]]:
    players_by_name: Dict[str, List[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
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
        fame = obj.get("fame_score")
        obj["_fame"] = float(fame) if fame is not None and not math.isnan(float(fame)) else 0.0
        names = [name]
        full_name = obj.get("full_name")
        if full_name and full_name not in names:
            names.append(full_name)
        aliases = obj.get("aliases") or []
        for alias in aliases:
            if alias and alias not in names:
                names.append(alias)
        for candidate_name in names:
            key = normalize(candidate_name)
            if not key:
                continue
            players_by_name.setdefault(key, []).append(obj)
    for key, entries in players_by_name.items():
        entries.sort(key=lambda p: p.get("_fame", 0.0), reverse=True)
    return players_by_name


def build_ngrams(tokens: List[dict], min_n: int, max_n: int) -> List[Tuple[str, int, int]]:
    normalized = [normalize(t["text"]) for t in tokens]
    ngrams: List[Tuple[str, int, int]] = []
    for n in range(min_n, max_n + 1):
        for i in range(0, len(tokens) - n + 1):
            chunk = " ".join(normalized[i : i + n]).strip()
            if not chunk:
                continue
            ngrams.append((chunk, i, i + n - 1))
    return ngrams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asr_json", help="Stage-1 ASR JSON output")
    parser.add_argument(
        "--players",
        default="data/all_players.jsonl",
        help="JSONL of player data with fame_score",
    )
    parser.add_argument("--min-gram", type=int, default=1)
    parser.add_argument("--max-gram", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--output", help="Write match results to a JSON file")
    args = parser.parse_args()

    asr_payload = json.loads(Path(args.asr_json).read_text(encoding="utf-8"))
    tokens = iter_text_tokens(asr_payload)
    if not tokens:
        raise SystemExit("No tokens found in ASR payload.")

    players_by_name = load_players(Path(args.players))

    matches = []
    candidates: Dict[str, dict] = {}

    for chunk, start_idx, end_idx in build_ngrams(tokens, args.min_gram, args.max_gram):
        entries = players_by_name.get(chunk)
        if not entries:
            continue
        start_time = tokens[start_idx].get("start")
        end_time = tokens[end_idx].get("end")
        match_candidates = []
        for player in entries:
            name = player.get("name") or player.get("full_name")
            if not name:
                continue
            match_candidates.append(
                {
                    "name": name,
                    "fame_score": player.get("fame_score", 0.0),
                }
            )
            key = name.lower()
            if key not in candidates:
                candidates[key] = {
                    "name": name,
                    "fame_score": player.get("fame_score", 0.0),
                    "mentions": 0,
                    "matches": [],
                }
            candidates[key]["mentions"] += 1
            candidates[key]["matches"].append(
                {
                    "ngram": chunk,
                    "start": start_time,
                    "end": end_time,
                }
            )

        matches.append(
            {
                "ngram": chunk,
                "start": start_time,
                "end": end_time,
                "candidates": match_candidates,
            }
        )

    candidates_list = list(candidates.values())
    candidates_list.sort(key=lambda c: c.get("fame_score", 0.0), reverse=True)
    if args.max_candidates:
        candidates_list = candidates_list[: args.max_candidates]

    payload = {
        "source_asr": str(args.asr_json),
        "token_count": len(tokens),
        "match_count": len(matches),
        "matches": matches,
        "candidates": candidates_list,
    }

    output = json.dumps(payload, ensure_ascii=True, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
