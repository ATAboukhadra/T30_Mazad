"""Post-process transcript tokens into best-matching player names."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from tqdm import tqdm


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Common ASR confusions
    text = text.replace("z", "s")
    return text


def load_players(path: str) -> Tuple[Dict[str, dict], Dict[str, List[str]], Dict[str, float], float]:
    full_map: Dict[str, dict] = {}
    last_map: Dict[str, List[str]] = {}
    fame_map: Dict[str, float] = {}
    max_fame = 0.0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("name") or ""
            if not name:
                continue
            key = normalize(name)
            full_map[key] = obj
            try:
                fame = float(obj.get("fame_score") or 0.0)
            except ValueError:
                fame = 0.0
            fame_map[name] = fame
            if fame > max_fame:
                max_fame = fame
            parts = name.split()
            if len(parts) > 1:
                last = normalize(parts[-1])
                last_map.setdefault(last, []).append(name)
                if "-" in parts[-1]:
                    tail = normalize(parts[-1].split("-")[-1])
                    if tail:
                        last_map.setdefault(tail, []).append(name)
    return full_map, last_map, fame_map, max_fame


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def best_match(
    candidate: str,
    full_map: Dict[str, dict],
    last_map: Dict[str, List[str]],
    fame_map: Dict[str, float],
    max_fame: float,
    last_only: bool,
    min_score: float,
    top_k: int,
    fame_weight: float,
) -> Tuple[str, float, str, List[Dict]]:
    """Return (matched_name, score, match_type, candidates)."""
    cand_norm = normalize(candidate)
    if not cand_norm:
        return "", 0.0, "", []

    best_name = ""
    best_score = 0.0
    match_type = ""
    candidates: List[Dict] = []

    def combined_score(score: float, name: str) -> float:
        if max_fame <= 0:
            return score
        return score + fame_weight * (fame_map.get(name, 0.0) / max_fame)

    # Full-name matches
    for key, obj in full_map.items():
        score = similarity(cand_norm, key)
        if score >= min_score:
            name = obj.get("name") or obj.get("full_name") or ""
            combo = combined_score(score, name)
            candidates.append(
                {
                    "name": name,
                    "score": score,
                    "match_type": "full",
                    "fame_score": fame_map.get(name, 0.0),
                    "combined_score": combo,
                }
            )
        if score > best_score:
            best_score = score
            best_name = obj.get("name") or obj.get("full_name") or ""
            match_type = "full"

    # Last-name matches
    if last_only:
        for last, names in last_map.items():
            score = similarity(cand_norm, last)
            if len(cand_norm) >= 2 and len(cand_norm) <= 3:
                if last.startswith(cand_norm) or cand_norm.startswith(last):
                    score = max(score, min_score)
            if score >= min_score:
                for name in names:
                    combo = combined_score(score, name)
                    candidates.append(
                        {
                            "name": name,
                            "score": score,
                            "match_type": "last",
                            "fame_score": fame_map.get(name, 0.0),
                            "combined_score": combo,
                        }
                    )
            if score > best_score and names:
                best_score = score
                best_name = max(names, key=lambda n: fame_map.get(n, 0.0))
                match_type = "last"

    if best_score < min_score:
        return "", best_score, "", []

    candidates.sort(key=lambda c: c.get("combined_score", c["score"]), reverse=True)
    candidates = candidates[:max(1, top_k)]
    return best_name, best_score, match_type, candidates


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text, flags=re.UNICODE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", required=True, help="Transcript text file")
    parser.add_argument("--player-db", required=True, help="Player database JSONL")
    parser.add_argument("--output", default="postprocessed_names.json")
    parser.add_argument("--max-gram", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.86)
    parser.add_argument("--last-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=5, help="Top candidates per token")
    parser.add_argument("--fame-weight", type=float, default=0.2,
                        help="Weight for fame bias in candidate ranking")
    args = parser.parse_args()

    full_map, last_map, fame_map, max_fame = load_players(args.player_db)
    text = open(args.transcript, "r", encoding="utf-8").read()
    tokens = tokenize(text)

    matches = []
    i = 0
    progress = tqdm(total=len(tokens), desc="Matching tokens")
    processed = 0
    while i < len(tokens):
        best = ("", 0.0, "", 1, [])
        for size in range(args.max_gram, 0, -1):
            if i + size > len(tokens):
                continue
            cand = " ".join(tokens[i : i + size])
            name, score, match_type, candidates = best_match(
                cand,
                full_map,
                last_map,
                fame_map,
                max_fame,
                args.last_only,
                args.min_score,
                args.top_k,
                args.fame_weight,
            )
            if score > best[1]:
                best = (name, score, match_type, size, candidates)
        if best[0]:
            matches.append(
                {
                    "name": best[0],
                    "score": best[1],
                    "match_type": best[2],
                    "token_span": [i, i + best[3]],
                    "token_text": " ".join(tokens[i : i + best[3]]),
                    "candidates": best[4],
                }
            )
            if best[4]:
                cand_str = ", ".join(
                    f"{c['name']}({c['score']:.2f})" for c in best[4]
                )
                progress.write(
                    f"Matched: {best[0]} ({best[2]}) -> {tokens[i:i+best[3]]} | candidates: {cand_str}"
                )
            else:
                progress.write(
                    f"Matched: {best[0]} ({best[2]}) -> {tokens[i:i+best[3]]}"
                )
            i += best[3]
        else:
            i += 1
        progress.update(1)
        processed += 1
        if processed % 10 == 0:
            progress.write(f"Processed {processed} tokens")

    progress.close()

    with open(args.output, "w", encoding="utf-8") as out:
        json.dump({"matches": matches}, out, indent=2, ensure_ascii=True)

    print(f"Wrote {len(matches)} matches to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
