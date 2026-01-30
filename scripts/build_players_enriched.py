"""Build an enriched player dataset by merging multiple sources.

Targets:
- All players in top 5 leagues (across available sources)
- All World Cup players (across history)

Merge strategy:
- Merge by normalized name
- Prefer newer sources for overlapping fields
- Accumulate clubs/leagues/sources lists
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TOP5_LEAGUES_FBREF = {
    "eng premier league",
    "es la liga",
    "it serie a",
    "fr ligue 1",
    "de bundesliga",
}

TOP5_LEAGUES_TRANSFERMARKT = {
    "gb1",
    "es1",
    "it1",
    "fr1",
    "l1",
    "de1",
}

TOP5_LEAGUES_FIFA = {
    "english premier league",
    "spanish primera division",
    "italian serie a",
    "french ligue 1",
    "german 1. bundesliga",
}


WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_name(name: str) -> str:
    return " ".join(WORD_RE.findall(name.lower())).strip()


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(clean_value(obj))
    return rows


def is_top5_league(source: str, league: Optional[str]) -> bool:
    if not league:
        return False
    league_norm = league.lower()
    if source.startswith("fbref"):
        return league_norm in TOP5_LEAGUES_FBREF
    if source.startswith("transfermarkt"):
        return league_norm in TOP5_LEAGUES_TRANSFERMARKT
    if source.startswith("fifa"):
        return league_norm in TOP5_LEAGUES_FIFA
    return False


def pick_name(record: Dict[str, Any]) -> Optional[str]:
    return record.get("name") or record.get("full_name")


def merge_records(base: Dict[str, Any], incoming: Dict[str, Any], prefer: bool) -> Dict[str, Any]:
    merged = dict(base)

    def set_if_missing(key: str) -> None:
        if key in incoming and incoming[key] not in (None, ""):
            if key not in merged or merged[key] in (None, ""):
                merged[key] = incoming[key]

    # Always preserve name
    if "name" in incoming and incoming["name"]:
        merged["name"] = merged.get("name") or incoming["name"]

    # Prefer incoming fields if it has higher priority
    if prefer:
        for key, value in incoming.items():
            if value in (None, ""):
                continue
            if key in ("sources", "clubs", "leagues"):
                continue
            merged[key] = value
    else:
        for key in ("nationality", "birth_year", "position", "club", "league", "full_name"):
            set_if_missing(key)

    # Accumulate clubs/leagues
    clubs = set(merged.get("clubs") or [])
    leagues = set(merged.get("leagues") or [])

    for key in ("club",):
        if incoming.get(key):
            clubs.add(str(incoming[key]))
    for key in ("clubs",):
        for club in incoming.get(key) or []:
            if club:
                clubs.add(str(club))

    if incoming.get("league"):
        leagues.add(str(incoming["league"]))
    for league in incoming.get("leagues") or []:
        if league:
            leagues.add(str(league))

    if clubs:
        merged["clubs"] = sorted(clubs)
    if leagues:
        merged["leagues"] = sorted(leagues)

    # Sources list
    sources = set(merged.get("sources") or [])
    source = incoming.get("source") or incoming.get("sources")
    if isinstance(source, list):
        sources.update([s for s in source if s])
    elif source:
        sources.add(source)
    if sources:
        merged["sources"] = sorted(sources)

    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbref", default="data/fbref_2025_players.jsonl")
    parser.add_argument("--transfermarkt", default="data/transfermarkt_players.jsonl")
    parser.add_argument("--fifa", default="data/fifa_all_players.jsonl")
    parser.add_argument("--worldcup", default="data/worldcup_players.jsonl")
    parser.add_argument("--openfootball", default="data/openfootball_players.jsonl")
    parser.add_argument("--output", default="data/players_enriched.jsonl")
    args = parser.parse_args()

    source_priority = {
        "fbref_2025_2026": 3,
        "transfermarkt_kaggle": 2,
        "fifa": 1,
        "worldcup": 0,
        "openfootball": 0,
    }

    def priority_of(record: Dict[str, Any]) -> int:
        source = record.get("source") or ""
        for key, val in source_priority.items():
            if key in str(source):
                return val
        return 0

    merged: Dict[str, Dict[str, Any]] = {}
    best_priority: Dict[str, int] = {}

    def ingest(records: Iterable[Dict[str, Any]], source_hint: str) -> None:
        for record in records:
            name = pick_name(record)
            if not name:
                continue
            record = dict(record)
            if "source" not in record and source_hint:
                record["source"] = source_hint
            key = normalize_name(name)
            if not key:
                continue

            # Filter: only top-5 leagues for league-based sources
            league = record.get("league")
            source = str(record.get("source") or "")
            if source_hint in ("fbref_2025_2026", "transfermarkt_kaggle", "fifa"):
                if not is_top5_league(source_hint, league):
                    continue

            prio = priority_of(record)
            if key not in merged:
                merged[key] = dict(record)
                merged[key]["name"] = name
                merged[key]["sources"] = [record.get("source") or source_hint]
                best_priority[key] = prio
                continue
            prefer = prio >= best_priority.get(key, 0)
            merged[key] = merge_records(merged[key], record, prefer)
            best_priority[key] = max(best_priority.get(key, 0), prio)

    fbref = load_jsonl(Path(args.fbref))
    transfermarkt = load_jsonl(Path(args.transfermarkt))
    fifa = load_jsonl(Path(args.fifa))
    worldcup = load_jsonl(Path(args.worldcup))
    openfootball = load_jsonl(Path(args.openfootball))

    ingest(fbref, "fbref_2025_2026")
    ingest(transfermarkt, "transfermarkt_kaggle")
    ingest(fifa, "fifa")
    ingest(worldcup, "worldcup")
    ingest(openfootball, "openfootball")

    with Path(args.output).open("w", encoding="utf-8") as handle:
        for obj in merged.values():
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(merged)} players to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
