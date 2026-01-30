"""Extract player names from FIFA video game dataset (Kaggle).

This includes ALL players from leagues worldwide, not just national team players.
Source: https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def extract_players_from_csv(filepath: str) -> List[Dict]:
    """Extract player data from FIFA CSV."""
    players = []
    
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            short_name = row.get("short_name", "").strip()
            long_name = row.get("long_name", "").strip()
            
            if not short_name and not long_name:
                continue
            
            player = {
                "name": long_name or short_name,
                "short_name": short_name,
                "overall": int(row.get("overall", 0) or 0),
                "potential": int(row.get("potential", 0) or 0),
                "age": int(row.get("age", 0) or 0),
                "nationality": row.get("nationality_name", ""),
                "club": row.get("club_name", ""),
                "league": row.get("league_name", ""),
                "position": row.get("player_positions", ""),
                "value_eur": int(float(row.get("value_eur", 0) or 0)),
                "wage_eur": int(float(row.get("wage_eur", 0) or 0)),
            }
            players.append(player)
    
    return players


def compute_fame_score(player: Dict) -> float:
    """Compute fame score based on rating, value, age."""
    # Higher overall rating = more famous
    # Higher value = more famous
    # Prime age (25-32) bonus
    
    rating_score = player["overall"] * 2  # 0-186
    value_score = min(player["value_eur"] / 1_000_000, 100)  # 0-100
    
    age = player["age"]
    if 25 <= age <= 32:
        age_bonus = 20  # Prime years
    elif 20 <= age < 25:
        age_bonus = 15  # Rising star
    elif age < 20:
        age_bonus = 10  # Youth prospect
    else:
        age_bonus = 5  # Veteran
    
    return rating_score + value_score + age_bonus


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract players from FIFA CSV datasets"
    )
    parser.add_argument("inputs", nargs="+", help="FIFA CSV files to process")
    parser.add_argument("--output", "-o", default="fifa_players.jsonl")
    parser.add_argument("--min-overall", type=int, default=0, help="Minimum overall rating")
    parser.add_argument("--top-n", type=int, default=0, help="Only output top N by fame score")
    parser.add_argument("--names-only", action="store_true", help="Output just names (no stats)")
    args = parser.parse_args()

    all_players: Dict[str, Dict] = {}  # Dedupe by long name
    
    for filepath in args.inputs:
        print(f"Processing {filepath}...", file=sys.stderr)
        players = extract_players_from_csv(filepath)
        print(f"  → {len(players)} players", file=sys.stderr)
        
        for p in players:
            key = p["name"].lower()
            # Keep the one with higher overall if duplicate
            if key not in all_players or p["overall"] > all_players[key]["overall"]:
                all_players[key] = p
    
    # Filter by minimum overall
    players_list = list(all_players.values())
    if args.min_overall > 0:
        players_list = [p for p in players_list if p["overall"] >= args.min_overall]
    
    # Compute fame scores and sort
    for p in players_list:
        p["fame_score"] = round(compute_fame_score(p), 2)
    
    players_list.sort(key=lambda x: x["fame_score"], reverse=True)
    
    # Limit to top N
    if args.top_n > 0:
        players_list = players_list[:args.top_n]
    
    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for i, p in enumerate(players_list, 1):
            if args.names_only:
                f.write(json.dumps({"name": p["name"]}, ensure_ascii=False) + "\n")
            else:
                p["rank"] = i
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Wrote {len(players_list)} players to {args.output}")
    
    # Show top 20
    print("\nTop 20 players by fame score:", file=sys.stderr)
    for i, p in enumerate(players_list[:20], 1):
        print(f"  {i:2}. {p['name']} ({p['club']}) - {p['overall']} overall", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
