"""Fetch 100k+ soccer players from Wikidata with fame ranking.

Fame score based on:
- Sitelinks (Wikipedia pages in multiple languages = global recognition)
- International caps
- Goals scored
- Number of clubs played for
- Awards/trophies
- Recency (active players ranked higher)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set


ENDPOINT = "https://query.wikidata.org/sparql"
CURRENT_YEAR = datetime.now().year


@dataclass
class Player:
    qid: str
    name: str
    sitelinks: int = 0
    caps: int = 0
    goals: int = 0
    clubs_count: int = 0
    awards_count: int = 0
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    nationality: Optional[str] = None
    clubs: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    
    @property
    def is_active(self) -> bool:
        """Estimate if player is likely still active."""
        if self.death_year:
            return False
        if not self.birth_year:
            return False
        age = CURRENT_YEAR - self.birth_year
        return 16 <= age <= 42
    
    @property
    def recency_score(self) -> float:
        """Higher for recent/active players."""
        if not self.birth_year:
            return 0
        if self.death_year:
            # Historical player - score based on era
            return max(0, (self.death_year - 1900) / 50)
        age = CURRENT_YEAR - self.birth_year
        if age < 20:
            return 1.5  # Young talent
        elif age < 35:
            return 2.0  # Prime/active
        elif age < 45:
            return 1.0  # Recently retired
        else:
            return 0.5  # Legend but older
    
    def fame_score(self, weights: Dict[str, float]) -> float:
        """Compute overall fame score."""
        return (
            self.sitelinks * weights.get("sitelinks", 1.0)
            + self.caps * weights.get("caps", 0.5)
            + self.goals * weights.get("goals", 0.3)
            + self.clubs_count * weights.get("clubs", 2.0)
            + self.awards_count * weights.get("awards", 5.0)
            + self.recency_score * weights.get("recency", 10.0)
        )
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qid": self.qid,
            "fame_score": round(self.fame_score(DEFAULT_WEIGHTS), 2),
            "sitelinks": self.sitelinks,
            "caps": self.caps,
            "goals": self.goals,
            "clubs_count": self.clubs_count,
            "awards_count": self.awards_count,
            "birth_year": self.birth_year,
            "nationality": self.nationality,
            "clubs": self.clubs[:10],  # Top 10 clubs
            "aliases": self.aliases[:5],  # Top 5 aliases
        }


DEFAULT_WEIGHTS = {
    "sitelinks": 1.0,      # Wikipedia presence is strong fame indicator
    "caps": 0.5,           # International appearances
    "goals": 0.3,          # Goals scored
    "clubs": 2.0,          # Each club adds to fame
    "awards": 5.0,         # Trophies/awards
    "recency": 10.0,       # Bonus for active/recent players
}


def run_query(query: str, user_agent: str, timeout: int = 180, max_retries: int = 5) -> List[dict]:
    """Execute SPARQL query with retry logic."""
    params = urllib.parse.urlencode({"format": "json", "query": query})
    url = ENDPOINT + "?" + params
    
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": user_agent,
                "Accept": "application/sparql-results+json",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", {}).get("bindings", [])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == max_retries:
                print(f"  ✗ Query failed after {max_retries} attempts: {e}", file=sys.stderr)
                return []
            wait = min(120, attempt * 15)
            print(f"  ⚠ Attempt {attempt} failed, retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return []


def fetch_players_batch(offset: int, limit: int, user_agent: str, timeout: int) -> Dict[str, Player]:
    """Fetch a batch of players with basic info."""
    # Simpler query - no ORDER BY (too slow), we'll sort locally after collecting all
    query = f"""
SELECT ?player ?name ?sitelinks WHERE {{
  ?player wdt:P31 wd:Q5;
          wdt:P106 wd:Q937857;
          wikibase:sitelinks ?sitelinks;
          rdfs:label ?name.
  FILTER(LANG(?name) = "en")
  FILTER(?sitelinks > 3)
}}
LIMIT {limit}
OFFSET {offset}
"""
    
    rows = run_query(query, user_agent, timeout)
    players: Dict[str, Player] = {}
    
    for row in rows:
        qid = row["player"]["value"].rsplit("/", 1)[-1]
        name = row.get("name", {}).get("value", "").strip()
        if not name or qid in players:
            continue
        
        players[qid] = Player(
            qid=qid,
            name=name,
            sitelinks=int(row.get("sitelinks", {}).get("value", 0)),
        )
    
    return players


def fetch_stats_batch(qids: List[str], user_agent: str, timeout: int) -> Dict[str, dict]:
    """Fetch caps, goals for a batch of players."""
    if not qids:
        return {}
    
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?player ?caps ?goals WHERE {{
  VALUES ?player {{ {values} }}
  OPTIONAL {{ ?player wdt:P1097 ?caps. }}
  OPTIONAL {{ ?player wdt:P1351 ?goals. }}
}}
"""
    
    rows = run_query(query, user_agent, timeout)
    stats: Dict[str, dict] = {}
    
    for row in rows:
        qid = row["player"]["value"].rsplit("/", 1)[-1]
        caps = row.get("caps", {}).get("value")
        goals = row.get("goals", {}).get("value")
        stats[qid] = {
            "caps": int(float(caps)) if caps else 0,
            "goals": int(float(goals)) if goals else 0,
        }
    
    return stats


def fetch_clubs_batch(qids: List[str], user_agent: str, timeout: int) -> Dict[str, List[str]]:
    """Fetch clubs for a batch of players."""
    if not qids:
        return {}
    
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?player ?clubLabel WHERE {{
  VALUES ?player {{ {values} }}
  ?player wdt:P54 ?club.
  ?club wdt:P31/wdt:P279* wd:Q476028.  # Football club
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""
    
    rows = run_query(query, user_agent, timeout)
    clubs: Dict[str, List[str]] = {}
    
    for row in rows:
        qid = row["player"]["value"].rsplit("/", 1)[-1]
        club = row.get("clubLabel", {}).get("value", "")
        if club and not club.startswith("Q"):
            clubs.setdefault(qid, []).append(club)
    
    return clubs


def fetch_awards_count_batch(qids: List[str], user_agent: str, timeout: int) -> Dict[str, int]:
    """Fetch count of awards/honors for a batch of players."""
    if not qids:
        return {}
    
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?player (COUNT(?award) AS ?awards) WHERE {{
  VALUES ?player {{ {values} }}
  ?player wdt:P166 ?award.
}}
GROUP BY ?player
"""
    
    rows = run_query(query, user_agent, timeout)
    awards: Dict[str, int] = {}
    
    for row in rows:
        qid = row["player"]["value"].rsplit("/", 1)[-1]
        count = row.get("awards", {}).get("value", "0")
        awards[qid] = int(count)
    
    return awards


def fetch_aliases_batch(qids: List[str], user_agent: str, timeout: int) -> Dict[str, List[str]]:
    """Fetch aliases/alternate names for a batch of players."""
    if not qids:
        return {}
    
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?player ?alias WHERE {{
  VALUES ?player {{ {values} }}
  ?player skos:altLabel ?alias.
  FILTER(LANG(?alias) = "en")
}}
"""
    
    rows = run_query(query, user_agent, timeout)
    aliases: Dict[str, List[str]] = {}
    
    for row in rows:
        qid = row["player"]["value"].rsplit("/", 1)[-1]
        alias = row.get("alias", {}).get("value", "").strip()
        if alias:
            aliases.setdefault(qid, []).append(alias)
    
    return aliases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch 100k+ soccer players from Wikidata with fame ranking"
    )
    parser.add_argument("--output", "-o", default="players_ranked.jsonl")
    parser.add_argument("--limit", type=int, default=100000, help="Total players to fetch")
    parser.add_argument("--batch-size", type=int, default=2000, help="Players per batch")
    parser.add_argument("--timeout", type=int, default=180, help="Query timeout")
    parser.add_argument("--sleep", type=float, default=2.0, help="Sleep between batches")
    parser.add_argument("--stats", action="store_true", help="Fetch caps/goals (slower)")
    parser.add_argument("--clubs", action="store_true", help="Fetch club history (slower)")
    parser.add_argument("--awards", action="store_true", help="Fetch awards count (slower)")
    parser.add_argument("--aliases", action="store_true", help="Fetch alternate names")
    parser.add_argument("--all", action="store_true", help="Fetch all extra data")
    parser.add_argument("--min-sitelinks", type=int, default=0, help="Minimum sitelinks filter")
    parser.add_argument(
        "--user-agent",
        default="T30_Mazad/1.0 (soccer player database; github.com)",
        help="User-Agent header (required by Wikidata)",
    )
    args = parser.parse_args()

    if args.all:
        args.stats = args.clubs = args.awards = args.aliases = True

    all_players: Dict[str, Player] = {}
    offset = 0
    batch_num = 0
    
    print(f"Fetching up to {args.limit:,} soccer players from Wikidata...", file=sys.stderr)
    print(f"Options: stats={args.stats}, clubs={args.clubs}, awards={args.awards}, aliases={args.aliases}", file=sys.stderr)
    
    while len(all_players) < args.limit:
        batch_num += 1
        print(f"\n[Batch {batch_num}] Fetching players {offset:,} - {offset + args.batch_size:,}...", file=sys.stderr)
        
        # Fetch main player data
        players = fetch_players_batch(offset, args.batch_size, args.user_agent, args.timeout)
        if not players:
            print("  No more players found.", file=sys.stderr)
            break
        
        # Filter by minimum sitelinks
        if args.min_sitelinks > 0:
            players = {q: p for q, p in players.items() if p.sitelinks >= args.min_sitelinks}
        
        print(f"  → {len(players)} players", file=sys.stderr)
        
        qids = list(players.keys())
        
        # Fetch additional data in sub-batches
        sub_batch = 200
        for i in range(0, len(qids), sub_batch):
            sub_qids = qids[i:i + sub_batch]
            
            if args.stats:
                stats = fetch_stats_batch(sub_qids, args.user_agent, args.timeout)
                for qid, s in stats.items():
                    if qid in players:
                        players[qid].caps = s["caps"]
                        players[qid].goals = s["goals"]
                time.sleep(0.5)
            
            if args.clubs:
                clubs = fetch_clubs_batch(sub_qids, args.user_agent, args.timeout)
                for qid, c in clubs.items():
                    if qid in players:
                        players[qid].clubs = list(set(c))
                        players[qid].clubs_count = len(players[qid].clubs)
                time.sleep(0.5)
            
            if args.awards:
                awards = fetch_awards_count_batch(sub_qids, args.user_agent, args.timeout)
                for qid, count in awards.items():
                    if qid in players:
                        players[qid].awards_count = count
                time.sleep(0.5)
            
            if args.aliases:
                aliases = fetch_aliases_batch(sub_qids, args.user_agent, args.timeout)
                for qid, a in aliases.items():
                    if qid in players:
                        players[qid].aliases = list(set(a))
                time.sleep(0.5)
        
        all_players.update(players)
        offset += args.batch_size
        
        print(f"  Total collected: {len(all_players):,}", file=sys.stderr)
        time.sleep(args.sleep)
    
    # Sort by fame score
    print(f"\nRanking {len(all_players):,} players by fame...", file=sys.stderr)
    ranked = sorted(all_players.values(), key=lambda p: p.fame_score(DEFAULT_WEIGHTS), reverse=True)
    
    # Limit to requested count
    ranked = ranked[:args.limit]
    
    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for i, player in enumerate(ranked, 1):
            data = player.to_dict()
            data["rank"] = i
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Wrote {len(ranked):,} players to {args.output}", file=sys.stderr)
    
    # Show top 20
    print("\nTop 20 players by fame score:", file=sys.stderr)
    for i, p in enumerate(ranked[:20], 1):
        print(f"  {i:2}. {p.name} (score={p.fame_score(DEFAULT_WEIGHTS):.1f}, sitelinks={p.sitelinks})", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
