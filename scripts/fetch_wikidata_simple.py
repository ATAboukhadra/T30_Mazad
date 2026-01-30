"""Fetch soccer players from Wikidata - simplified and reliable version.

Uses smaller batches and simpler queries to avoid timeouts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Dict, List, Set


ENDPOINT = "https://query.wikidata.org/sparql"


def run_query(query: str, user_agent: str, timeout: int = 120) -> List[dict]:
    """Execute SPARQL query with proper headers."""
    params = urllib.parse.urlencode({"format": "json", "query": query})
    url = ENDPOINT + "?" + params
    
    req = urllib.request.Request(url, headers={
        "User-Agent": user_agent,
        "Accept": "application/sparql-results+json",
    })
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", {}).get("bindings", [])
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return []


def fetch_players_simple(limit: int, offset: int, user_agent: str, timeout: int) -> List[dict]:
    """Simple query - just get player names. No ORDER BY (too slow)."""
    query = f"""
SELECT ?player ?name WHERE {{
  ?player wdt:P31 wd:Q5 .
  ?player wdt:P106 wd:Q937857 .
  ?player rdfs:label ?name .
  FILTER(LANG(?name) = "en")
}}
LIMIT {limit}
OFFSET {offset}
"""
    return run_query(query, user_agent, timeout)


def fetch_players_with_sitelinks(min_sitelinks: int, limit: int, offset: int, user_agent: str, timeout: int) -> List[dict]:
    """Query with sitelinks filter - gets more famous players."""
    query = f"""
SELECT ?player ?name ?sitelinks WHERE {{
  ?player wdt:P31 wd:Q5 ;
          wdt:P106 wd:Q937857 ;
          wikibase:sitelinks ?sitelinks ;
          rdfs:label ?name .
  FILTER(LANG(?name) = "en")
  FILTER(?sitelinks >= {min_sitelinks})
}}
LIMIT {limit}
OFFSET {offset}
"""
    return run_query(query, user_agent, timeout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch soccer players from Wikidata (simple, reliable)"
    )
    parser.add_argument("--output", "-o", default="wikidata_players.jsonl")
    parser.add_argument("--limit", type=int, default=50000, help="Total players to fetch")
    parser.add_argument("--batch", type=int, default=5000, help="Players per query")
    parser.add_argument("--timeout", type=int, default=120, help="Query timeout")
    parser.add_argument("--sleep", type=float, default=3.0, help="Sleep between queries")
    parser.add_argument("--min-sitelinks", type=int, default=0, 
                        help="Minimum Wikipedia sitelinks (0=all, 5=notable, 20=famous)")
    parser.add_argument(
        "--user-agent",
        default="T30_Mazad/1.0 (football database project; https://github.com)",
        help="User-Agent with contact info (required by Wikidata)",
    )
    args = parser.parse_args()

    all_players: Dict[str, dict] = {}
    offset = 0
    batch_num = 0
    consecutive_empty = 0
    max_consecutive_empty = 3
    
    print(f"Fetching up to {args.limit:,} players from Wikidata...", file=sys.stderr)
    print(f"Min sitelinks: {args.min_sitelinks}, Batch size: {args.batch}", file=sys.stderr)
    
    while len(all_players) < args.limit:
        batch_num += 1
        print(f"\n[Batch {batch_num}] Offset {offset:,}...", file=sys.stderr, end=" ", flush=True)
        
        # Choose query based on sitelinks filter
        if args.min_sitelinks > 0:
            rows = fetch_players_with_sitelinks(
                args.min_sitelinks, args.batch, offset, args.user_agent, args.timeout
            )
        else:
            rows = fetch_players_simple(args.batch, offset, args.user_agent, args.timeout)
        
        if not rows:
            consecutive_empty += 1
            print(f"empty (attempt {consecutive_empty}/{max_consecutive_empty})", file=sys.stderr)
            if consecutive_empty >= max_consecutive_empty:
                print("Too many empty results, stopping.", file=sys.stderr)
                break
            time.sleep(args.sleep * 2)  # Wait longer on empty
            offset += args.batch
            continue
        
        consecutive_empty = 0
        
        for row in rows:
            qid = row["player"]["value"].rsplit("/", 1)[-1]
            name = row["name"]["value"].strip()
            if not name or len(name) < 2:
                continue
            
            sitelinks = int(row.get("sitelinks", {}).get("value", 0))
            
            # Keep entry with most sitelinks if duplicate
            if qid not in all_players or sitelinks > all_players[qid].get("sitelinks", 0):
                all_players[qid] = {
                    "name": name,
                    "qid": qid,
                    "sitelinks": sitelinks,
                }
        
        print(f"got {len(rows)}, total: {len(all_players):,}", file=sys.stderr)
        
        offset += args.batch
        time.sleep(args.sleep)
    
    # Sort by sitelinks (fame proxy) and write
    players_list = sorted(all_players.values(), key=lambda x: x.get("sitelinks", 0), reverse=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for i, p in enumerate(players_list[:args.limit], 1):
            p["rank"] = i
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Wrote {min(len(players_list), args.limit):,} players to {args.output}")
    
    # Show top 20
    print("\nTop 20 by Wikipedia presence:", file=sys.stderr)
    for p in players_list[:20]:
        print(f"  {p['name']} (sitelinks={p.get('sitelinks', 0)})", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
