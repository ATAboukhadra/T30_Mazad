"""Simple, fast footballer name fetcher from Wikidata.

Uses the most minimal query possible - no sitelinks, no ordering.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request


ENDPOINT = "https://query.wikidata.org/sparql"


def fetch_batch(offset: int, limit: int, timeout: int) -> list:
    """Minimal query - just names."""
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
    
    params = urllib.parse.urlencode({"format": "json", "query": query})
    url = ENDPOINT + "?" + params
    
    req = urllib.request.Request(url, headers={
        "User-Agent": "T30_Mazad/1.0 (soccer; contact@example.com)",
        "Accept": "application/sparql-results+json",
    })
    
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("results", {}).get("bindings", [])
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="players_simple.jsonl")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    all_names = {}
    offset = 0
    
    print(f"Fetching players...", file=sys.stderr)
    
    while len(all_names) < args.limit:
        print(f"  Batch at offset {offset}...", file=sys.stderr, end=" ", flush=True)
        rows = fetch_batch(offset, args.batch, args.timeout)
        
        if not rows:
            print("no results, stopping.", file=sys.stderr)
            break
        
        for row in rows:
            qid = row["player"]["value"].rsplit("/", 1)[-1]
            name = row["name"]["value"].strip()
            if name and qid not in all_names:
                all_names[qid] = name
        
        print(f"got {len(rows)}, total: {len(all_names)}", file=sys.stderr)
        offset += args.batch
        time.sleep(args.sleep)
    
    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        for qid, name in sorted(all_names.items(), key=lambda x: x[1]):
            f.write(json.dumps({"name": name, "qid": qid}) + "\n")
    
    print(f"\n✓ Wrote {len(all_names)} players to {args.output}")


if __name__ == "__main__":
    main()
