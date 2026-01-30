"""Fetch footballer names from Wikidata into JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple


ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_USER_AGENT = "T30_Mazad/1.0 (contact: not-set)"


def build_query(limit: int, offset: int, language: str, priority_qids: List[str]) -> str:
    priority_clause = ""
    if priority_qids:
        values = " ".join(f"wd:{qid}" for qid in priority_qids)
        priority_clause = f"VALUES ?team {{ {values} }} . ?player wdt:P54 ?team."
    return f"""
SELECT ?player ?playerLabel WHERE {{
  ?player wdt:P31 wd:Q5.
  ?player wdt:P106 wd:Q937857.
  ?player rdfs:label ?playerLabel.
  FILTER (lang(?playerLabel) = \"{language}\").
  {priority_clause}
}}
LIMIT {limit}
OFFSET {offset}
""".strip()


def build_alias_query(qids: List[str], language: str) -> str:
    values = " ".join(f"wd:{qid}" for qid in qids)
    return f"""
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?player ?altLabel WHERE {{
  VALUES ?player {{ {values} }}
  ?player skos:altLabel ?altLabel.
  FILTER (lang(?altLabel) = \"{language}\").
}}
""".strip()


def build_sitelinks_query(qids: List[str]) -> str:
    values = " ".join(f"wd:{qid}" for qid in qids)
    return f"""
SELECT ?player ?sitelinks WHERE {{
  VALUES ?player {{ {values} }}
  ?player wikibase:sitelinks ?sitelinks.
}}
""".strip()


def build_stats_query(qids: List[str]) -> str:
    values = " ".join(f"wd:{qid}" for qid in qids)
    return f"""
SELECT ?player ?gender ?birth ?caps ?goals WHERE {{
  VALUES ?player {{ {values} }}
  OPTIONAL {{ ?player wdt:P21 ?gender. }}
  OPTIONAL {{ ?player wdt:P569 ?birth. }}
  OPTIONAL {{ ?player wdt:P1097 ?caps. }}
  OPTIONAL {{ ?player wdt:P1351 ?goals. }}
}}
""".strip()


def parse_int(value: Optional[str]) -> int:
    if not value:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def parse_year(date_value: Optional[str]) -> Optional[int]:
    if not date_value:
        return None
    try:
        return int(date_value[:4])
    except ValueError:
        return None


def compute_score(entry: dict, weights: dict) -> float:
    return (
        entry.get("sitelinks", 0) * weights.get("sitelinks", 1.0)
        + entry.get("caps", 0) * weights.get("caps", 0.2)
        + entry.get("goals", 0) * weights.get("goals", 0.1)
        + entry.get("recency", 0) * weights.get("recency", 0.05)
    )


RETRYABLE_CODES = {403, 429, 500, 502, 503, 504}


def run_query(
    query: str,
    timeout: int,
    user_agent: str,
    max_retries: int = 5,
) -> List[dict]:
    params = urllib.parse.urlencode({"format": "json", "query": query})
    url = ENDPOINT + "?" + params
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("results", {}).get("bindings", [])
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in RETRYABLE_CODES:
                raise
            if attempt == max_retries:
                raise
            sleep_for = min(90, attempt * 10)
            time.sleep(sleep_for)
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="wikidata_players.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-pages", type=int, default=0, help="0 = no limit")
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-sitelinks", type=int, default=0)
    parser.add_argument("--aliases", action="store_true", help="Include alt labels")
    parser.add_argument("--sitelinks", action="store_true", help="Attach sitelinks counts")
    parser.add_argument("--sort-by-sitelinks", action="store_true", help="Sort each batch by sitelinks")
    parser.add_argument("--sort-by-score", action="store_true", help="Sort each batch by computed score")
    parser.add_argument("--gender", choices=("male", "female", "any"), default="male")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--append", action="store_true", help="Append to output")
    parser.add_argument("--no-order", action="store_true", help="Disable ordering by sitelinks")
    parser.add_argument(
        "--priority-qids",
        default="",
        help="Comma-separated QIDs for priority clubs/national teams (P54)",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent with contact info (required by Wikidata)",
    )
    args = parser.parse_args()

    output_path = args.output
    offset = args.start_offset
    total = 0
    priority_qids = [qid.strip() for qid in args.priority_qids.split(",") if qid.strip()]
    gender_qid = {"male": "Q6581097", "female": "Q6581072"}.get(args.gender)
    weights = {"sitelinks": 1.0, "caps": 0.2, "goals": 0.1, "recency": 0.05}

    page = 0
    mode = "a" if args.append else "w"
    sleep_seconds = max(1.0, args.sleep)
    with open(output_path, mode, encoding="utf-8") as handle:
        while True:
            page += 1
            if args.max_pages and page > args.max_pages:
                break
            query = build_query(
                args.limit,
                offset,
                args.language,
                priority_qids=priority_qids,
            )
            rows = run_query(query, timeout=args.timeout, user_agent=args.user_agent)
            if not rows:
                break

            page_players: Dict[str, dict] = {}
            for row in rows:
                label = row["playerLabel"]["value"].strip()
                if not label:
                    continue
                qid = row["player"]["value"].rsplit("/", 1)[-1]
                page_players.setdefault(
                    qid, {"name": label, "qid": qid, "aliases": [], "sitelinks": 0}
                )

            if page_players:
                stats_query = build_stats_query(list(page_players.keys()))
                stats_rows = run_query(stats_query, timeout=args.timeout, user_agent=args.user_agent)
                for row in stats_rows:
                    qid = row["player"]["value"].rsplit("/", 1)[-1]
                    if qid not in page_players:
                        continue
                    entry = page_players[qid]
                    entry["gender"] = row.get("gender", {}).get("value")
                    entry["caps"] = parse_int(row.get("caps", {}).get("value"))
                    entry["goals"] = parse_int(row.get("goals", {}).get("value"))
                    birth = row.get("birth", {}).get("value")
                    entry["birth_year"] = parse_year(birth)
                    if entry.get("birth_year"):
                        entry["recency"] = entry["birth_year"]

            if args.sitelinks or args.min_sitelinks > 0 or args.sort_by_sitelinks or args.sort_by_score:
                sitelinks_query = build_sitelinks_query(list(page_players.keys()))
                sitelinks_rows = run_query(sitelinks_query, timeout=args.timeout, user_agent=args.user_agent)
                for row in sitelinks_rows:
                    qid = row["player"]["value"].rsplit("/", 1)[-1]
                    if qid in page_players:
                        page_players[qid]["sitelinks"] = int(row["sitelinks"]["value"])

            if args.aliases and page_players:
                alias_query = build_alias_query(list(page_players.keys()), args.language)
                alias_rows = run_query(alias_query, timeout=args.timeout, user_agent=args.user_agent)
                for row in alias_rows:
                    qid = row["player"]["value"].rsplit("/", 1)[-1]
                    alias = row["altLabel"]["value"].strip()
                    if alias and qid in page_players:
                        entry = page_players[qid]
                        if alias not in entry["aliases"]:
                            entry["aliases"].append(alias)

            entries = list(page_players.values())
            if gender_qid:
                entries = [e for e in entries if e.get("gender", "").endswith(gender_qid)]
            if args.min_sitelinks > 0:
                entries = [e for e in entries if e["sitelinks"] >= args.min_sitelinks]
            if args.sort_by_score:
                entries.sort(key=lambda e: compute_score(e, weights), reverse=True)
            elif args.sort_by_sitelinks and not args.no_order:
                entries.sort(key=lambda e: e.get("sitelinks", 0), reverse=True)

            for entry in entries:
                if not args.aliases:
                    entry.pop("aliases", None)
                if not args.sitelinks and args.min_sitelinks == 0 and not args.sort_by_sitelinks:
                    entry.pop("sitelinks", None)
                if "gender" in entry:
                    entry.pop("gender", None)
                entry.pop("caps", None)
                entry.pop("goals", None)
                entry.pop("birth_year", None)
                entry.pop("recency", None)
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
                total += 1

            offset += args.limit
            handle.flush()
            print(f"Fetched page {page} offset {offset} (total {total})", file=sys.stderr)
            time.sleep(sleep_seconds)

    print(f"Wrote {total} names to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
