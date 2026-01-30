"""Stream-extract footballer names from the Wikidata JSON dump."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import urllib.request
from typing import Iterable, Optional


WIKIDATA_FOOTBALLER_QID = "Q937857"
HUMAN_QID = "Q5"
MALE_QID = "Q6581097"
FEMALE_QID = "Q6581072"


def iter_entities(handle: io.TextIOBase) -> Iterable[dict]:
    for line in handle:
        line = line.strip()
        if not line or line in ("[", "]"):
            continue
        if line.endswith(","):
            line = line[:-1]
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def extract_qids(entity: dict, prop: str) -> set:
    qids = set()
    claims = entity.get("claims", {}).get(prop, [])
    for claim in claims:
        datavalue = claim.get("mainsnak", {}).get("datavalue", {})
        value = datavalue.get("value", {})
        if isinstance(value, dict):
            qid = value.get("id")
            if qid:
                qids.add(qid)
    return qids


def open_source(path: Optional[str], url: Optional[str]) -> io.TextIOBase:
    if url:
        req = urllib.request.Request(url, headers={"User-Agent": "T30_Mazad/1.0"})
        resp = urllib.request.urlopen(req, timeout=60)
        gz = gzip.GzipFile(fileobj=resp)
        return io.TextIOWrapper(gz, encoding="utf-8")
    if not path:
        raise ValueError("Provide --path or --url")
    if path.endswith(".gz"):
        gz = gzip.open(path, "rb")
        return io.TextIOWrapper(gz, encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to latest-all.json.gz")
    parser.add_argument("--url", help="URL to latest-all.json.gz")
    parser.add_argument("--output", default="wikidata_players.jsonl")
    parser.add_argument("--language", default="en")
    parser.add_argument("--gender", choices=("male", "female", "any"), default="male")
    parser.add_argument("--with-aliases", action="store_true")
    args = parser.parse_args()

    gender_qid = {"male": MALE_QID, "female": FEMALE_QID}.get(args.gender)

    count = 0
    with open_source(args.path, args.url) as handle, open(
        args.output, "w", encoding="utf-8"
    ) as out:
        for entity in iter_entities(handle):
            if entity.get("type") != "item":
                continue
            if HUMAN_QID not in extract_qids(entity, "P31"):
                continue
            if WIKIDATA_FOOTBALLER_QID not in extract_qids(entity, "P106"):
                continue
            if gender_qid and gender_qid not in extract_qids(entity, "P21"):
                continue

            labels = entity.get("labels", {})
            label = labels.get(args.language, {}).get("value")
            if not label:
                continue

            payload = {"name": label, "qid": entity.get("id")}
            if args.with_aliases:
                aliases = entity.get("aliases", {}).get(args.language, [])
                payload["aliases"] = [a.get("value") for a in aliases if a.get("value")]

            out.write(json.dumps(payload, ensure_ascii=True) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"Extracted {count}", file=sys.stderr)

    print(f"Wrote {count} names to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
