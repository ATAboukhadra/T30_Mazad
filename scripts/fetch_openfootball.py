"""Extract player names from OpenFootball datasets.

Downloads from:
- World Cup (with full lineups)
- Players database (by nationality) 
- Champions League goalscorers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from typing import Dict, List, Set, Tuple


# World Cup files with full lineups
WORLDCUP_FILES = [
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/2022_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/2018_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/2014_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/2010_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/2006_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/2002_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/1998_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/1994_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/1990_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/1986_worldcup.txt",
    "https://raw.githubusercontent.com/openfootball/worldcup.more/master/worldcup/1982_worldcup.txt",
]

# Player database files by country (top 5 leagues + major nations)
PLAYER_DB_BASE = "https://raw.githubusercontent.com/openfootball/players/master"
PLAYER_DB_COUNTRIES = {
    # Top 5 leagues
    "england": "europe/england/eng.players.txt",
    "spain": "europe/spain/es.players.txt",
    "germany": "europe/germany/de.players.txt",
    "italy": "europe/italy/it.players.txt",
    "france": "europe/france/fr.players.txt",
    # Other major European
    "netherlands": "europe/netherlands/nl.players.txt",
    "portugal": "europe/portugal/pt.players.txt",
    "belgium": "europe/belgium/be.players.txt",
    "croatia": "europe/croatia/hr.players.txt",
    "denmark": "europe/denmark/dk.players.txt",
    "switzerland": "europe/switzerland/ch.players.txt",
    "austria": "europe/austria/at.players.txt",
    "poland": "europe/poland/pl.players.txt",
    "scotland": "europe/scotland/sco.players.txt",
    "turkey": "europe/turkey/tr.players.txt",
    "greece": "europe/greece/gr.players.txt",
    "ukraine": "europe/ukraine/ua.players.txt",
    "russia": "europe/russia/ru.players.txt",
    "serbia": "europe/serbia/rs.players.txt",
    "wales": "europe/wales/wal.players.txt",
    "ireland": "europe/ireland/ie.players.txt",
    "czech-republic": "europe/czech-republic/cz.players.txt",
    "sweden": "europe/sweden/se.players.txt",
    "norway": "europe/norway/no.players.txt",
    # South America
    "argentina": "south-america/argentina/ar.players.txt",
    "brazil": "south-america/brazil/br.players.txt",
    "colombia": "south-america/colombia/co.players.txt",
    "uruguay": "south-america/uruguay/uy.players.txt",
    "chile": "south-america/chile/cl.players.txt",
    "peru": "south-america/peru/pe.players.txt",
    "ecuador": "south-america/ecuador/ec.players.txt",
    # Africa
    "nigeria": "africa/nigeria/ng.players.txt",
    "senegal": "africa/senegal/sn.players.txt",
    "morocco": "africa/morocco/ma.players.txt",
    "egypt": "africa/egypt/eg.players.txt",
    "cameroon": "africa/cameroon/cm.players.txt",
    "ivory-coast": "africa/ivory-coast/ci.players.txt",
    "ghana": "africa/ghana/gh.players.txt",
    "algeria": "africa/algeria/dz.players.txt",
    "tunisia": "africa/tunisia/tn.players.txt",
    # North/Central America
    "mexico": "north-america/mexico/mx.players.txt",
    "usa": "north-america/united-states/us.players.txt",
    "canada": "north-america/canada/ca.players.txt",
    "costa-rica": "central-america/costa-rica/cr.players.txt",
    # Asia
    "japan": "asia/japan/jp.players.txt",
    "south-korea": "asia/south-korea/kr.players.txt",
    "iran": "middle-east/iran/ir.players.txt",
    "saudi-arabia": "middle-east/saudi-arabia/sa.players.txt",
    "australia": "pacific/australia/au.players.txt",
}

# Pattern to match lineup lines - they start with country name followed by colon
# e.g., "France: Hugo Lloris, Benjamin Pavard..."
LINEUP_PATTERN = re.compile(r'^([A-Z][a-zA-Z\s]+):\s+(.+)$', re.MULTILINE)

# Pattern to extract individual player names from lineup
# Handles: "Name", "Name (sub)", "Name (45' Sub Name)"
PLAYER_PATTERN = re.compile(r"([A-ZÀ-ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-ÿ]?[a-zà-ÿ'-]+)*)")

# Pattern to match goalscorers in goal lines
GOAL_PATTERN = re.compile(r"([A-ZÀ-ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-ÿ]?[a-zà-ÿ'-]+)*)\s+\d+")


def fetch_file(url: str, timeout: int = 60) -> str:
    """Download a file from URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_players_from_lineup(lineup_text: str) -> Set[str]:
    """Extract player names from a lineup string."""
    names: Set[str] = set()
    
    # Remove substitution times like (45' Name) - extract the sub name
    # Pattern: (number' Name)
    subs = re.findall(r"\((\d+['+]*)\s+([A-ZÀ-ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-ÿ]?[a-zà-ÿ'-]+)*)\)", lineup_text)
    for _, sub_name in subs:
        if len(sub_name) > 2:
            names.add(sub_name.strip())
    
    # Remove the substitution markers to clean the string
    clean = re.sub(r"\(\d+['+]*\s+[^)]+\)", "", lineup_text)
    clean = re.sub(r"\(\d+['+]*\)", "", clean)  # Remove time-only markers
    
    # Split by comma and extract names
    parts = clean.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Get the first name-like pattern
        match = PLAYER_PATTERN.match(part)
        if match:
            name = match.group(1).strip()
            if len(name) > 2 and not name.lower() in ("the", "and", "for"):
                names.add(name)
    
    return names


def extract_players_from_text(text: str) -> Set[str]:
    """Extract all player names from match data text."""
    players: Set[str] = set()
    
    # Find all lineup blocks
    for match in LINEUP_PATTERN.finditer(text):
        country = match.group(1).strip()
        lineup = match.group(2).strip()
        
        # Skip if it looks like a match header, not a lineup
        if "v " in country or len(lineup) < 20:
            continue
        
        extracted = extract_players_from_lineup(lineup)
        players.update(extracted)
    
    # Also extract goalscorers from goal lines
    for match in GOAL_PATTERN.finditer(text):
        name = match.group(1).strip()
        if len(name) > 2:
            players.add(name)
    
    return players


def extract_players_from_db_file(text: str) -> Set[str]:
    """Extract player names from players.txt database format.
    
    Format: "Name,  Position,  Height,  b. Date @ Place"
    """
    players: Set[str] = set()
    
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("#"):
            continue
        
        # Split by comma, first part is the name
        parts = line.split(",")
        if parts:
            name = parts[0].strip()
            # Validate it looks like a name (has letters, reasonable length)
            if name and len(name) > 2 and re.match(r"^[A-ZÀ-ÿ]", name):
                # Skip position markers that might be parsed as names
                if name not in ("F", "G", "M", "D", "GK", "DF", "MF", "FW"):
                    players.add(name)
    
    return players


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract player names from OpenFootball datasets"
    )
    parser.add_argument("--output", "-o", default="openfootball_players.jsonl")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--worldcup", action="store_true", help="Fetch World Cup lineups")
    parser.add_argument("--players-db", action="store_true", help="Fetch players database")
    parser.add_argument("--all", action="store_true", help="Fetch all sources")
    parser.add_argument("--countries", default="all", help="Comma-separated countries or 'all'")
    args = parser.parse_args()

    if args.all:
        args.worldcup = args.players_db = True
    
    # Default to all if nothing specified
    if not args.worldcup and not args.players_db:
        args.worldcup = args.players_db = True

    all_players: Set[str] = set()
    
    # Fetch World Cup data
    if args.worldcup:
        print("\n=== World Cup Lineups ===", file=sys.stderr)
        for url in WORLDCUP_FILES:
            year = re.search(r"(\d{4})", url)
            year_str = year.group(1) if year else "unknown"
            print(f"Fetching World Cup {year_str}...", file=sys.stderr, end=" ", flush=True)
            
            try:
                text = fetch_file(url, args.timeout)
                players = extract_players_from_text(text)
                print(f"→ {len(players)} players", file=sys.stderr)
                all_players.update(players)
            except Exception as e:
                print(f"✗ {e}", file=sys.stderr)
    
    # Fetch players database
    if args.players_db:
        print("\n=== Players Database ===", file=sys.stderr)
        
        countries = list(PLAYER_DB_COUNTRIES.keys())
        if args.countries != "all":
            requested = [c.strip().lower() for c in args.countries.split(",")]
            countries = [c for c in countries if c in requested]
        
        for country in countries:
            path = PLAYER_DB_COUNTRIES[country]
            url = f"{PLAYER_DB_BASE}/{path}"
            print(f"Fetching {country}...", file=sys.stderr, end=" ", flush=True)
            
            try:
                text = fetch_file(url, args.timeout)
                players = extract_players_from_db_file(text)
                print(f"→ {len(players)} players", file=sys.stderr)
                all_players.update(players)
            except Exception as e:
                print(f"✗ {e}", file=sys.stderr)
    
    # Sort and write
    final = sorted(all_players)
    with open(args.output, "w", encoding="utf-8") as f:
        for name in final:
            f.write(json.dumps({"name": name}, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Wrote {len(final)} unique player names to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
