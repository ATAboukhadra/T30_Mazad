"""Stage 2: Match token n-grams to player names using fuzzy matching.

Reads tokens CSV from stage1 (with multiple passes), processes each pass
independently with sliding window n-grams, and suggests multiple player
candidates for each detected name pattern.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

try:
    from thefuzz import fuzz as thefuzz_fuzz, process as thefuzz_process
    HAS_THEFUZZ = True
except ImportError:
    HAS_THEFUZZ = False


_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Normalize text for matching."""
    text = text.lower()
    parts = _WORD_RE.findall(text)
    return " ".join(parts).strip()


def load_tokens_csv(path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """Load tokens CSV and group by pass number."""
    passes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pass_num = int(row.get("pass", 1))
            passes[pass_num].append({
                "token": row.get("token", ""),
                "segment_start": float(row.get("segment_start", 0)),
                "segment_end": float(row.get("segment_end", 0)),
                "probability": float(row.get("probability", 0)),
                "avg_logprob": float(row.get("avg_logprob", 0)),
            })
    return dict(passes)


def load_players(path: Path) -> Tuple[Dict[str, List[dict]], List[str]]:
    """Load player database and build lookup structures.
    
    Returns:
        - players_by_name: dict mapping normalized name -> list of player records
        - all_names: list of all normalized player names for fuzzy matching
    """
    players_by_name: Dict[str, List[dict]] = {}
    all_names: Set[str] = set()
    
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
        
        # Compute career-based score
        obj["_career_score"] = compute_career_score(obj)
        
        # Collect all name variants
        names = [name]
        full_name = obj.get("full_name")
        if full_name and full_name not in names:
            names.append(full_name)
        
        # Also add last name as a variant
        parts = name.split()
        if len(parts) > 1:
            names.append(parts[-1])  # Last name only
        
        aliases = obj.get("aliases") or []
        for alias in aliases:
            if alias and alias not in names:
                names.append(alias)
        
        for candidate_name in names:
            key = normalize(candidate_name)
            if not key:
                continue
            all_names.add(key)
            players_by_name.setdefault(key, []).append(obj)
    
    # Sort each name's players by career score
    for key, entries in players_by_name.items():
        entries.sort(key=lambda p: p.get("_career_score", 0.0), reverse=True)
    
    return players_by_name, sorted(all_names)


# League tier weights (higher = more prestigious)
LEAGUE_WEIGHTS = {
    # Top 5 leagues
    "eng premier league": 100, "premier league": 100, "english premier league": 100, "gb1": 100,
    "es la liga": 95, "la liga": 95, "es1": 95,
    "de bundesliga": 90, "bundesliga": 90, "l1": 90,
    "it serie a": 88, "serie a": 88, "it1": 88,
    "fr ligue 1": 85, "ligue 1": 85, "fr1": 85,
    # Second tier
    "eredivisie": 70, "nl1": 70,
    "primeira liga": 70, "pt1": 70,
    "scottish premiership": 60,
    "championship": 55, "eng championship": 55,
    "mls": 50,
    # International
    "champions league": 120, "ucl": 120,
    "europa league": 80,
    "world cup": 150,
    "euro": 130,
}

# Club prestige weights
CLUB_WEIGHTS = {
    # English
    "manchester city": 95, "manchester city football club": 95,
    "liverpool": 95, "liverpool fc": 95,
    "arsenal": 90, "arsenal fc": 90,
    "chelsea": 88, "chelsea fc": 88,
    "manchester united": 88, "manchester united football club": 88,
    "tottenham": 80, "tottenham hotspur": 80,
    # Spanish
    "real madrid": 98, "real madrid club de fútbol": 98,
    "barcelona": 95, "fc barcelona": 95,
    "atletico madrid": 85,
    # German
    "bayern munich": 95, "fc bayern münchen": 95,
    "borussia dortmund": 82,
    # Italian
    "juventus": 88,
    "inter milan": 85, "inter": 85,
    "ac milan": 85, "milan": 85,
    # French
    "paris saint-germain": 90, "psg": 90,
}


def compute_career_score(player: dict) -> float:
    """Compute a career-based score for a player.
    
    Factors:
    - Club prestige (current and historical)
    - League prestige
    - Minutes played (experience)
    - Goals and assists (contribution)
    - Number of data sources (recognition/coverage)
    """
    score = 0.0
    
    # Current club weight
    current_club = (player.get("club") or "").lower()
    score += CLUB_WEIGHTS.get(current_club, 30)  # Default 30 for unknown clubs
    
    # Historical clubs
    clubs = player.get("clubs") or []
    for club in clubs:
        club_lower = club.lower()
        club_score = CLUB_WEIGHTS.get(club_lower, 20)
        score += club_score * 0.3  # Historical clubs contribute 30% of their weight
    
    # Current league weight
    current_league = (player.get("league") or "").lower()
    score += LEAGUE_WEIGHTS.get(current_league, 20)
    
    # Historical leagues
    leagues = player.get("leagues") or []
    for league in leagues:
        league_lower = league.lower()
        league_score = LEAGUE_WEIGHTS.get(league_lower, 15)
        score += league_score * 0.2  # Historical leagues contribute 20%
    
    # Minutes played (experience factor)
    minutes = player.get("minutes_played") or 0
    if minutes > 0:
        # Log scale: 1000 mins = +10, 2000 mins = +15, 3000 mins = +18
        score += min(25, math.log10(minutes + 1) * 8)
    
    # Goals and assists (contribution)
    goals = player.get("goals") or 0
    assists = player.get("assists") or 0
    score += goals * 1.5 + assists * 1.0
    
    # Data source coverage (more sources = more recognized player)
    sources = player.get("sources") or []
    score += len(sources) * 5
    
    # World Cup participation bonus
    if "worldcup" in sources:
        score += 30
    
    return round(score, 2)


def fuzzy_match(query: str, choices: List[str], limit: int = 5, threshold: int = 70) -> List[Tuple[str, int]]:
    """Find fuzzy matches for a query string.
    
    Returns list of (matched_name, score) tuples.
    """
    if not query or not choices:
        return []
    
    if HAS_RAPIDFUZZ:
        # Use WRatio for better short string matching (handles transpositions, partial matches)
        results = process.extract(query, choices, scorer=fuzz.WRatio, limit=limit)
        return [(name, int(score)) for name, score, _ in results if score >= threshold]
    elif HAS_THEFUZZ:
        # Use WRatio for better matching
        results = thefuzz_process.extract(query, choices, scorer=thefuzz_fuzz.WRatio, limit=limit)
        return [(name, score) for name, score in results if score >= threshold]
    else:
        # Fallback: exact prefix/substring matching + 1-char difference tolerance
        matches = []
        query_lower = query.lower()
        for choice in choices:
            if query_lower == choice:
                matches.append((choice, 100))
            elif query_lower in choice or choice in query_lower:
                matches.append((choice, 85))
            # Check for 1-character difference (for short names like aki/ake)
            elif len(query_lower) == len(choice) and len(query_lower) <= 5:
                diff = sum(1 for a, b in zip(query_lower, choice) if a != b)
                if diff == 1:
                    matches.append((choice, 75))
        return sorted(matches, key=lambda x: -x[1])[:limit]


def build_ngrams(tokens: List[Dict], min_n: int, max_n: int) -> List[Tuple[str, int, int, List[Dict]]]:
    """Build n-grams from token list.
    
    Returns list of (ngram_text, start_idx, end_idx, tokens_in_ngram).
    """
    ngrams: List[Tuple[str, int, int, List[Dict]]] = []
    
    for n in range(min_n, max_n + 1):
        for i in range(0, len(tokens) - n + 1):
            token_slice = tokens[i:i + n]
            chunk = " ".join(normalize(t["token"]) for t in token_slice).strip()
            if not chunk:
                continue
            ngrams.append((chunk, i, i + n - 1, token_slice))
    
    return ngrams


def process_pass(
    pass_num: int,
    tokens: List[Dict],
    players_by_name: Dict[str, List[dict]],
    all_names: List[str],
    min_gram: int,
    max_gram: int,
    fuzzy_threshold: int,
    max_suggestions: int,
    search_cache: Dict[str, List[Dict]],
    debug: bool = False,
) -> List[Dict]:
    """Process a single pass and return match suggestions.
    
    Args:
        search_cache: Shared cache mapping ngram -> list of suggestions.
                      Avoids redundant searches across passes.
    
    Returns list of match records with multiple player suggestions.
    """
    matches = []
    seen_ngrams: Set[str] = set()
    cache_hits = 0
    
    ngrams = build_ngrams(tokens, min_gram, max_n=max_gram)
    
    # Sort by n-gram length descending (prefer longer matches)
    ngrams.sort(key=lambda x: len(x[0].split()), reverse=True)
    
    for chunk, start_idx, end_idx, token_slice in ngrams:
        if chunk in seen_ngrams:
            continue
        
        # Check cache first
        if chunk in search_cache:
            suggestions = search_cache[chunk]
            cache_hits += 1
            if debug and suggestions:
                print(f"[debug] Pass {pass_num}: '{chunk}' -> CACHE HIT ({len(suggestions)} suggestions)")
        else:
            suggestions = []
            
            # 1. Exact match
            exact_players = players_by_name.get(chunk, [])
            for player in exact_players[:max_suggestions]:
                name = player.get("name") or player.get("full_name")
                career_score = player.get("_career_score") or 0.0
                suggestions.append({
                    "name": name,
                    "match_type": "exact",
                    "score": 100,
                    "career_score": career_score,
                    "player": {
                        "name": name,
                        "full_name": player.get("full_name"),
                        "nationality": player.get("nationality"),
                        "position": player.get("position"),
                        "current_club": player.get("current_club") or player.get("club"),
                        "career_score": career_score,
                    }
                })
            
            # 2. Fuzzy match (if no exact or want more suggestions)
            if len(suggestions) < max_suggestions:
                fuzzy_matches = fuzzy_match(chunk, all_names, limit=max_suggestions * 2, threshold=fuzzy_threshold)
                for matched_name, score in fuzzy_matches:
                    # Skip if already suggested via exact match
                    if any(s["name"].lower() == matched_name.lower() for s in suggestions):
                        continue
                    
                    players = players_by_name.get(matched_name, [])
                    for player in players[:2]:  # Top 2 players per fuzzy match
                        name = player.get("name") or player.get("full_name")
                        if any(s["name"].lower() == name.lower() for s in suggestions):
                            continue
                        career_score = player.get("_career_score") or 0.0
                        suggestions.append({
                            "name": name,
                            "match_type": "fuzzy",
                            "score": score,
                            "career_score": career_score,
                            "player": {
                                "name": name,
                                "full_name": player.get("full_name"),
                                "nationality": player.get("nationality"),
                                "position": player.get("position"),
                                "current_club": player.get("current_club") or player.get("club"),
                                "career_score": career_score,
                            }
                        })
                        if len(suggestions) >= max_suggestions:
                            break
                    if len(suggestions) >= max_suggestions:
                        break
            
            # Store in cache (even if empty, to avoid re-searching)
            search_cache[chunk] = suggestions
        
        if suggestions:
            seen_ngrams.add(chunk)
            
            # Sort suggestions by match score then career score
            suggestions.sort(key=lambda s: (s["score"] or 0, s["career_score"] or 0.0), reverse=True)
            suggestions = suggestions[:max_suggestions]
            
            match_record = {
                "pass": pass_num,
                "ngram": chunk,
                "token_indices": [start_idx, end_idx],
                "segment_start": token_slice[0].get("segment_start", 0),
                "segment_end": token_slice[-1].get("segment_end", 0),
                "avg_probability": sum(t.get("probability", 0) for t in token_slice) / len(token_slice),
                "suggestions": suggestions,
            }
            matches.append(match_record)
            
            if debug and chunk not in search_cache:
                top_names = [f"{s['name']} ({s['career_score']:.1f})" for s in suggestions[:3]]
                print(f"[debug] Pass {pass_num}: '{chunk}' -> {top_names}")
    
    if debug:
        print(f"[debug] Pass {pass_num}: {cache_hits} cache hits")
    
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Match ASR tokens to player names with fuzzy matching")
    parser.add_argument("tokens_csv", help="Stage-1 tokens CSV file")
    parser.add_argument(
        "--players",
        default="data/players_enriched.jsonl",
        help="JSONL of player data with fame_score",
    )
    parser.add_argument("--min-gram", type=int, default=1, help="Minimum n-gram size (default: 1)")
    parser.add_argument("--max-gram", type=int, default=3, help="Maximum n-gram size (default: 3)")
    parser.add_argument("--fuzzy-threshold", type=int, default=70, help="Fuzzy match threshold 0-100 (default: 70)")
    parser.add_argument("--max-suggestions", type=int, default=5, help="Max player suggestions per match (default: 5)")
    parser.add_argument("--output", help="Write match results to a JSON file")
    parser.add_argument("--players-output", help="Write unique player candidates to JSONL for stage3")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    # Load tokens grouped by pass
    passes = load_tokens_csv(Path(args.tokens_csv))
    if not passes:
        raise SystemExit("No tokens found in CSV.")
    
    if args.debug:
        print(f"[debug] Loaded {len(passes)} passes from {args.tokens_csv}")
        for pass_num, tokens in passes.items():
            print(f"[debug]   Pass {pass_num}: {len(tokens)} tokens")
    
    # Load player database
    players_by_name, all_names = load_players(Path(args.players))
    if args.debug:
        print(f"[debug] Loaded {len(all_names)} unique player name variants")
    
    # Shared search cache across all passes
    search_cache: Dict[str, List[Dict]] = {}
    
    # Process each pass independently
    all_matches = []
    for pass_num in sorted(passes.keys()):
        tokens = passes[pass_num]
        matches = process_pass(
            pass_num=pass_num,
            tokens=tokens,
            players_by_name=players_by_name,
            all_names=all_names,
            min_gram=args.min_gram,
            max_gram=args.max_gram,
            fuzzy_threshold=args.fuzzy_threshold,
            max_suggestions=args.max_suggestions,
            search_cache=search_cache,
            debug=args.debug,
        )
        all_matches.extend(matches)
    
    if args.debug:
        print(f"[debug] Search cache size: {len(search_cache)} entries")
    
    # Collect unique player candidates across all matches
    unique_players: Dict[str, Dict] = {}
    for match in all_matches:
        for suggestion in match.get("suggestions", []):
            player = suggestion.get("player", {})
            name = player.get("name", "")
            if not name:
                continue
            key = name.lower()
            if key not in unique_players:
                unique_players[key] = player
            else:
                # Update with higher career score if available
                if player.get("career_score", 0) > unique_players[key].get("career_score", 0):
                    unique_players[key] = player
    
    # Build output payload
    payload = {
        "source_csv": str(args.tokens_csv),
        "num_passes": len(passes),
        "total_matches": len(all_matches),
        "unique_players": len(unique_players),
        "matches": all_matches,
    }
    
    # Write main output
    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
        if args.debug:
            print(f"[debug] Wrote {len(all_matches)} matches to {args.output}")
    else:
        print(output)
    
    # Write unique players JSONL for stage3
    if args.players_output:
        players_list = sorted(unique_players.values(), key=lambda p: p.get("career_score", 0), reverse=True)
        with open(args.players_output, "w", encoding="utf-8") as f:
            for player in players_list:
                f.write(json.dumps(player, ensure_ascii=False) + "\n")
        if args.debug:
            print(f"[debug] Wrote {len(players_list)} unique player candidates to {args.players_output}")
    
    # Print summary
    if args.debug:
        print(f"\n[summary]")
        print(f"  Passes processed: {len(passes)}")
        print(f"  Total matches: {len(all_matches)}")
        print(f"  Unique players: {len(unique_players)}")
        
        # Show top players by mention count with career scores
        mention_counts: Dict[str, Tuple[int, float]] = {}
        for match in all_matches:
            for suggestion in match.get("suggestions", []):
                name = suggestion.get("name", "")
                career = suggestion.get("career_score") or 0.0
                if name:
                    if name not in mention_counts:
                        mention_counts[name] = (0, career)
                    mention_counts[name] = (mention_counts[name][0] + 1, max(mention_counts[name][1], career))
        
        top_mentioned = sorted(mention_counts.items(), key=lambda x: -x[1][0])[:15]
        print(f"  Top mentioned players:")
        for name, (count, career) in top_mentioned:
            print(f"    {name} (career: {career:.1f}): {count} mentions")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
