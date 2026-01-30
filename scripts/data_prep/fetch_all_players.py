#!/usr/bin/env python3
"""
Master script to fetch and combine footballer names from multiple sources.
This creates a combined, deduplicated database that can be refreshed periodically.

Sources:
1. Kaggle Transfermarkt (davidcariboo/player-scores) - ~33k players, WEEKLY UPDATES
2. OpenFootball national teams - ~30k players (historical national team players)
3. FIFA Kaggle dataset - ~20k players per year (FIFA game data)

Usage:
    python fetch_all_players.py [--output FILE] [--sources SOURCE1,SOURCE2,...]

To keep up-to-date, run periodically (e.g., weekly cron job):
    0 0 * * 0 cd /path/to/project && python scripts/data_prep/fetch_all_players.py
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

SCRIPTS_DIR = Path(__file__).parent

def normalize_name(name: str) -> str:
    """Normalize a player name for deduplication."""
    import unicodedata
    # Normalize unicode
    name = unicodedata.normalize('NFKD', name)
    # Lowercase and strip
    name = name.lower().strip()
    # Remove common suffixes/prefixes
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' ii', ' iii']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name

def run_fetcher(script_name: str, output_file: str, extra_args: List[str] = None) -> bool:
    """Run a fetcher script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  Warning: {script_name} not found, skipping")
        return False
    
    cmd = [sys.executable, str(script_path), '--output', output_file]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"  Running {script_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:500]}")
        return False
    
    return True

def load_jsonl(filepath: str) -> List[Dict]:
    """Load players from a JSONL file."""
    players = []
    if not Path(filepath).exists():
        return players
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    players.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return players

def merge_players(all_players: List[Dict]) -> List[Dict]:
    """Merge and deduplicate players from multiple sources."""
    # Group by normalized name + nationality (if available)
    name_groups: Dict[str, List[Dict]] = defaultdict(list)
    
    for player in all_players:
        name = player.get('name', '') or player.get('full_name', '')
        if not name:
            continue
        
        # Use name + nationality + birth_year as key for better deduplication
        norm_name = normalize_name(name)
        nat_raw = player.get('nationality', '') or player.get('country', '') or ''
        nationality = str(nat_raw).lower().strip() if nat_raw else ''
        birth_year = player.get('birth_year', '')
        
        # Create a more specific key when we have more info
        if birth_year and nationality:
            key = f"{norm_name}|{nationality}|{birth_year}"
        elif nationality:
            key = f"{norm_name}|{nationality}"
        else:
            key = norm_name
        
        name_groups[key].append(player)
    
    # Merge each group
    merged = []
    for key, group in name_groups.items():
        if not group:
            continue
        
        # Take the record with highest fame score as base
        group.sort(key=lambda x: x.get('fame_score', 0), reverse=True)
        best = group[0].copy()
        
        # Collect all sources
        sources = set()
        for p in group:
            src = p.get('source', 'unknown')
            if isinstance(src, list):
                sources.update(src)
            else:
                sources.add(src)
        best['sources'] = list(sources)
        
        # Don't aggregate fame - just use the max
        # This prevents common names from getting inflated scores
        best['fame_score'] = max(p.get('fame_score', 0) for p in group)
        
        # Fill in missing fields from other records
        for p in group[1:]:
            for field, val in p.items():
                if field not in best or best[field] is None or best[field] == '':
                    best[field] = val
        
        merged.append(best)
    
    # Sort by fame score
    merged.sort(key=lambda x: x.get('fame_score', 0), reverse=True)
    
    return merged

def main():
    parser = argparse.ArgumentParser(
        description="Fetch and combine footballer names from all sources"
    )
    parser.add_argument(
        '--output', '-o',
        default='data/all_players.jsonl',
        help='Output JSONL file (default: data/all_players.jsonl)'
    )
    parser.add_argument(
        '--sources', '-s',
        default='transfermarkt,openfootball,fbref',
        help='Comma-separated list of sources (default: transfermarkt,openfootball,fbref)'
    )
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetching and just merge existing files'
    )
    
    args = parser.parse_args()
    sources = [s.strip() for s in args.sources.split(',')]
    
    temp_files = []
    
    # Fetch from each source
    if not args.skip_fetch:
        print("Fetching players from sources...")
        
        if 'transfermarkt' in sources:
            out = '/tmp/transfermarkt_temp.jsonl'
            if run_fetcher('fetch_transfermarkt_kaggle.py', out):
                temp_files.append(out)
        
        if 'openfootball' in sources:
            out = '/tmp/openfootball_temp.jsonl'
            if run_fetcher('fetch_openfootball.py', out):
                temp_files.append(out)
        
        if 'fifa' in sources:
            out = '/tmp/fifa_temp.jsonl'
            if run_fetcher('extract_fifa.py', out):
                temp_files.append(out)
        
        if 'fbref' in sources:
            out = '/tmp/fbref_temp.jsonl'
            if run_fetcher('fetch_fbref_2025.py', out):
                temp_files.append(out)
    else:
        # Use existing files in workspace
        for f in ['data/transfermarkt_players.jsonl', 'data/openfootball_players.jsonl', 
                  'data/fifa_all_players.jsonl', 'data/fbref_2025_players.jsonl']:
            if Path(f).exists():
                temp_files.append(f)
    
    # Load all players
    print("\nLoading and merging players...")
    all_players = []
    for filepath in temp_files:
        players = load_jsonl(filepath)
        print(f"  {Path(filepath).name}: {len(players)} players")
        all_players.extend(players)
    
    print(f"\nTotal before dedup: {len(all_players)}")
    
    # Merge and deduplicate
    merged = merge_players(all_players)
    
    print(f"Total after dedup: {len(merged)}")
    
    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        for player in merged:
            f.write(json.dumps(player, ensure_ascii=False, default=str) + '\n')
    
    print(f"\nWrote {len(merged)} unique players to {args.output}")
    
    # Print top players
    print("\nTop 20 players by combined fame score:")
    for i, p in enumerate(merged[:20], 1):
        name = p.get('name', 'Unknown')
        sources = ', '.join(p.get('sources', ['unknown']))
        fame = p.get('fame_score', 0)
        print(f"  {i:2}. {name} (sources: {sources}) - fame: {fame:.1f}")
    
    # Stats
    print(f"\nStatistics:")
    print(f"  Total unique players: {len(merged)}")
    source_counts = defaultdict(int)
    for p in merged:
        for s in p.get('sources', ['unknown']):
            source_counts[s] += 1
    for source, count in sorted(source_counts.items()):
        print(f"  From {source}: {count}")

if __name__ == '__main__':
    main()
