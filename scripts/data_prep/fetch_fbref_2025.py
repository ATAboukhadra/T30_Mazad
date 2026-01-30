#!/usr/bin/env python3
"""
Fetch fresh footballer names from the Kaggle FBref 2025-2026 dataset.
This dataset contains players from current season (updated frequently).

Source: https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2025-2026
Last updated: Check Kaggle for latest date (typically updated weekly/monthly)

Requirements:
    pip install kaggle pandas

Usage:
    python fetch_fbref_2025.py [--output FILE]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

def download_from_kaggle(dataset: str, file: str, output_dir: str) -> Path:
    """Download a specific file from a Kaggle dataset."""
    cmd = [
        "kaggle", "datasets", "download", dataset,
        "-f", file,
        "-p", output_dir,
        "--force"
    ]
    print(f"Downloading {file} from Kaggle...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        sys.exit(1)
    
    return Path(output_dir) / file

def parse_nationality(nation_str):
    """Parse nationality from format like 'us USA' or 'br BRA'."""
    if not nation_str or str(nation_str) == 'nan':
        return ''
    parts = str(nation_str).strip().split()
    if len(parts) >= 2:
        return parts[1]  # Return the country code like 'USA', 'BRA'
    return str(nation_str)

def fetch_players(output_file: str):
    """Fetch players from Kaggle FBref 2025-2026 dataset."""
    import pandas as pd
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download players data
        players_file = download_from_kaggle(
            "hubertsidorowicz/football-players-stats-2025-2026",
            "players_data-2025_2026.csv",
            tmpdir
        )
        
        print(f"Reading {players_file}...")
        df = pd.read_csv(players_file)
        
        print(f"Total players in dataset: {len(df)}")
        
        # Extract unique players
        players = []
        seen = set()
        
        for _, row in df.iterrows():
            name = row.get('Player', '')
            if not name or name in seen:
                continue
            seen.add(name)
            
            # Parse birth year
            birth_year = None
            born = row.get('Born')
            if pd.notna(born):
                try:
                    birth_year = int(float(born))
                except:
                    pass
            
            # Compute simple fame score based on minutes played and goals
            fame_score = 0.0
            minutes = row.get('Min', 0) or 0
            goals = row.get('Gls', 0) or 0
            assists = row.get('Ast', 0) or 0
            
            fame_score = float(minutes) / 100 + float(goals) * 5 + float(assists) * 3
            
            player = {
                'name': name,
                'position': row.get('Pos', ''),
                'nationality': parse_nationality(row.get('Nation', '')),
                'birth_year': birth_year,
                'club': row.get('Squad', ''),
                'league': row.get('Comp', ''),
                'minutes_played': int(minutes) if pd.notna(minutes) else None,
                'goals': int(goals) if pd.notna(goals) else None,
                'assists': int(assists) if pd.notna(assists) else None,
                'source': 'fbref_2025_2026',
                'fame_score': fame_score,
            }
            
            players.append(player)
        
        # Sort by fame score
        players.sort(key=lambda x: x['fame_score'], reverse=True)
        
        # Write output
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for player in players:
                f.write(json.dumps(player, ensure_ascii=False) + '\n')
        
        print(f"\nWrote {len(players)} players to {output_file}")
        
        # Print some stats
        if players:
            print("\nTop 20 players by playing time + goals:")
            for i, p in enumerate(players[:20], 1):
                print(f"  {i:2}. {p['name']} ({p['club']}) - {p['goals']}G {p['assists']}A, {p['minutes_played']}min")
            
            # Show youngest players
            young = [p for p in players if p['birth_year'] and p['birth_year'] >= 2006]
            young.sort(key=lambda x: -x['birth_year'])
            print(f"\nYoungest players (born 2006+): {len(young)}")
            for p in young[:10]:
                print(f"  - {p['name']} (born {p['birth_year']}, {p['club']})")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch footballer names from Kaggle FBref 2025-2026 dataset"
    )
    parser.add_argument(
        '--output', '-o',
        default='data/fbref_2025_players.jsonl',
        help='Output JSONL file (default: data/fbref_2025_players.jsonl)'
    )
    
    args = parser.parse_args()
    
    # Check for kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)
    
    fetch_players(args.output)

if __name__ == '__main__':
    main()
