#!/usr/bin/env python3
"""
Fetch footballer names from the Kaggle Transfermarkt dataset.
This dataset is REGULARLY UPDATED (weekly) with fresh data from Transfermarkt.

Source: https://www.kaggle.com/datasets/davidcariboo/player-scores
Last updated: Check Kaggle for latest date

Requirements:
    pip install kaggle pandas
    
    You need a Kaggle API key at ~/.kaggle/kaggle.json
    Get it from: https://www.kaggle.com/settings -> API -> Create New Token

Usage:
    python fetch_transfermarkt_kaggle.py [--output FILE] [--min-market-value VAL] [--limit N]
"""

import argparse
import json
import os
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
    
    # Kaggle downloads as .zip, unzip it
    zip_path = Path(output_dir) / f"{file}.zip"
    if zip_path.exists():
        subprocess.run(["unzip", "-o", str(zip_path), "-d", output_dir], 
                      capture_output=True)
        zip_path.unlink()
    
    return Path(output_dir) / file

def compute_fame_score(row: dict) -> float:
    """Compute fame score based on market value, appearances, and other factors."""
    score = 0.0
    
    # Use HIGHEST market value as main fame indicator
    # This captures peak fame even for retired/older players
    highest_value = row.get('highest_market_value_in_eur') or row.get('market_value_in_eur') or 0
    if highest_value:
        score += min(highest_value / 1_000_000, 200)  # Cap at 200 points for €200M+
    
    # Current market value bonus
    current_value = row.get('market_value_in_eur') or 0
    if current_value:
        score += min(current_value / 2_000_000, 100)  # Extra 100 points max
    
    # Contract expiry recency (more recent = still active)
    contract = str(row.get('contract_expiration_date', '') or '')
    if contract and contract >= '2025':
        score += 30  # Active player
    elif contract and contract >= '2020':
        score += 10
    
    # Date of birth gives us roughly their activity period
    dob = row.get('date_of_birth', '')
    if dob:
        try:
            birth_year = int(str(dob).split('-')[0])
            age = 2025 - birth_year
            if 20 <= age <= 38:
                score += 20  # Likely still active or recently retired
        except:
            pass
    
    return score

def fetch_players(output_file: str, min_market_value: int = 0, limit: int = None):
    """Fetch players from Kaggle Transfermarkt dataset."""
    import pandas as pd
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download players.csv
        players_file = download_from_kaggle(
            "davidcariboo/player-scores",
            "players.csv",
            tmpdir
        )
        
        print(f"Reading {players_file}...")
        df = pd.read_csv(players_file)
        
        print(f"Total players in dataset: {len(df)}")
        
        # Filter by market value if specified
        if min_market_value > 0:
            df = df[df['market_value_in_eur'].fillna(0) >= min_market_value]
            print(f"Players with market value >= €{min_market_value:,}: {len(df)}")
        
        # Compute fame scores
        players = []
        for _, row in df.iterrows():
            player_data = row.to_dict()
            
            # Extract name
            name = player_data.get('name', player_data.get('pretty_name', ''))
            if not name:
                continue
            
            # Build player record
            player = {
                'name': name,
                'full_name': player_data.get('pretty_name', name),
                'position': player_data.get('position', player_data.get('sub_position', '')),
                'nationality': player_data.get('country_of_citizenship', ''),
                'birth_year': None,
                'club': player_data.get('current_club_name', ''),
                'league': player_data.get('current_club_domestic_competition_id', ''),
                'market_value_eur': player_data.get('market_value_in_eur'),
                'highest_market_value_eur': player_data.get('highest_market_value_in_eur'),
                'foot': player_data.get('foot', ''),
                'height_cm': player_data.get('height_in_cm'),
                'agent': player_data.get('agent_name', ''),
                'contract_expires': player_data.get('contract_expiration_date', ''),
                'source': 'transfermarkt_kaggle',
                'transfermarkt_id': player_data.get('player_id'),
            }
            
            # Parse birth year
            dob = player_data.get('date_of_birth', '')
            if dob:
                try:
                    player['birth_year'] = int(str(dob).split('-')[0])
                except:
                    pass
            
            # Compute fame score
            player['fame_score'] = compute_fame_score(player_data)
            
            players.append(player)
        
        # Sort by fame score
        players.sort(key=lambda x: x['fame_score'], reverse=True)
        
        if limit:
            players = players[:limit]
        
        # Write output
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for player in players:
                f.write(json.dumps(player, ensure_ascii=False) + '\n')
        
        print(f"\nWrote {len(players)} players to {output_file}")
        
        # Print some stats
        if players:
            print("\nTop 10 players by fame score:")
            for i, p in enumerate(players[:10], 1):
                print(f"  {i}. {p['name']} ({p['club'] or 'N/A'}) - €{(p['market_value_eur'] or 0)/1e6:.1f}M")

def main():
    parser = argparse.ArgumentParser(
        description="Fetch footballer names from Kaggle Transfermarkt dataset"
    )
    parser.add_argument(
        '--output', '-o',
        default='data/transfermarkt_players.jsonl',
        help='Output JSONL file (default: data/transfermarkt_players.jsonl)'
    )
    parser.add_argument(
        '--min-market-value', '-m',
        type=int,
        default=0,
        help='Minimum market value in EUR (default: 0, all players)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of players (default: no limit)'
    )
    
    args = parser.parse_args()
    
    # Check for kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: kaggle CLI not found. Install with: pip install kaggle")
        print("And set up API key: https://www.kaggle.com/settings -> API -> Create New Token")
        sys.exit(1)
    
    fetch_players(args.output, args.min_market_value, args.limit)

if __name__ == '__main__':
    main()
