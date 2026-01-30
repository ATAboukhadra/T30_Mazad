#!/usr/bin/env bash
set -euo pipefail

mkdir -p out

python3 scripts/stage1_asr.py \
  --video videos/abdelaziz1.mp4 \
  --start 1:36.2 --end 2:01 \
  --model large --language en --task transcribe \
  --prompt-db data/players_enriched.jsonl \
  --prompt-limit 200 \
  --question "Active Defenders in the english premier league" \
  --knowledge data/players_enriched.jsonl \
  --debug --print-prompt \
  --temperature 0.4 --num-passes 3 \
  --prompt-output out/stage1_prompt.txt \
  --transcript-output out/stage1_transcript.txt \
  --tokens-csv out/tokens.csv \
  --tokens-output out/stage1_tokens.txt

python3 scripts/stage2_match_names.py \
  out/tokens.csv \
  --players data/players_enriched.jsonl \
  --min-gram 1 --max-gram 3 \
  --fuzzy-threshold 70 \
  --max-suggestions 5 \
  --output out/stage2_matches.json \
  --players-output out/stage2_candidates.jsonl \
  --debug

python3 scripts/verify_names.py \
  videos/abdelaziz1.mp4 \
  "Active Defenders in the english premier league" \
  --start 1:36.2 --end 2:01 \
  --whisper-model large \
  --language en \
  --asr whisper \
  --player-db data/players_enriched.jsonl \
  --prompt-db data/players_enriched.jsonl \
  --prompt-limit 1000 \
  --question-filter \
  --knowledge data/players_enriched.jsonl \
  --debug --print-prompt --question-filter \
  --prompt-output out/verify_prompt.txt \
  --transcript-output out/verify_transcript.txt \
  --tokens-output out/verify_tokens.txt
