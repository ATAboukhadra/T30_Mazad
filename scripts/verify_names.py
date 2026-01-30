#!/usr/bin/env python3
"""
Video-based footballer name verification pipeline.

Given a video file (with optional timestamps), a trivia question, and optionally
the player database, this script:
1. Extracts audio from the video (optionally clipping to timestamps)
2. Transcribes the audio using Whisper
3. Extracts footballer names from the transcript
4. Uses an LLM to verify if all names satisfy the question's conditions

Usage:
    python verify_names.py video.mp4 "Name 10 players who played for Real Madrid" \
        --start 0:30 --end 1:00 --llm openai

Requirements:
    pip install openai-whisper ffmpeg-python openai
    # or for local LLM: pip install ollama
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env vars


@dataclass
class VerificationResult:
    """Result of the name verification pipeline."""
    video_path: str
    question: str
    transcript: str
    extracted_names: List[str]
    verified_names: List[Dict]
    all_valid: bool
    invalid_names: List[str]
    llm_reasoning: str
    errors: List[str] = field(default_factory=list)
    asr_result: Optional[Dict] = None


def parse_timestamp(ts: str) -> float:
    """Parse timestamp like '1:30' or '90' into seconds."""
    if not ts:
        return 0.0
    parts = ts.split(':')
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def extract_audio(video_path: str, output_path: str, 
                  start: Optional[str] = None, end: Optional[str] = None,
                  slowdown: float = 1.0) -> str:
    """Extract audio from video, optionally clipping to timestamps."""
    cmd = ["ffmpeg", "-y", "-i", video_path]
    
    if start:
        cmd.extend(["-ss", str(parse_timestamp(start))])
    if end:
        start_sec = parse_timestamp(start) if start else 0
        end_sec = parse_timestamp(end)
        duration = end_sec - start_sec
        cmd.extend(["-t", str(duration)])
    
    # Audio filters
    filters = []
    if slowdown != 1.0:
        # Build atempo chain for slowdown
        speed = slowdown
        while speed < 0.5:
            filters.append("atempo=0.5")
            speed /= 0.5
        while speed > 2.0:
            filters.append("atempo=2.0")
            speed /= 2.0
        filters.append(f"atempo={speed:.3f}")
    
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    
    cmd.extend(["-ar", "16000", "-ac", "1", output_path])
    
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def transcribe_audio(audio_path: str, model_size: str = "medium",
                     known_names: Optional[List[str]] = None,
                     use_gemini_asr: bool = True,
                     language: Optional[str] = None,
                     word_timestamps: bool = False,
                     debug: bool = False,
                     prompt_output: Optional[str] = None,
                     print_prompt: bool = False) -> Tuple[str, Optional[Dict]]:
    """Transcribe audio using Whisper or Gemini.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (if using Whisper)
        known_names: List of known player names to bias transcription
        use_gemini_asr: If True, use Gemini for ASR with player-name conditioning
    """
    
    if use_gemini_asr:
        return _transcribe_with_gemini(audio_path, known_names), None
    
    # Fall back to Whisper
    try:
        import whisper
    except ImportError:
        raise RuntimeError("Install openai-whisper: pip install openai-whisper")
    
    model = whisper.load_model(model_size)
    if debug:
        print(f"[debug] whisper_model={model_size} language={language} word_timestamps={word_timestamps}")
    
    # Build prompt with known names to improve recognition
    prompt = None
    if known_names:
        # Use a sample of names to bias transcription
        if debug:
            print(f"[debug] known_names_count={len(known_names)}")
            print(f"[debug] known_names_head={known_names[:10]}")
            print(f"[debug] known_names_tail={known_names[-10:]}")
        sample = known_names[:200]  # Whisper has token limits
        prompt = "Football players mentioned: " + ", ".join(sample)
    if debug:
        print(f"[debug] prompt_sample_count={len(sample) if known_names else 0}")
        if prompt:
            print(f"[debug] initial_prompt_chars={len(prompt)}")
    if prompt and print_prompt:
        print(prompt)
    if prompt and prompt_output:
        Path(prompt_output).write_text(prompt + "\n", encoding="utf-8")
    
    result = model.transcribe(
        audio_path,
        initial_prompt=prompt,
        language=language,
        word_timestamps=word_timestamps,
    )
    return result.get("text", "").strip(), result


def _transcribe_with_gemini(audio_path: str, known_names: Optional[List[str]] = None) -> str:
    """Transcribe audio using Gemini with player name conditioning.
    
    This approach conditions the model to ONLY output recognized player names,
    not arbitrary text transcription.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("Install google-genai: pip install google-genai")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY env var for Gemini.")
    client = genai.Client(api_key=api_key)
    
    # Build a conditioning prompt with known names
    name_examples = ""
    if known_names:
        # Sample famous names for conditioning
        famous = [n for n in known_names if any(x in n.lower() for x in 
                  ['messi', 'ronaldo', 'neymar', 'mbappe', 'haaland', 'salah', 'kane'])]
        sample = (famous + known_names[:500])[:500]
        name_examples = "\n\nKnown player names for reference:\n" + ", ".join(sample[:100])
    
    prompt = f"""You are a football/soccer player name transcription expert.

Listen to this audio and extract ONLY the footballer names being spoken.
The audio contains someone quickly naming football players.

IMPORTANT:
- Output ONLY the player names you hear, one per line
- Use the standard spelling of each player's name
- If you're unsure about a name, make your best guess based on how it sounds
- Do NOT include any other text, commentary, or explanations
- Do NOT include words like "and", "um", "the", etc.
- Just the player names, nothing else
{name_examples}

Output format (one name per line):
Lionel Messi
Cristiano Ronaldo
Kylian Mbappé
..."""

    # Upload the audio file and generate response
    audio_file = client.files.upload(file=audio_path)
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, audio_file]
    )
    
    # Clean up uploaded file
    try:
        client.files.delete(name=audio_file.name)
    except:
        pass
    
    return response.text.strip()


def extract_names_from_transcript(
    transcript: str,
    known_names: Optional[List[str]] = None,
    from_gemini_asr: bool = True,
    last_name_only: bool = False,
    player_db: Optional[Dict[str, Dict]] = None,
) -> List[str]:
    """Extract footballer names from transcript.
    
    Args:
        transcript: The transcribed text
        known_names: List of known player names for matching
        from_gemini_asr: If True, transcript is already line-by-line names from Gemini
    """
    names = []
    
    # If from Gemini ASR, the transcript is already player names (one per line)
    if from_gemini_asr:
        for line in transcript.split('\n'):
            name = line.strip().strip('-•*').strip()
            if name and len(name) > 2 and not name.startswith('#'):
                # Clean up common artifacts
                name = re.sub(r'^\d+[\.\)]\s*', '', name)  # Remove numbering
                if name:
                    names.append(name)
        
        # Deduplicate
        seen = set()
        unique = []
        for name in names:
            key = name.lower()
            if key not in seen:
                unique.append(name)
                seen.add(key)
        return unique
    
    # Standard extraction for Whisper transcripts
    # Build lookup for known names
    known_lookup = {}
    last_name_map = {}
    if known_names:
        for name in known_names:
            known_lookup[name.lower()] = name
            parts = name.split()
            if len(parts) > 1:
                last = parts[-1].lower()
                if last_name_only:
                    last_name_map.setdefault(last, []).append(name)
                else:
                    known_lookup[last] = name

    # Prefer the most famous entry for a last name if available
    def pick_best(full_names: List[str]) -> str:
        if not player_db:
            return full_names[0]
        best = None
        best_score = -1.0
        for full in full_names:
            info = player_db.get(full.lower())
            score = 0.0
            if info:
                score = float(info.get("fame_score") or 0.0)
            if score > best_score:
                best_score = score
                best = full
        return best or full_names[0]
    
    # First, try to match known names
    transcript_lower = transcript.lower()
    if last_name_only and last_name_map:
        for key, full_names in last_name_map.items():
            if len(key) > 3 and key in transcript_lower:
                display_name = pick_best(full_names)
                if display_name not in names:
                    names.append(display_name)
    else:
        for key, display_name in known_lookup.items():
            if len(key) > 3 and key in transcript_lower:  # Avoid short matches
                if display_name not in names:
                    names.append(display_name)
    
    # Also extract capitalized word sequences as potential names
    tokens = transcript.replace("-", " ").split()
    buffer = []
    
    def flush():
        if buffer:
            candidate = " ".join(buffer).strip(",.?!;:")
            if candidate and len(candidate) > 2:
                if known_lookup or last_name_map:
                    if last_name_only and last_name_map:
                        key = candidate.split()[-1].lower()
                        if key in last_name_map:
                            match = pick_best(last_name_map[key])
                            if match not in names:
                                names.append(match)
                    else:
                        match = known_lookup.get(candidate.lower())
                        if match and match not in names:
                            names.append(match)
                elif candidate not in names:
                    names.append(candidate)
            buffer.clear()
    
    for token in tokens:
        clean = token.strip(",.?!;:()")
        if clean and clean[0].isupper():
            buffer.append(clean)
        else:
            flush()
    flush()
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for name in names:
        key = name.lower()
        if key not in seen:
            unique.append(name)
            seen.add(key)
    
    return unique


def build_name_mappings(
    transcript: str,
    names: List[str],
    player_db: Optional[Dict[str, Dict]] = None,
    last_name_only: bool = False,
) -> List[Dict]:
    """Map extracted names back to transcript tokens and player info."""
    mappings = []
    lower = transcript.lower()
    for name in names:
        full = name
        last = name.split()[-1] if name.split() else name
        matched_token = None
        idx = lower.find(full.lower())
        if idx != -1:
            matched_token = full
        elif last_name_only and last:
            idx = lower.find(last.lower())
            if idx != -1:
                matched_token = last
        info = player_db.get(full.lower()) if player_db else None
        snippet = ""
        if idx != -1:
            start = max(0, idx - 20)
            end = min(len(transcript), idx + len(matched_token or "") + 20)
            snippet = transcript[start:end]
        mappings.append(
            {
                "name": full,
                "matched_token": matched_token,
                "match_index": idx if idx != -1 else None,
                "snippet": snippet,
                "player": info,
            }
        )
    return mappings


def load_player_database(db_path: str) -> Dict[str, Dict]:
    """Load player database from JSONL file."""
    players = {}
    with open(db_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                player = json.loads(line)
                name = player.get('name', player.get('full_name', ''))
                if name:
                    players[name.lower()] = player
                    # Also index by last name
                    parts = name.split()
                    if len(parts) > 1:
                        players[parts[-1].lower()] = player
            except json.JSONDecodeError:
                continue
    return players


def _load_knowledge(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    if p.suffix == ".jsonl":
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = obj.get("name") or obj.get("full_name")
            if name:
                obj = dict(obj)
                obj["name"] = name
                rows.append(obj)
        return rows

    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = []
        for obj in payload:
            if not isinstance(obj, dict):
                continue
            name = obj.get("name") or obj.get("full_name")
            if name:
                row = dict(obj)
                row["name"] = name
                rows.append(row)
        return rows

    if isinstance(payload, dict):
        rows = []
        for name, attrs in payload.items():
            if not isinstance(attrs, dict):
                continue
            row = dict(attrs)
            row["name"] = name
            rows.append(row)
        return rows

    return []


def _extract_phrase(question: str, keyword: str) -> Optional[str]:
    idx = question.find(keyword)
    if idx == -1:
        return None
    tail = question[idx + len(keyword):]
    tail = tail.strip(" ?.!,:;")
    if not tail:
        return None
    return tail


def _score_player(question: str, player: Dict) -> Tuple[int, float]:
    q = question.lower()
    score = 0
    fame = float(player.get("fame_score") or 0.0)

    # Normalize position string once for reuse.
    position = str(player.get("position") or "").lower()

    nationality = str(player.get("nationality") or "").lower()
    if nationality and nationality in q:
        score += 3

    clubs = []
    for key in ("clubs", "club_history", "club", "current_club"):
        value = player.get(key)
        if isinstance(value, list):
            clubs.extend([str(v).lower() for v in value])
        elif isinstance(value, dict):
            club = value.get("club")
            if club:
                clubs.append(str(club).lower())
        elif isinstance(value, str):
            clubs.append(value.lower())

    club_phrase = _extract_phrase(q, "played for") or _extract_phrase(q, "play for")
    if club_phrase:
        if any(club_phrase in c for c in clubs):
            score += 4

    league = str(player.get("league") or "").lower()
    leagues = [str(l).lower() for l in player.get("leagues", []) if l]
    league_targets = []
    if "english premier league" in q or "premier league" in q:
        league_targets.extend(["english premier league", "eng premier league", "premier league", "gb1"])
    if league and (league in q or any(t in league for t in league_targets)):
        score += 3
    if any(l in q for l in leagues) or any(any(t in l for t in league_targets) for l in leagues):
        score += 3

    if position and position in q:
        score += 1
    if any(k in q for k in ["goalkeeper", "keeper"]):
        if "goal" in position or position == "gk":
            score += 2
    if any(k in q for k in ["defender", "defence", "defense", "cb", "lb", "rb", "fullback", "full-back"]):
        if any(k in position for k in ["def", "cb", "lb", "rb", "lwb", "rwb", "back"]):
            score += 5
        elif position:
            score -= 1
    if any(k in q for k in ["midfielder", "midfield", "cm", "dm", "am"]):
        if any(k in position for k in ["mid", "cm", "dm", "am"]):
            score += 3
        elif position:
            score -= 1
    if any(k in q for k in ["forward", "striker", "winger", "attack"]):
        if any(k in position for k in ["for", "wing", "att", "st"]):
            score += 3
        elif position:
            score -= 1

    return score, fame


def _select_prompt_names(
    question: Optional[str],
    knowledge_path: Optional[str],
    player_db: Optional[Dict[str, Dict]],
    limit: int,
    last_names_only: bool,
) -> List[str]:
    if question and knowledge_path:
        candidates = _load_knowledge(knowledge_path)
        scored = []
        for player in candidates:
            score, fame = _score_player(question, player)
            scored.append((score, fame, player))
        scored.sort(key=lambda x: (x[0], x[1], str(x[2].get("name", "")).lower()), reverse=True)
        filtered = [p for s, f, p in scored if s > 0]
        if filtered:
            names = [p.get("name") for p in filtered if p.get("name")]
        else:
            names = [p.get("name") for p in candidates if p.get("name")]
        if last_names_only:
            names = [n.split()[-1] for n in names if n.split()]
        return names[:limit]

    if player_db:
        names = [p.get("name", "") for p in player_db.values() if p.get("name")]
        unique = list(set(names))
        if last_names_only:
            unique = [n.split()[-1] for n in unique if n.split()]
        return unique[:limit]

    return []


def verify_with_llm(names: List[str], question: str, 
                    player_db: Optional[Dict[str, Dict]] = None,
                    llm_provider: str = "openai",
                    model: str = None) -> Tuple[bool, List[str], str]:
    """
    Use LLM to verify if names satisfy the question conditions.
    Returns: (all_valid, invalid_names, reasoning)
    """
    
    # Build context about players from database
    player_info = []
    for name in names:
        info = player_db.get(name.lower()) if player_db else None
        if info:
            details = f"- {name}: "
            if info.get('nationality'):
                details += f"nationality: {info['nationality']}, "
            if info.get('club'):
                details += f"current club: {info['club']}, "
            if info.get('position'):
                details += f"position: {info['position']}, "
            if info.get('birth_year'):
                details += f"born: {info['birth_year']}"
            player_info.append(details.rstrip(", "))
        else:
            player_info.append(f"- {name}: (no database info available)")
    
    player_context = "\n".join(player_info) if player_info else "No player information available."
    
    prompt = f"""You are a football/soccer expert. A user was asked this trivia question:

QUESTION: {question}

They named these players:
{chr(10).join(f'- {name}' for name in names)}

Player information from database:
{player_context}

TASK: Determine if ALL the named players satisfy the condition in the question.

Consider:
1. Does each player meet the criteria in the question?
2. Are there any players that clearly do NOT fit?
3. If you're unsure about a player, mention it.

Respond in this exact JSON format:
{{
    "all_valid": true/false,
    "invalid_names": ["list", "of", "invalid", "names"],
    "reasoning": "Brief explanation of your decision"
}}

Only output the JSON, nothing else."""

    if llm_provider == "gemini":
        return _verify_with_gemini(prompt, model or "gemini-2.0-flash")
    elif llm_provider == "openai":
        return _verify_with_openai(prompt, model or "gpt-4o-mini")
    elif llm_provider == "ollama":
        return _verify_with_ollama(prompt, model or "llama3.2")
    elif llm_provider == "anthropic":
        return _verify_with_anthropic(prompt, model or "claude-3-5-sonnet-20241022")
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")


def _verify_with_gemini(prompt: str, model: str) -> Tuple[bool, List[str], str]:
    """Verify using Google Gemini API."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("Install google-genai: pip install google-genai")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY env var for Gemini.")
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
        )
    )
    
    text = response.text.strip()
    return _parse_llm_response(text)


def _verify_with_openai(prompt: str, model: str) -> Tuple[bool, List[str], str]:
    """Verify using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Install openai: pip install openai")
    
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    
    text = response.choices[0].message.content.strip()
    return _parse_llm_response(text)


def _verify_with_ollama(prompt: str, model: str) -> Tuple[bool, List[str], str]:
    """Verify using local Ollama."""
    try:
        import ollama
    except ImportError:
        raise RuntimeError("Install ollama: pip install ollama")
    
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    
    text = response['message']['content'].strip()
    return _parse_llm_response(text)


def _verify_with_anthropic(prompt: str, model: str) -> Tuple[bool, List[str], str]:
    """Verify using Anthropic API."""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("Install anthropic: pip install anthropic")
    
    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    
    text = response.content[0].text.strip()
    return _parse_llm_response(text)


def _parse_llm_response(text: str) -> Tuple[bool, List[str], str]:
    """Parse LLM JSON response."""
    # Try to extract JSON from response
    try:
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text)
        
        all_valid = data.get('all_valid', False)
        invalid_names = data.get('invalid_names', [])
        reasoning = data.get('reasoning', 'No reasoning provided')
        
        return all_valid, invalid_names, reasoning
    except (json.JSONDecodeError, KeyError) as e:
        return False, [], f"Failed to parse LLM response: {text[:500]}"


def run_pipeline(video_path: str, question: str,
                 start: Optional[str] = None,
                 end: Optional[str] = None,
                 slowdown: float = 1.0,
                 whisper_model: str = "medium",
                 player_db_path: Optional[str] = None,
                 prompt_db_path: Optional[str] = None,
                 llm_provider: str = "gemini",
                 llm_model: Optional[str] = None,
                 use_gemini_asr: bool = True,
                 language: Optional[str] = None,
                 word_timestamps: bool = False,
                 last_name_only: bool = False,
                 debug: bool = False,
                 prompt_output: Optional[str] = None,
                 print_prompt: bool = False,
                 question_filter: bool = False,
                 knowledge_path: Optional[str] = None,
                 prompt_limit: int = 1000,
                 prompt_last_names: bool = False) -> VerificationResult:
    """Run the full verification pipeline."""
    
    errors = []
    
    # Load player database
    player_db = None
    known_names = None
    prompt_db = None
    prompt_names = None
    if player_db_path and Path(player_db_path).exists():
        print(f"Loading player database from {player_db_path}...")
        player_db = load_player_database(player_db_path)
        known_names = list(set(p.get('name', '') for p in player_db.values() if p.get('name')))
        print(f"  Loaded {len(player_db)} players")
        if debug:
            print(f"[debug] known_names_unique={len(known_names)}")
    if prompt_db_path and Path(prompt_db_path).exists():
        print(f"Loading prompt database from {prompt_db_path}...")
        prompt_db = load_player_database(prompt_db_path)
        if debug:
            print(f"[debug] prompt_db_players={len(prompt_db)}")
    if debug:
        print(f"[debug] prompt_db_path={prompt_db_path}")
    
    # Extract audio from video
    print(f"Extracting audio from {video_path}...")
    if debug:
        print(f"[debug] start={start} end={end} slowdown={slowdown}")
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = str(Path(tmpdir) / "audio.wav")
        asr_result = None
        try:
            extract_audio(video_path, audio_path, start, end, slowdown)
        except subprocess.CalledProcessError as e:
            errors.append(f"Failed to extract audio: {e}")
            return VerificationResult(
                video_path=video_path, question=question,
                transcript="", extracted_names=[], verified_names=[],
                all_valid=False, invalid_names=[], llm_reasoning="",
                errors=errors
            )
        
        # Transcribe audio
        asr_method = "Gemini" if use_gemini_asr else f"Whisper ({whisper_model})"
        print(f"Transcribing audio with {asr_method}...")
        try:
            prompt_names = known_names
            if prompt_db:
                prompt_names = list(set(p.get("name", "") for p in prompt_db.values() if p.get("name")))
            if use_gemini_asr is False and (question and question_filter):
                prompt_names = _select_prompt_names(
                    question,
                    knowledge_path,
                    prompt_db or player_db,
                    prompt_limit,
                    prompt_last_names,
                )
            if debug and use_gemini_asr is False:
                print(f"[debug] question_filter={question_filter} prompt_limit={prompt_limit} prompt_last_names={prompt_last_names}")
                print(f"[debug] prompt_names_count={len(prompt_names or [])}")
            transcript, asr_result = transcribe_audio(
                audio_path,
                whisper_model,
                prompt_names,
                use_gemini_asr,
                language,
                word_timestamps=word_timestamps,
                debug=debug,
                prompt_output=prompt_output,
                print_prompt=print_prompt,
            )
            print(f"  Transcript: {transcript[:200]}...")
        except Exception as e:
            errors.append(f"Transcription failed: {e}")
            return VerificationResult(
                video_path=video_path, question=question,
                transcript="", extracted_names=[], verified_names=[],
                all_valid=False, invalid_names=[], llm_reasoning="",
                errors=errors
            )
    
    # Extract names
    print("Extracting player names...")
    names = extract_names_from_transcript(
        transcript,
        known_names,
        from_gemini_asr=use_gemini_asr,
        last_name_only=last_name_only,
        player_db=player_db,
    )
    print(f"  Found {len(names)} names: {names}")
    
    if not names:
        return VerificationResult(
            video_path=video_path, question=question,
            transcript=transcript, extracted_names=[],
            verified_names=[], all_valid=False, invalid_names=[],
            llm_reasoning="No player names could be extracted from the transcript.",
            errors=errors,
            asr_result=asr_result,
        )
    
    # Verify with LLM
    print(f"Verifying names with LLM ({llm_provider})...")
    try:
        all_valid, invalid_names, reasoning = verify_with_llm(
            names, question, player_db, llm_provider, llm_model
        )
    except Exception as e:
        errors.append(f"LLM verification failed: {e}")
        all_valid = False
        invalid_names = []
        reasoning = str(e)
    
    # Build verified names info
    verified_names = []
    for name in names:
        info = player_db.get(name.lower()) if player_db else {}
        verified_names.append({
            "name": name,
            "valid": name not in invalid_names,
            "info": {
                "nationality": info.get('nationality', 'Unknown'),
                "club": info.get('club', 'Unknown'),
                "position": info.get('position', 'Unknown'),
            } if info else None
        })
    
    return VerificationResult(
        video_path=video_path,
        question=question,
        transcript=transcript,
        extracted_names=names,
        verified_names=verified_names,
        all_valid=all_valid,
        invalid_names=invalid_names,
        llm_reasoning=reasoning,
        errors=errors,
        asr_result=asr_result
    )


def main():
    parser = argparse.ArgumentParser(
        description="Verify footballer names from video against a trivia question"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("question", help="Trivia question to verify against")
    parser.add_argument("--start", "-s", help="Start timestamp (e.g., '0:30' or '30')")
    parser.add_argument("--end", "-e", help="End timestamp (e.g., '1:00' or '60')")
    parser.add_argument("--slowdown", type=float, default=1.0,
                       help="Audio slowdown factor (0.5 = half speed, default: 1.0)")
    parser.add_argument("--whisper-model", "-w", default="large",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: medium)")
    parser.add_argument("--language", default=None,
                       help="Force ASR language (e.g., en, ar). Default: auto-detect")
    parser.add_argument("--word-timestamps", action="store_true",
                       help="Enable word-level timestamps (Whisper only)")
    parser.add_argument("--last-name-only", action="store_true",
                       help="Match transcript tokens to last names only")
    parser.add_argument("--player-db", "-d", default="data/players_enriched.jsonl",
                       help="Path to player database JSONL (default: data/players_enriched.jsonl)")
    parser.add_argument("--prompt-db", default="data/players_enriched.jsonl",
                       help="Prompt database JSONL for ASR biasing (default: data/players_enriched.jsonl)")
    parser.add_argument("--llm", "-l", default="gemini",
                       choices=["openai", "ollama", "anthropic", "gemini"],
                       help="LLM provider (default: gemini)")
    parser.add_argument("--llm-model", "-m", 
                       help="Specific LLM model to use (default: provider's default)")
    parser.add_argument("--asr", default="whisper", choices=["whisper", "gemini"],
                       help="ASR provider (default: whisper)")
    parser.add_argument("--transcript-output", help="Write raw transcript to a text file")
    parser.add_argument("--mapping-output", help="Write transcript-to-name mapping JSON")
    parser.add_argument("--tokens-output", help="Write transcript tokens to a text file")
    parser.add_argument("--probs-output", help="Write ASR segment/token confidences JSON")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    parser.add_argument("--question-filter", action="store_true", help="Filter ASR prompt names by question")
    parser.add_argument("--knowledge", default="data/players_enriched.jsonl", help="Knowledge base JSON/JSONL")
    parser.add_argument("--prompt-limit", type=int, default=1000, help="Max names in ASR prompt")
    parser.add_argument("--prompt-last-names", action="store_true", help="Prompt with last names only")
    parser.add_argument("--prompt-output", help="Write Whisper initial prompt to a file")
    parser.add_argument("--print-prompt", action="store_true", help="Print Whisper initial prompt")
    parser.add_argument("--json-stdout", action="store_true",
                       help="Print JSON output to stdout")
    parser.add_argument("--output", "-o", help="Output JSON file (optional)")
    
    args = parser.parse_args()
    
    # Check video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    # Run pipeline
    result = run_pipeline(
        video_path=args.video,
        question=args.question,
        start=args.start,
        end=args.end,
        slowdown=args.slowdown,
        whisper_model=args.whisper_model,
        player_db_path=args.player_db,
        prompt_db_path=args.prompt_db,
        llm_provider=args.llm,
        llm_model=args.llm_model,
        use_gemini_asr=(args.asr == "gemini"),
        language=args.language,
        word_timestamps=args.word_timestamps,
        last_name_only=args.last_name_only,
        debug=args.debug,
        prompt_output=args.prompt_output,
        print_prompt=args.print_prompt,
        question_filter=args.question_filter,
        knowledge_path=args.knowledge,
        prompt_limit=args.prompt_limit,
        prompt_last_names=args.prompt_last_names,
    )

    if args.transcript_output:
        Path(args.transcript_output).write_text(
            result.transcript + "\n", encoding="utf-8"
        )
    
    # Output results
    name_mappings = build_name_mappings(
        result.transcript,
        result.extracted_names,
        player_db=load_player_database(args.player_db) if args.player_db else None,
        last_name_only=args.last_name_only,
    )

    output = {
        "video": result.video_path,
        "question": result.question,
        "transcript": result.transcript,
        "names_found": result.extracted_names,
        "names_count": len(result.extracted_names),
        "all_valid": result.all_valid,
        "invalid_names": result.invalid_names,
        "reasoning": result.llm_reasoning,
        "verified_names": result.verified_names,
        "name_mappings": name_mappings,
        "errors": result.errors,
    }
    
    print("\n" + "="*60)
    print("VERIFICATION RESULT")
    print("="*60)
    print(f"Question: {result.question}")
    print(f"Names found ({len(result.extracted_names)}): {', '.join(result.extracted_names)}")
    print(f"All valid: {'✅ YES' if result.all_valid else '❌ NO'}")
    if result.invalid_names:
        print(f"Invalid names: {', '.join(result.invalid_names)}")
    print(f"Reasoning: {result.llm_reasoning}")
    
    if result.errors:
        print(f"\nErrors: {result.errors}")
    
    if args.tokens_output:
        tokens = re.findall(r"\w+", result.transcript, flags=re.UNICODE)
        Path(args.tokens_output).write_text("\n".join(tokens) + "\n", encoding="utf-8")

    if args.probs_output:
        if result.asr_result:
            segments = []
            for seg in result.asr_result.get("segments", []):
                segments.append(
                    {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text"),
                        "avg_logprob": seg.get("avg_logprob"),
                        "no_speech_prob": seg.get("no_speech_prob"),
                        "compression_ratio": seg.get("compression_ratio"),
                        "temperature": seg.get("temperature"),
                        "words": seg.get("words", []),
                    }
                )
            Path(args.probs_output).write_text(
                json.dumps({"segments": segments}, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            Path(args.probs_output).write_text(
                json.dumps(
                    {
                        "error": "No ASR result available (Gemini ASR does not expose token probabilities)."
                    },
                    ensure_ascii=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to {args.output}")
    elif args.json_stdout:
        print("\nFull JSON output:")
        print(json.dumps(output, indent=2, ensure_ascii=False))

    if args.mapping_output:
        with open(args.mapping_output, "w", encoding="utf-8") as f:
            json.dump(name_mappings, f, indent=2, ensure_ascii=False)
        print(f"\nName mapping saved to {args.mapping_output}")
    
    return 0 if result.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
