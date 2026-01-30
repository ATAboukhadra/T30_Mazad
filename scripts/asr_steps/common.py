"""Shared helpers for ASR step scripts."""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def safe_transcribe(model: Any, audio_path: str, **kwargs: Any) -> dict:
    """Transcribe audio using Whisper model, passing parameters directly."""
    debug = kwargs.pop('debug', False)
    
    if debug:
        print(f"[debug] safe_transcribe: passing kwargs={list(kwargs.keys())}")
        if 'language' in kwargs:
            print(f"[debug] safe_transcribe: language={kwargs['language']}")
        if 'task' in kwargs:
            print(f"[debug] safe_transcribe: task={kwargs['task']}")
    
    # Pass parameters directly to model.transcribe - it handles them via **kwargs internally
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    return model.transcribe(audio_path, **filtered)


def parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def extract_audio(video_path: str, output_path: str, start: str, end: str) -> None:
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if start:
        cmd.extend(["-ss", str(parse_timestamp(start))])
    if end:
        start_sec = parse_timestamp(start) if start else 0
        end_sec = parse_timestamp(end)
        duration = max(0.0, end_sec - start_sec)
        cmd.extend(["-t", str(duration)])
    cmd.extend(["-ar", "16000", "-ac", "1", output_path])
    subprocess.run(cmd, check=True, capture_output=True)


def load_player_database(path: Path) -> Dict[str, Dict[str, Any]]:
    players: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            player = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = player.get("name") or player.get("full_name")
        if name:
            players[str(name).lower()] = player
            parts = str(name).split()
            if len(parts) > 1:
                players[parts[-1].lower()] = player
    return players


def load_known_names(path: Path) -> List[str]:
    player_db = load_player_database(path)
    names = [p.get("name", "") for p in player_db.values() if p.get("name")]
    return list(set(names))


def load_knowledge(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
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

    payload = json.loads(path.read_text(encoding="utf-8"))
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


def extract_phrase(question: str, keyword: str) -> Optional[str]:
    idx = question.find(keyword)
    if idx == -1:
        return None
    tail = question[idx + len(keyword):]
    tail = tail.strip(" ?.!,:;")
    if not tail:
        return None
    return tail


def score_player(question: str, player: Dict[str, Any]) -> Tuple[int, float]:
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

    club_phrase = extract_phrase(q, "played for") or extract_phrase(q, "play for")
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


def select_prompt_names(
    question: Optional[str],
    knowledge_path: Optional[Path],
    fallback_db_path: Path,
    limit: int,
    last_names_only: bool,
) -> List[str]:
    candidates: List[Dict[str, Any]] = []
    if question and knowledge_path and knowledge_path.exists():
        candidates = load_knowledge(knowledge_path)
        scored = []
        for player in candidates:
            score, fame = score_player(question, player)
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

    names = load_known_names(fallback_db_path)
    if last_names_only:
        names = [n.split()[-1] for n in names if n.split()]
    return names[:limit]


def build_initial_prompt(names: Iterable[str]) -> Optional[str]:
    names = list(names)
    if not names:
        return None
    return "Football players mentioned: " + ", ".join(names)
