"""Stage 1: Whisper ASR -> tokens with timestamps and confidence."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _normalize_token_text(text: str) -> str:
    return text.replace("\n", " ")


def _timestamp_seconds(token_id: int, timestamp_begin: int) -> float:
    return (token_id - timestamp_begin) * 0.02


def _build_token_payload(segment: dict, tokenizer: Any) -> Dict[str, Any]:
    tokens: List[int] = segment.get("tokens") or []
    timestamp_begin = getattr(tokenizer, "timestamp_begin", None)

    token_entries: List[dict] = []
    timeframes: List[dict] = []

    current_start: Optional[float] = None
    pending_indices: List[int] = []

    def close_timeframe(end_time: Optional[float]) -> None:
        if current_start is None or not pending_indices:
            return
        for idx in pending_indices:
            token_entries[idx]["end"] = end_time
        tokens_text = "".join(token_entries[idx]["text"] for idx in pending_indices).strip()
        timeframes.append(
            {
                "start": current_start,
                "end": end_time,
                "text": tokens_text,
                "token_indices": pending_indices[:],
                "avg_logprob": segment.get("avg_logprob"),
            }
        )

    for tid in tokens:
        if timestamp_begin is not None and tid >= timestamp_begin:
            ts = _timestamp_seconds(tid, timestamp_begin)
            close_timeframe(ts)
            current_start = ts
            pending_indices = []
            token_entries.append(
                {
                    "id": tid,
                    "text": "",
                    "is_timestamp": True,
                    "timestamp": ts,
                }
            )
            continue

        text = _normalize_token_text(tokenizer.decode([tid])) if tokenizer else ""
        entry = {
            "id": tid,
            "text": text,
            "is_timestamp": False,
            "start": current_start if current_start is not None else segment.get("start"),
            "end": None,
            "confidence": segment.get("avg_logprob"),
        }
        token_entries.append(entry)
        pending_indices.append(len(token_entries) - 1)

    close_timeframe(segment.get("end"))

    return {
        "tokens": token_entries,
        "timeframes": timeframes,
    }


def _safe_transcribe(model: Any, audio_path: str, **kwargs: Any) -> dict:
    transcribe = model.transcribe
    params = inspect.signature(transcribe).parameters
    filtered = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return transcribe(audio_path, **filtered)

def _load_player_database(path: Path) -> Dict[str, Dict[str, Any]]:
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


def _load_known_names(path: Path) -> List[str]:
    player_db = _load_player_database(path)
    names = [p.get("name", "") for p in player_db.values() if p.get("name")]
    unique = list(set(names))
    return unique


def _load_knowledge(path: Path) -> List[Dict[str, Any]]:
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


def _extract_phrase(question: str, keyword: str) -> Optional[str]:
    idx = question.find(keyword)
    if idx == -1:
        return None
    tail = question[idx + len(keyword):]
    tail = tail.strip(" ?.!,:;")
    if not tail:
        return None
    return tail


def _score_player(question: str, player: Dict[str, Any]) -> Tuple[int, float]:
    q = question.lower()
    score = 0
    fame = float(player.get("fame_score") or 0.0)

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
        score += 2
    if any(l in q for l in leagues) or any(any(t in l for t in league_targets) for l in leagues):
        score += 2

    position = str(player.get("position") or "").lower()
    if position:
        if any(p in q for p in [position]):
            score += 1
    if any(k in q for k in ["goalkeeper", "keeper"]):
        if "goal" in position or position == "gk":
            score += 1
    if any(k in q for k in ["defender", "defence", "defense", "cb", "lb", "rb"]):
        if any(k in position for k in ["def", "cb", "lb", "rb"]):
            score += 1
    if any(k in q for k in ["midfielder", "midfield", "cm", "dm", "am"]):
        if any(k in position for k in ["mid"]):
            score += 1
    if any(k in q for k in ["forward", "striker", "winger", "attack"]):
        if any(k in position for k in ["for", "wing", "att", "st"]):
            score += 1

    return score, fame


def _select_prompt_names(
    question: Optional[str],
    knowledge_path: Optional[Path],
    fallback_db_path: Path,
    limit: int,
    last_names_only: bool,
) -> List[str]:
    candidates: List[Dict[str, Any]] = []
    if question and knowledge_path and knowledge_path.exists():
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

    names = _load_known_names(fallback_db_path)
    if last_names_only:
        names = [n.split()[-1] for n in names if n.split()]
    return names[:limit]

def _parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def _extract_audio(
    video_path: str,
    output_path: str,
    start: str,
    end: str,
) -> None:
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if start:
        cmd.extend(["-ss", str(_parse_timestamp(start))])
    if end:
        start_sec = _parse_timestamp(start) if start else 0
        end_sec = _parse_timestamp(end)
        duration = max(0.0, end_sec - start_sec)
        cmd.extend(["-t", str(duration)])
    cmd.extend(["-ar", "16000", "-ac", "1", output_path])
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", help="Path to audio file (wav/mp3/etc)")
    parser.add_argument("--video", help="Path to video file to extract audio from")
    parser.add_argument("--start", help="Start timestamp (e.g., 1:35)")
    parser.add_argument("--end", help="End timestamp (e.g., 2:01)")
    parser.add_argument("--model", default="large", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Transcription language")
    parser.add_argument("--task", default="transcribe", help="Whisper task")
    parser.add_argument("--word-timestamps", action="store_true", help="Enable word timestamps")
    parser.add_argument(
        "--player-db",
        default="data/all_players.jsonl",
        help="Player database JSONL for name biasing",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=1000,
        help="Max number of names to include in the ASR prompt",
    )
    parser.add_argument("--question", help="Question used to filter prompt candidates")
    parser.add_argument("--knowledge", default="knowledge.json", help="Knowledge base JSON/JSONL")
    parser.add_argument("--last-names-only", action="store_true", help="Prompt Whisper with last names only")
    parser.add_argument("--transcript-output", help="Write transcript text to a file")
    parser.add_argument("--probs-output", help="Write segment probabilities JSON")
    parser.add_argument("--tokens-output", help="Write transcript tokens to a text file")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    parser.add_argument("--prompt-output", help="Write Whisper initial prompt to a file")
    parser.add_argument("--print-prompt", action="store_true", help="Print Whisper initial prompt")
    parser.add_argument("--print-transcript", action="store_true", help="Print transcript text")
    parser.add_argument("--output", help="Write JSON output to a file")
    args = parser.parse_args()

    if not args.audio and not args.video:
        raise SystemExit("Provide either an audio path or --video.")

    audio_path: Optional[Path] = None
    tempdir: Optional[tempfile.TemporaryDirectory[str]] = None

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise SystemExit(f"Video not found: {video_path}")
        tempdir = tempfile.TemporaryDirectory()
        audio_path = Path(tempdir.name) / "clip.wav"
        _extract_audio(
            str(video_path),
            str(audio_path),
            args.start or "",
            args.end or "",
        )
    else:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise SystemExit(f"Audio not found: {audio_path}")

    try:
        import whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Whisper not installed. Install `openai-whisper` and ffmpeg.") from exc

    model = whisper.load_model(args.model)
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, language=args.language, task=args.task
    )

    initial_prompt = None
    db_path = Path(args.player_db)
    knowledge_path = Path(args.knowledge) if args.knowledge else None
    known_names: List[str] = []
    if db_path.exists():
        known_names = _select_prompt_names(
            args.question,
            knowledge_path,
            db_path,
            args.prompt_limit,
            args.last_names_only,
        )
        if known_names:
            initial_prompt = "Football players mentioned: " + ", ".join(known_names)
    if args.debug:
        print(f"[debug] video={args.video} audio={args.audio}")
        print(f"[debug] start={args.start} end={args.end}")
        print(f"[debug] model={args.model} language={args.language} task={args.task}")
        print(f"[debug] player_db={args.player_db} prompt_limit={args.prompt_limit}")
        print(f"[debug] known_names_count={len(known_names)}")
        print(f"[debug] prompt_head={known_names[:10]}")
        print(f"[debug] prompt_tail={known_names[-10:]}")
        if initial_prompt:
            print(f"[debug] initial_prompt_chars={len(initial_prompt)}")
            print(f"[debug] prompt_sample_count={len(known_names)}")
    if initial_prompt and args.print_prompt:
        print(initial_prompt)
    if initial_prompt and args.prompt_output:
        Path(args.prompt_output).write_text(initial_prompt + "\n", encoding="utf-8")

    result = _safe_transcribe(
        model,
        str(audio_path),
        language=args.language,
        task=args.task,
        word_timestamps=args.word_timestamps,
        initial_prompt=initial_prompt,
    )

    segments_payload = []
    for segment in result.get("segments", []):
        token_payload = _build_token_payload(segment, tokenizer)
        segments_payload.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
                "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
                "compression_ratio": segment.get("compression_ratio"),
                "temperature": segment.get("temperature"),
                "words": segment.get("words") or [],
                **token_payload,
            }
        )

    payload = {
        "model": args.model,
        "language": args.language,
        "task": args.task,
        "text": result.get("text", "").strip(),
        "segments": segments_payload,
    }
    if args.debug:
        print(f"[debug] transcript_head={payload['text'][:120]}")

    if args.transcript_output:
        Path(args.transcript_output).write_text(
            payload["text"] + "\n", encoding="utf-8"
        )
    if args.print_transcript:
        print(payload["text"])

    if args.tokens_output:
        tokens = re.findall(r"\w+", payload["text"], flags=re.UNICODE)
        Path(args.tokens_output).write_text("\n".join(tokens) + "\n", encoding="utf-8")

    if args.probs_output:
        segments = []
        for seg in result.get("segments", []):
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
            json.dumps({"segments": segments}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.output:
        output = json.dumps(payload, ensure_ascii=False, indent=2)
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    if tempdir is not None:
        tempdir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
