#!/usr/bin/env python3
"""Run Whisper transcription with an optional initial prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_steps.common import safe_transcribe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to audio file (wav/mp3/etc)")
    parser.add_argument("--model", default="large", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Transcription language")
    parser.add_argument("--task", default="transcribe", help="Whisper task")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--initial-prompt", help="Prompt string to bias ASR")
    parser.add_argument("--initial-prompt-file", help="Read prompt from file")
    parser.add_argument("--output", help="Write raw Whisper JSON output to file")
    parser.add_argument("--print-transcript", action="store_true")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Audio not found: {audio_path}")

    try:
        import whisper  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Whisper not installed. Install `openai-whisper` and ffmpeg.") from exc

    prompt = args.initial_prompt
    if args.initial_prompt_file:
        prompt = Path(args.initial_prompt_file).read_text(encoding="utf-8").strip()

    model = whisper.load_model(args.model)

    beam_size = args.beam_size
    best_of = args.best_of
    if args.temperature == 0.0:
        best_of = None
    else:
        beam_size = None

    result = safe_transcribe(
        model,
        str(audio_path),
        language=args.language,
        task=args.task,
        word_timestamps=args.word_timestamps,
        initial_prompt=prompt,
        temperature=args.temperature,
        beam_size=beam_size,
        best_of=best_of,
    )

    payload = {
        "model": args.model,
        "language": args.language,
        "task": args.task,
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", []),
    }
    if args.print_transcript:
        print(payload["text"])
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
