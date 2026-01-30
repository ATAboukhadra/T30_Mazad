"""Stage 1: Whisper ASR -> segment-level text output."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path
from typing import List, Optional

from asr_steps.common import (
    build_initial_prompt,
    extract_audio,
    safe_transcribe,
    select_prompt_names,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="?", help="Path to audio file (wav/mp3/etc)")
    parser.add_argument("--video", help="Path to video file to extract audio from")
    parser.add_argument("--start", help="Start timestamp (e.g., 1:35)")
    parser.add_argument("--end", help="End timestamp (e.g., 2:01)")
    parser.add_argument("--model", default="large", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Transcription language")
    parser.add_argument("--task", default="transcribe", help="Whisper task")
    parser.add_argument(
        "--player-db",
        default="data/players_enriched.jsonl",
        help="Player database JSONL for name biasing",
    )
    parser.add_argument(
        "--prompt-db",
        default="data/players_enriched.jsonl",
        help="Prompt database JSONL for ASR biasing (default: data/players_enriched.jsonl)",
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=1000,
        help="Max number of names to include in the ASR prompt",
    )
    parser.add_argument("--question", help="Question used to filter prompt candidates")
    parser.add_argument("--knowledge", default="data/players_enriched.jsonl", help="Knowledge base JSON/JSONL")
    parser.add_argument("--last-names-only", action="store_true", help="Prompt Whisper with last names only")
    parser.add_argument("--transcript-output", help="Write transcript text to a file")
    parser.add_argument("--probs-output", help="Write segment probabilities JSON")
    parser.add_argument("--tokens-output", help="Write transcript tokens to a text file")
    parser.add_argument("--tokens-csv", help="Write tokens to CSV with segment timing and confidence")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature for variation (default: 0.4)")
    parser.add_argument("--num-passes", type=int, default=1, help="Number of transcription passes to run (default: 1)")
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
        extract_audio(
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
    initial_prompt = None
    db_path = Path(args.prompt_db) if args.prompt_db else Path(args.player_db)
    knowledge_path = Path(args.knowledge) if args.knowledge else None
    known_names: List[str] = []
    if db_path.exists():
        known_names = select_prompt_names(
            args.question,
            knowledge_path,
            db_path,
            args.prompt_limit,
            args.last_names_only,
        )
        initial_prompt = build_initial_prompt(known_names)
    if args.debug:
        print(f"[debug] video={args.video} audio={args.audio}")
        print(f"[debug] start={args.start} end={args.end}")
        print(f"[debug] model={args.model} language={args.language} task={args.task}")
        print(f"[debug] player_db={args.player_db} prompt_db={args.prompt_db} prompt_limit={args.prompt_limit}")
        print(f"[debug] known_names_count={len(known_names)}")
        print(f"[debug] prompt_head={known_names[:10]}")
        print(f"[debug] prompt_tail={known_names[-10:]}")
        if initial_prompt:
            print(f"[debug] initial_prompt_chars={len(initial_prompt)}")
            print(f"[debug] prompt_sample_count={len(known_names)}")
        # Check for non-ASCII/Arabic characters in prompt
        if initial_prompt:
            non_ascii = [c for c in initial_prompt if ord(c) > 127]
            arabic_chars = [c for c in initial_prompt if '\u0600' <= c <= '\u06FF']
            print(f"[debug] prompt_non_ascii_count={len(non_ascii)}")
            print(f"[debug] prompt_arabic_chars_count={len(arabic_chars)}")
            if arabic_chars:
                print(f"[debug] prompt_arabic_sample={arabic_chars[:20]}")
    if initial_prompt and args.print_prompt:
        print(initial_prompt)
    if initial_prompt and args.prompt_output:
        Path(args.prompt_output).write_text(initial_prompt + "\n", encoding="utf-8")

    if args.debug:
        print(f"[debug] CALLING safe_transcribe with:")
        print(f"[debug]   audio_path={audio_path}")
        print(f"[debug]   language={args.language}")
        print(f"[debug]   task={args.task}")
        print(f"[debug]   temperature={args.temperature}")
        print(f"[debug]   num_passes={args.num_passes}")
        print(f"[debug]   initial_prompt_len={len(initial_prompt) if initial_prompt else 0}")

    # Run multiple passes to get alternative transcriptions
    all_results = []
    for pass_num in range(args.num_passes):
        result = safe_transcribe(
            model,
            str(audio_path),
            language=args.language,
            task=args.task,
            initial_prompt=initial_prompt,
            temperature=args.temperature,
            debug=args.debug if pass_num == 0 else False,
        )
        all_results.append(result)
        if args.debug:
            print(f"[debug] Pass {pass_num + 1}: {result.get('text', '')[:80]}...")

    # Use the first result as the primary one
    result = all_results[0]

    if args.debug:
        print(f"[debug] RESULT from Whisper:")
        print(f"[debug]   result_language={result.get('language')}")
        print(f"[debug]   result_text_len={len(result.get('text', ''))}")
        result_text = result.get('text', '')
        result_arabic = [c for c in result_text if '\u0600' <= c <= '\u06FF']
        print(f"[debug]   result_arabic_chars_count={len(result_arabic)}")
        if result_arabic:
            print(f"[debug]   result_arabic_sample={''.join(result_arabic[:50])}")

    segments_payload = []
    for segment in result.get("segments", []):
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

    if args.tokens_csv:
        # Extract tokens from each pass with segment-level timing/confidence
        import math
        tokens_data = []
        for pass_num, pass_result in enumerate(all_results):
            for seg in pass_result.get("segments", []):
                seg_text = seg.get("text", "")
                seg_tokens = re.findall(r"\w+", seg_text, flags=re.UNICODE)
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                avg_logprob = seg.get("avg_logprob", 0.0)
                # Convert logprob to probability
                prob = math.exp(avg_logprob) if avg_logprob else 0.0
                
                for token in seg_tokens:
                    tokens_data.append({
                        "pass": pass_num + 1,
                        "token": token,
                        "segment_start": seg_start,
                        "segment_end": seg_end,
                        "probability": round(prob, 4),
                        "avg_logprob": round(avg_logprob, 4) if avg_logprob else 0.0,
                        "no_speech_prob": round(seg.get("no_speech_prob", 0.0), 4),
                    })
        
        with open(args.tokens_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pass", "token", "segment_start", "segment_end", "probability", "avg_logprob", "no_speech_prob"])
            writer.writeheader()
            writer.writerows(tokens_data)
        
        if args.debug:
            print(f"[debug] Wrote {len(tokens_data)} tokens from {args.num_passes} passes to {args.tokens_csv}")
            # Show unique tokens across passes
            unique_tokens = set(t["token"] for t in tokens_data)
            print(f"[debug] Unique tokens across all passes: {len(unique_tokens)}")
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
