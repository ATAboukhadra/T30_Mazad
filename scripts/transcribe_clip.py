"""Transcribe a clip from a video without name conditioning."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def build_atempo_chain(speed: float) -> str:
    if speed <= 0:
        raise ValueError("Speed must be positive.")
    filters = []
    remaining = speed
    while remaining < 0.5:
        filters.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        filters.append(2.0)
        remaining /= 2.0
    filters.append(remaining)
    return ",".join(f"atempo={f:.3f}" for f in filters)


def extract_audio(
    video_path: str,
    output_path: str,
    start: str,
    end: str,
    slowdown: float,
) -> None:
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if start:
        cmd.extend(["-ss", str(parse_timestamp(start))])
    if end:
        start_sec = parse_timestamp(start) if start else 0
        end_sec = parse_timestamp(end)
        duration = max(0.0, end_sec - start_sec)
        cmd.extend(["-t", str(duration)])
    if slowdown != 1.0:
        cmd.extend(["-af", build_atempo_chain(slowdown)])
    cmd.extend(["-ar", "16000", "-ac", "1", output_path])
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--start", required=True, help="Start timestamp (e.g., 1:35)")
    parser.add_argument("--end", required=True, help="End timestamp (e.g., 2:01)")
    parser.add_argument("--model", default="large", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Transcription language")
    parser.add_argument("--slowdown", type=float, default=1.0, help="Audio speed (0.5 = half speed)")
    parser.add_argument("--output", help="Write transcript to a text file")
    parser.add_argument("--probs-output", help="Write segment probabilities JSON")
    args = parser.parse_args()

    if not Path(args.video).exists():
        raise SystemExit(f"Video not found: {args.video}")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = str(Path(tmpdir) / "clip.wav")
        extract_audio(args.video, audio_path, args.start, args.end, args.slowdown)
        import whisper

        model = whisper.load_model(args.model)
        result = model.transcribe(audio_path, language=args.language)
        transcript = result.get("text", "").strip()
        if args.output:
            Path(args.output).write_text(transcript + "\n", encoding="utf-8")
        else:
            print(transcript)
        if args.probs_output:
            tokenizer = whisper.tokenizer.get_tokenizer(
                model.is_multilingual, language=args.language, task="transcribe"
            )
            segments = []
            for seg in result.get("segments", []):
                token_details = []
                token_ids = seg.get("tokens") or []
                for tid in token_ids:
                    token_details.append(
                        {
                            "id": tid,
                            "text": tokenizer.decode([tid]),
                            "confidence": seg.get("avg_logprob"),
                        }
                    )
                segments.append(
                    {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text"),
                        "avg_logprob": seg.get("avg_logprob"),
                        "no_speech_prob": seg.get("no_speech_prob"),
                        "compression_ratio": seg.get("compression_ratio"),
                        "temperature": seg.get("temperature"),
                        "tokens": token_details,
                    }
                )
            Path(args.probs_output).write_text(
                json.dumps({"segments": segments}, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
