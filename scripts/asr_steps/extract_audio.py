#!/usr/bin/env python3
"""Extract audio from a video clip for ASR debugging."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_steps.common import extract_audio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--start", default="", help="Start timestamp (e.g., 1:35)")
    parser.add_argument("--end", default="", help="End timestamp (e.g., 2:01)")
    parser.add_argument("--output", required=True, help="Output WAV path")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    extract_audio(str(video_path), args.output, args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
