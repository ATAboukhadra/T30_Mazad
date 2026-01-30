"""Audio preprocessing utilities."""

from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def _build_atempo_chain(speed: float) -> str:
    if speed <= 0:
        raise ValueError("Speed must be positive.")

    # ffmpeg atempo supports 0.5 to 2.0 per filter; chain if needed.
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


def maybe_speed_adjust(audio_path: str, speed: Optional[float], workdir: Path) -> str:
    if speed is None or math.isclose(speed, 1.0, rel_tol=1e-4):
        return audio_path

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found; cannot adjust audio speed.")

    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / (Path(audio_path).stem + f"_speed{speed:.2f}.wav")
    filter_chain = _build_atempo_chain(speed)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-filter:a",
        filter_chain,
        "-ar",
        "16000",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(output_path)
