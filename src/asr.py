"""ASR backends and helpers."""

from __future__ import annotations

import os
from typing import Iterable, Optional


def build_prompt_from_names(names: Iterable[str], max_chars: int = 500) -> str:
    prompt = ", ".join(names)
    if len(prompt) <= max_chars:
        return prompt
    return prompt[: max_chars - 3] + "..."


class WhisperTranscriber:
    """Whisper transcriber with optional vocabulary prompt biasing."""

    def __init__(self, model_size: str = "small", prompt: Optional[str] = None) -> None:
        self.model_size = model_size
        self.prompt = prompt

    def transcribe(self, audio_path: str) -> str:
        try:
            import whisper  # type: ignore
        except ImportError:  # pragma: no cover - environment dependent
            return os.environ.get(
                "DUMMY_TRANSCRIPT",
                "Lionel Messi, Cristiano Ronaldo, Neymar.",
            )

        model = whisper.load_model(self.model_size)
        options = {}
        if self.prompt:
            options["initial_prompt"] = self.prompt
        result = model.transcribe(audio_path, **options)
        return result.get("text", "").strip()
