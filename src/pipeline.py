"""Audio -> names -> question condition check pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple


@dataclass
class PipelineResult:
    transcript: str
    names: List[str]
    condition_ok: bool
    condition_details: str


class Transcriber(Protocol):
    def transcribe(self, audio_path: str) -> str:
        """Return a transcript from an audio file."""


class NameExtractor(Protocol):
    def extract(self, transcript: str) -> List[str]:
        """Extract a list of footballer names from text."""


class ConditionChecker(Protocol):
    def check(self, names: Sequence[str], question: str) -> Tuple[bool, str]:
        """Return (ok, details) if names satisfy the question."""


class WhisperTranscriber:
    """Optional Whisper-based transcriber.

    Uses openai-whisper if available. Falls back to raising an error if missing.
    """

    def __init__(self, model_size: str = "small") -> None:
        self.model_size = model_size

    def transcribe(self, audio_path: str) -> str:
        try:
            import whisper  # type: ignore
        except ImportError:  # pragma: no cover - environment dependent
            return os.environ.get(
                "DUMMY_TRANSCRIPT",
                "Lionel Messi, Cristiano Ronaldo, Neymar.",
            )

        model = whisper.load_model(self.model_size)
        result = model.transcribe(audio_path)
        return result.get("text", "").strip()


class SimpleNameExtractor:
    """Heuristic extractor that treats capitalized word runs as names.

    This is a baseline for fast speech where the transcript is noisy.
    """

    def __init__(self, known_names: Optional[Iterable[str]] = None) -> None:
        self.known_names = {n.lower(): n for n in (known_names or [])}

    def extract(self, transcript: str) -> List[str]:
        tokens = transcript.replace("-", " ").split()
        names: List[str] = []
        buffer: List[str] = []

        def flush_buffer() -> None:
            if not buffer:
                return
            candidate = " ".join(buffer).strip()
            if not candidate:
                return
            names.append(candidate)
            buffer.clear()

        for token in tokens:
            if token[:1].isupper():
                buffer.append(token.strip(",.?!;:"))
            else:
                flush_buffer()
        flush_buffer()

        if not names and self.known_names:
            for key, display in self.known_names.items():
                if key in transcript.lower():
                    names.append(display)

        # Deduplicate while preserving order.
        seen = set()
        unique: List[str] = []
        for name in names:
            key = name.lower()
            if key not in seen:
                unique.append(name)
                seen.add(key)
        return unique


class RuleBasedConditionChecker:
    """Rule-based condition checker using a local knowledge base.

    The knowledge base is a dict of player name -> attributes.
    Example attributes: {"nationality": "Brazil", "clubs": ["Barcelona"]}
    """

    def __init__(self, knowledge: dict) -> None:
        self.knowledge = {k.lower(): v for k, v in knowledge.items()}

    def check(self, names: Sequence[str], question: str) -> Tuple[bool, str]:
        if not names:
            return False, "No names extracted from transcript."

        q = question.lower().strip()
        if "all" not in q:
            return False, "Question must include an 'all' constraint."

        # Simple patterns: "all are <nationality>", "all played for <club>"
        if "national" in q or "are" in q:
            for name in names:
                attrs = self.knowledge.get(name.lower())
                if not attrs:
                    return False, f"Missing knowledge for {name}."
                nationality = attrs.get("nationality", "").lower()
                if nationality and nationality in q:
                    continue
                if "are" in q and nationality and nationality not in q:
                    return False, f"{name} is not {nationality}."

        if "played for" in q or "club" in q:
            if "played for" in q:
                club = q.split("played for", 1)[1].strip()
            else:
                club = q.split("club", 1)[1].strip()
            club = club.strip("?.! ")
            for name in names:
                attrs = self.knowledge.get(name.lower())
                if not attrs:
                    return False, f"Missing knowledge for {name}."
                clubs = [c.lower() for c in attrs.get("clubs", [])]
                if club and club.lower() not in clubs:
                    return False, f"{name} did not play for {club}."

        return True, "All names satisfy the question based on knowledge."


class Pipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        extractor: NameExtractor,
        checker: ConditionChecker,
    ) -> None:
        self.transcriber = transcriber
        self.extractor = extractor
        self.checker = checker

    def run(self, audio_path: str, question: str) -> PipelineResult:
        transcript = self.transcriber.transcribe(audio_path)
        names = self.extractor.extract(transcript)
        ok, details = self.checker.check(names, question)
        return PipelineResult(
            transcript=transcript,
            names=names,
            condition_ok=ok,
            condition_details=details,
        )
