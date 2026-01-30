"""LLM client interfaces for question reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    answer: bool
    justification: str


class LLMClient(Protocol):
    def ask(self, prompt: str) -> str:
        """Return LLM raw response text for a prompt."""


class NullLLMClient:
    def ask(self, prompt: str) -> str:
        raise RuntimeError(
            "No LLM configured. Provide an LLM client implementation."
        )
