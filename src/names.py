"""Name extraction and normalization utilities."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Dict, Tuple


_WORD_RE = re.compile(r"[A-Za-z\-']+")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def tokenize(text: str) -> List[str]:
    return [m.group(0) for m in _WORD_RE.finditer(text)]


def last_name(name: str) -> str:
    parts = name.split()
    return parts[-1] if parts else name


def best_fuzzy_match(candidate: str, choices: Iterable[str], cutoff: float) -> Tuple[str, float]:
    best = ("", 0.0)
    for choice in choices:
        score = SequenceMatcher(None, candidate, choice).ratio()
        if score > best[1]:
            best = (choice, score)
    if best[1] < cutoff:
        return "", best[1]
    return best


class DictionaryNameExtractor:
    """Extract names by matching against a dictionary of known players."""

    def __init__(
        self,
        known_names: Iterable[str],
        allow_last_name_only: bool = True,
        fuzzy_cutoff: float = 0.86,
    ) -> None:
        self.known = {normalize(n): n for n in known_names}
        self.allow_last_name_only = allow_last_name_only
        self.fuzzy_cutoff = fuzzy_cutoff
        self._last_name_map: Dict[str, str] = {}
        if allow_last_name_only:
            for name in known_names:
                self._last_name_map[normalize(last_name(name))] = name

    def extract(self, transcript: str) -> List[str]:
        words = tokenize(transcript)
        lower_words = [normalize(w) for w in words]
        candidates: List[str] = []

        # Try n-grams up to length 4 for full-name matches.
        for size in range(4, 0, -1):
            for i in range(0, len(lower_words) - size + 1):
                chunk = " ".join(lower_words[i : i + size]).strip()
                if not chunk:
                    continue
                if chunk in self.known:
                    candidates.append(self.known[chunk])
                elif self.allow_last_name_only and chunk in self._last_name_map:
                    candidates.append(self._last_name_map[chunk])

        if not candidates and transcript:
            normalized = normalize(transcript)
            match, score = best_fuzzy_match(normalized, self.known.keys(), self.fuzzy_cutoff)
            if match:
                candidates.append(self.known[match])

        seen = set()
        unique: List[str] = []
        for name in candidates:
            key = normalize(name)
            if key not in seen:
                unique.append(name)
                seen.add(key)
        return unique
