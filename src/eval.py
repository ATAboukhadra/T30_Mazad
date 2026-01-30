"""Question evaluation using a knowledge base and optional LLM."""

from __future__ import annotations

import json
from typing import Iterable, List, Sequence, Tuple

from knowledge import KnowledgeBase
from llm import LLMClient


class RuleBasedConditionChecker:
    """Rule-based condition checker using a local knowledge base."""

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge

    def check(self, names: Sequence[str], question: str) -> Tuple[bool, str]:
        if not names:
            return False, "No names extracted from transcript."

        q = question.lower().strip()
        if "all" not in q:
            return False, "Question must include an 'all' constraint."

        if "world cup" in q:
            for name in names:
                player = self.knowledge.get(name)
                if not player:
                    return False, f"Missing knowledge for {name}."
                if not player.world_cup_years:
                    return False, f"{name} never played in a World Cup."
            return True, "All players have World Cup appearances."

        if "treble" in q:
            club = None
            if "with" in q:
                club = q.split("with", 1)[1].strip("?.! ")
            for name in names:
                player = self.knowledge.get(name)
                if not player:
                    return False, f"Missing knowledge for {name}."
                matching = [
                    entry
                    for entry in player.honors
                    if entry.get("type") == "treble"
                    and (not club or entry.get("club", "").lower() == club.lower())
                ]
                if not matching:
                    if club:
                        return False, f"{name} did not win a treble with {club}."
                    return False, f"{name} did not win a treble."
            details = []
            for name in names:
                player = self.knowledge.get(name)
                for entry in player.honors:
                    if entry.get("type") == "treble":
                        if not club or entry.get("club", "").lower() == club.lower():
                            year = entry.get("season")
                            details.append(f"{name}: {year}")
                            break
            if club:
                return True, f"All players won a treble with {club}. " + "; ".join(details)
            return True, "All players won a treble. " + "; ".join(details)

        if "played for" in q or "club" in q:
            if "played for" in q:
                club = q.split("played for", 1)[1].strip()
            else:
                club = q.split("club", 1)[1].strip()
            club = club.strip("?.! ")
            details: List[str] = []
            for name in names:
                player = self.knowledge.get(name)
                if not player:
                    return False, f"Missing knowledge for {name}."
                clubs = [c.lower() for c in player.clubs]
                if club and club.lower() not in clubs:
                    return False, f"{name} did not play for {club}."
                if player.club_history:
                    for entry in player.club_history:
                        if entry.get("club", "").lower() == club.lower():
                            start = entry.get("from")
                            end = entry.get("to")
                            details.append(f"{name}: {start} to {end}")
                            break
            if details:
                return True, f"All players played for {club}. " + "; ".join(details)
            return True, f"All players played for {club}."

        if "national" in q or "are" in q:
            for name in names:
                player = self.knowledge.get(name)
                if not player:
                    return False, f"Missing knowledge for {name}."
                nationality = (player.nationality or "").lower()
                if nationality and nationality in q:
                    continue
                if "are" in q and nationality and nationality not in q:
                    return False, f"{name} is not {nationality}."
            return True, "All players match the nationality constraint."

        return False, "Rule-based checker does not support this question type."


class LLMConditionChecker:
    """LLM-backed condition checker using knowledge snippets as context."""

    def __init__(self, knowledge: KnowledgeBase, llm: LLMClient) -> None:
        self.knowledge = knowledge
        self.llm = llm

    def _build_context(self, names: Iterable[str]) -> List[dict]:
        context = []
        for name in names:
            player = self.knowledge.get(name)
            if not player:
                context.append({"name": name, "missing": True})
                continue
            context.append(player.as_dict())
        return context

    def check(self, names: Sequence[str], question: str) -> Tuple[bool, str]:
        if not names:
            return False, "No names extracted from transcript."

        context = self._build_context(names)
        prompt = (
            "You are given a question and a JSON knowledge base for the named players. "
            "Answer with strict JSON: {\"answer\": true|false, \"justification\": \"...\"}. "
            "Include years/dates in the justification if present in the knowledge.\n"
            f"Question: {question}\n"
            f"Players: {json.dumps(context, ensure_ascii=True)}"
        )

        raw = self.llm.ask(prompt)
        try:
            payload = json.loads(raw)
            answer = bool(payload.get("answer"))
            justification = str(payload.get("justification", ""))
            if not justification:
                justification = "LLM did not provide justification."
            return answer, justification
        except json.JSONDecodeError:
            return False, "LLM response was not valid JSON."
