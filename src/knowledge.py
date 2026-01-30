"""Knowledge base utilities for footballer data."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class Player:
    name: str
    nationality: Optional[str] = None
    clubs: List[str] = field(default_factory=list)
    leagues: List[str] = field(default_factory=list)
    world_cup_years: List[int] = field(default_factory=list)
    club_history: List[dict] = field(default_factory=list)
    honors: List[dict] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(name: str, payload: dict) -> "Player":
        return Player(
            name=name,
            nationality=payload.get("nationality"),
            clubs=list(payload.get("clubs", [])),
            leagues=list(payload.get("leagues", [])),
            world_cup_years=list(payload.get("world_cup_years", [])),
            club_history=list(payload.get("club_history", [])),
            honors=list(payload.get("honors", [])),
            aliases=list(payload.get("aliases", [])),
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "nationality": self.nationality,
            "clubs": self.clubs,
            "leagues": self.leagues,
            "world_cup_years": self.world_cup_years,
            "club_history": self.club_history,
            "honors": self.honors,
            "aliases": self.aliases,
        }


class KnowledgeBase:
    def __init__(self, players: Iterable[Player]) -> None:
        self.players = list(players)
        self._name_index: Dict[str, Player] = {}
        for player in self.players:
            self._name_index[player.name.lower()] = player
            for alias in player.aliases:
                self._name_index[alias.lower()] = player

    def get(self, name: str) -> Optional[Player]:
        return self._name_index.get(name.lower())

    def all_names(self) -> List[str]:
        names = []
        for player in self.players:
            names.append(player.name)
            names.extend(player.aliases)
        return names

    @staticmethod
    def load(path: Path) -> "KnowledgeBase":
        if not path.exists():
            return KnowledgeBase([])

        if path.suffix == ".jsonl":
            players = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                name = payload.pop("name")
                players.append(Player.from_dict(name, payload))
            return KnowledgeBase(players)

        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            players = []
            for row in payload:
                name = row.pop("name")
                players.append(Player.from_dict(name, row))
            return KnowledgeBase(players)

        if isinstance(payload, dict):
            players = [Player.from_dict(name, attrs) for name, attrs in payload.items()]
            return KnowledgeBase(players)

        raise ValueError("Unsupported knowledge format.")
