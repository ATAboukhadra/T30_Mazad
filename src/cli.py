"""CLI for the audio -> names -> question condition pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr import WhisperTranscriber, build_prompt_from_names
from audio import maybe_speed_adjust
from eval import LLMConditionChecker, RuleBasedConditionChecker
from knowledge import KnowledgeBase
from llm import NullLLMClient
from names import DictionaryNameExtractor
from pipeline import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("question", help="Constraint question to evaluate")
    parser.add_argument(
        "--knowledge",
        default="data/players_enriched.jsonl",
        help="Path to player knowledge JSON/JSONL",
    )
    parser.add_argument("--model", default="small", help="Whisper model size")
    parser.add_argument(
        "--checker",
        choices=("rule", "llm"),
        default="rule",
        help="Condition checker mode",
    )
    parser.add_argument(
        "--slowdown",
        type=float,
        default=1.0,
        help="Audio speed multiplier (0.5 = half speed)",
    )
    parser.add_argument(
        "--bias-names",
        action="store_true",
        help="Bias ASR with known player names prompt",
    )
    parser.add_argument(
        "--allow-last-name",
        action="store_true",
        help="Allow last-name-only matches",
    )
    args = parser.parse_args()

    knowledge = KnowledgeBase.load(Path(args.knowledge))
    known_names = knowledge.all_names()

    prompt = build_prompt_from_names(known_names) if args.bias_names else None
    transcriber = WhisperTranscriber(model_size=args.model, prompt=prompt)

    extractor = DictionaryNameExtractor(
        known_names=known_names,
        allow_last_name_only=args.allow_last_name,
    )

    if args.checker == "llm":
        checker = LLMConditionChecker(knowledge=knowledge, llm=NullLLMClient())
    else:
        checker = RuleBasedConditionChecker(knowledge=knowledge)

    pipeline = Pipeline(transcriber=transcriber, extractor=extractor, checker=checker)

    audio_path = maybe_speed_adjust(args.audio, args.slowdown, Path(".cache/audio"))
    result = pipeline.run(audio_path, args.question)

    payload = {
        "transcript": result.transcript,
        "names": result.names,
        "condition_ok": result.condition_ok,
        "condition_details": result.condition_details,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
