#!/usr/bin/env python3
"""Build an initial Whisper prompt from a player database."""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_steps.common import build_initial_prompt, select_prompt_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-db", default="data/players_enriched.jsonl")
    parser.add_argument("--question", default=None)
    parser.add_argument("--knowledge", default="data/players_enriched.jsonl")
    parser.add_argument("--prompt-limit", type=int, default=1000)
    parser.add_argument("--last-names-only", action="store_true")
    parser.add_argument("--output", help="Write prompt to file")
    parser.add_argument("--names-output", help="Write names list to file")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.prompt_db)
    knowledge_path = Path(args.knowledge) if args.knowledge else None
    names = select_prompt_names(
        args.question,
        knowledge_path,
        db_path,
        args.prompt_limit,
        args.last_names_only,
    )
    prompt = build_initial_prompt(names) or ""

    if args.debug:
        print(f"[debug] prompt_db={args.prompt_db} knowledge={args.knowledge}")
        print(f"[debug] prompt_limit={args.prompt_limit} last_names_only={args.last_names_only}")
        print(f"[debug] names_count={len(names)}")
        print(f"[debug] prompt_head={names[:10]}")
        print(f"[debug] prompt_tail={names[-10:]}")
        print(f"[debug] prompt_chars={len(prompt)}")

    if args.print_prompt:
        print(prompt)
    if args.output:
        Path(args.output).write_text(prompt + "\n", encoding="utf-8")
    if args.names_output:
        Path(args.names_output).write_text("\n".join(names) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
