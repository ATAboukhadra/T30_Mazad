"""Stage 3: Ask an LLM about the question for each candidate."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_llm_client(path: str) -> Any:
    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise ValueError("LLM client must be in module:Class format")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()


def build_prompt(question: str, name: str) -> str:
    return (
        "Answer the question for the single player below. "
        "Return strict JSON: {\"answer\": true|false, \"justification\": \"...\"}. "
        "Include years/dates if relevant.\n"
        f"Question: {question}\n"
        f"Player: {name}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates_json", help="Stage-2 candidates JSON output")
    parser.add_argument("question", help="Constraint question to evaluate")
    parser.add_argument("--llm-client", help="Import path module:Class for an LLM client")
    parser.add_argument("--output", help="Write responses to a JSON file")
    args = parser.parse_args()

    payload = json.loads(Path(args.candidates_json).read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])

    llm = None
    if args.llm_client:
        llm = load_llm_client(args.llm_client)

    responses: List[Dict[str, Any]] = []
    for entry in candidates:
        name = entry.get("name")
        if not name:
            continue
        prompt = build_prompt(args.question, name)
        response: Optional[str] = None
        if llm is not None:
            response = llm.ask(prompt)
        responses.append(
            {
                "name": name,
                "prompt": prompt,
                "response": response,
            }
        )

    output_payload = {
        "question": args.question,
        "source_candidates": str(args.candidates_json),
        "responses": responses,
    }

    output = json.dumps(output_payload, ensure_ascii=True, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
