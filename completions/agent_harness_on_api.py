"""Run the toast-harness search loop on toast-1 through the Mixedbread Completions API.

The harness drives the model over a Mixedbread store with its own search tools,
context management and answer modes. ``--answer-mode`` picks the ending: a
ranked list of chunks, a plain-text answer, or both.

    python agent_harness_on_api.py --store my-store --answer-mode submit_ranking "question"

``pip install toast-harness openai``; ``MXBAI_API_KEY`` comes from the
environment, a ``.env`` file or ``--api-key``. Letting the API run the search
is ``hosted_tools.py``; a loop with your own retrieval is ``own_harness.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from typing import Any

import agent_harness
from agent_harness import apply_force_submit

MODEL = "toast-1"
BASE_URL = os.environ.get("MXBAI_COMPLETIONS_BASE_URL", "https://api.mixedbread.com/v1")
SAMPLING = {"temperature": 0.7, "top_p": 0.95}


def build_generation_fn(client: Any) -> agent_harness.GenerationFn:
    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        config = dict(completion_config)
        if force_submit:
            apply_force_submit(config, forced_tool_name)
            # name the terminal tool so the final turn is the structured answer
            config["tool_choice"] = {"type": "function", "function": {"name": forced_tool_name}}
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,  # the harness's own search tools
            tool_choice=config["tool_choice"],
            parallel_tool_calls=config["parallel_tool_calls"],
            store=False,  # nothing continues this completion
            **SAMPLING,
        )

    return generation_fn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--store", required=True, help="the Mixedbread store to search")
    parser.add_argument(
        "--answer-mode", choices=("none", "submit_ranking", "plain_text"), default="none"
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", help="write the full rollout record here")
    parser.add_argument("--api-key", help="Mixedbread API key; defaults to MXBAI_API_KEY")
    args = parser.parse_args()

    with contextlib.suppress(ImportError):
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    api_key = args.api_key or os.environ.get("MXBAI_API_KEY")
    if not api_key:
        parser.error("pass --api-key or set MXBAI_API_KEY")
    if not os.environ.get("AGENT_HARNESS_TOKENIZER"):
        # the hosted model's tokenizer is not available locally
        os.environ["AGENT_HARNESS_TOKENIZER"] = "estimate"
    import openai  # noqa: PLC0415

    client = openai.OpenAI(base_url=BASE_URL, api_key=api_key)
    result = agent_harness.run_searcher(
        args.query,
        store_identifiers=[args.store],
        top_k=None if args.answer_mode == "plain_text" else args.top_k,
        generation_fn=build_generation_fn(client),
        answer_mode=args.answer_mode,
    )
    agent = result["openai"]["metadata"]["agent"]
    print("answer:", result.get("answer"))
    print("ranked:", result["retrieval"].get("ranked_ids"))
    print("rounds:", agent["rounds_executed"], "tokens:", agent["total_tokens"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2, ensure_ascii=False, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
