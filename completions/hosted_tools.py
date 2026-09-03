"""Ask toast-1 over a Mixedbread store in one request: the API runs the search loop.

Declare the hosted store tools and the API executes them inside the completion:
the model searches and greps the store as often as it needs, and the request
returns when it has the answer. No tool call comes back to you and there is no
loop to write. ``include`` adds the chunks every hosted call retrieved to the
response, so the answer can be shown next to what the model read, and
``context_management`` lets the model prune results it no longer needs when a
long search approaches the context window; the response reports what was
cleared as ``context_management.applied_edits``.

    python hosted_tools.py --store my-store "Which robot vacuums run for at least 200 minutes?"

``openai`` is the only dependency; ``MXBAI_API_KEY`` comes from the environment,
a ``.env`` file or ``--api-key``. Bringing your own retrieval is
``own_harness.py``; a ranked, cited answer from the toast-harness loop is
``agent_harness_on_api.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from typing import Any

MODEL = "toast-1"
BASE_URL = os.environ.get("MXBAI_COMPLETIONS_BASE_URL", "https://api.mixedbread.com/v1")


def ask(client: Any, query: str, *, store: str) -> dict[str, Any]:
    """One request: the answer, every hosted call the API ran with its results, and usage."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": query}],
        tools=[
            {"type": "search_corpus", "store_identifiers": [store]},
            {"type": "grep", "store_identifiers": [store]},
        ],
        # extension fields, so they go through extra_body: return the retrieved
        # chunks, and let the model prune results it no longer needs when a
        # long search approaches the context window
        extra_body={
            "include": ["search_corpus_call.results", "grep_call.results"],
            "context_management": {"edits": [{"type": "prune_context"}]},
        },
        temperature=0.7,
        top_p=0.95,
        store=False,  # nothing continues this completion, so opt out of retention
    )
    return {
        "answer": response.choices[0].message.content or "",
        "hosted_calls": getattr(response, "hosted_tool_calls", None) or [],
        "context_edits": (getattr(response, "context_management", None) or {}).get("applied_edits")
        or [],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--store", required=True, help="the Mixedbread store to search")
    parser.add_argument("--out", help="write the record (answer, hosted calls, chunks) here")
    parser.add_argument("--api-key", help="Mixedbread API key; defaults to MXBAI_API_KEY")
    args = parser.parse_args()

    with contextlib.suppress(ImportError):
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    api_key = args.api_key or os.environ.get("MXBAI_API_KEY")
    if not api_key:
        parser.error("pass --api-key or set MXBAI_API_KEY")
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(base_url=BASE_URL, api_key=api_key)
    record = ask(client, args.query, store=args.store)
    print(record["answer"])
    print()
    for call in record["hosted_calls"]:
        chunks = call.get("results") or []
        searched = call.get("queries") or [call.get("pattern", "")]
        handles = ", ".join(f"{c['filename']}:{c['chunk_index']}" for c in chunks[:5])
        print(f"{call['type']:<19} {call['status']:<10} {'; '.join(searched)}")
        print(
            f"{'':<19} {len(chunks)} chunks{': ' if handles else ''}{handles}{', …' if len(chunks) > 5 else ''}"
        )
    for edit in record["context_edits"]:
        print(
            f"{edit['type']}: {edit.get('calls', 1)} calls cleared "
            f"{edit['cleared_input_tokens']} input tokens"
        )
    print("usage:", record["usage"])
    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            json.dump(record, out, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
