"""Drive BrowseComp-Plus queries through the harness against a served model.

This is deliberately the *whole* client: the harness wheel, `openai` as the
caller-owned generation dependency, and nothing else. What `main` does per
query is exactly what the README shows -- ensure_token_counter on the local
tokenizer directory, an OpenAI client pointed at an OpenAI-compatible server,
a generation_fn honoring completion_config/force_submit, and run_searcher.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import openai
from mixedbread import AsyncMixedbread
from scoring import score

import agent_harness
from agent_harness import apply_force_submit

BASE_URL = os.environ.get("TOAST_MODEL_BASE_URL", "http://127.0.0.1:30000/v1")
SERVED_MODEL = os.environ.get("TOAST_SERVED_MODEL", "policy")
STORE = os.environ.get("TOAST_STORE", "browsecomp-plus")
TEMPERATURE = float(os.environ.get("TOAST_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOAST_TOP_P", "0.95"))
# 0 omits the parameter; the engine accepts top_k only through extra_body.
SAMPLING_TOP_K = int(os.environ.get("TOAST_SAMPLING_TOP_K", "0"))
TOP_K = 10
SEED = 0
# How the episode ends: "none" (ranking only), "submit_ranking" (ranking plus a
# required answer), or "plain_text" (answer only; the harness then takes no top_k).
ANSWER_MODE = os.environ.get("TOAST_ANSWER_MODE", "none")

# The harness completion_config carries harness-side policy keys (model,
# require_tool_calls, num_retries, reasoning_effort) that are not wire params.
# Translating it is the caller's job -- that is the seam.
WIRE_KEYS = ("tool_choice", "parallel_tool_calls")


class MixedbreadReranker:
    def __init__(self, model: str):
        self.model = model

    async def rerank(self, request: agent_harness.RerankRequest) -> agent_harness.RerankResult:
        async with AsyncMixedbread(api_key=os.environ["MXBAI_API_KEY"]) as client:
            result = await client.rerank(
                model=self.model,
                query=request.query,
                input=request.documents,
                top_k=len(request.documents),
            )
        return agent_harness.RerankResult(
            scores=tuple(
                agent_harness.RerankScore(index=item.index, score=item.score)
                for item in result.data
            ),
            model=result.model,
            input_tokens=result.usage.prompt_tokens,
        )


def build_generation_fn(client: openai.OpenAI, record: list[dict[str, Any]]):
    def generation_fn(
        messages,
        *,
        tools,
        completion_config,
        force_submit=False,
        forced_tool_name="submit_ranking",
    ):
        config = dict(completion_config)
        if force_submit:
            # The harness's own force-submit policy: always require a tool call,
            # and pin the wire tool_choice to the named function only when the
            # caller is already forcing on the wire.
            apply_force_submit(config, forced_tool_name)
        wire = {key: config[key] for key in WIRE_KEYS if key in config}
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=SERVED_MODEL,
            messages=messages,
            tools=tools,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=4096,
            seed=SEED,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "preserve_thinking": False,
                },
                **({"top_k": SAMPLING_TOP_K} if SAMPLING_TOP_K > 0 else {}),
            },
            **wire,
        )
        usage = response.usage
        record.append(
            {
                "force_submit": force_submit,
                "forced_tool_name": forced_tool_name if force_submit else None,
                "tool_choice": wire.get("tool_choice"),
                "wall_s": time.perf_counter() - started,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        )
        return response

    return generation_fn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run BrowseComp-Plus queries against a served model."
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("queries", type=Path)
    parser.add_argument("--strict-top-k", action="store_true", help="Request ten ranked chunks.")
    parser.add_argument(
        "--rerank-model",
        metavar="MODEL",
        help="Rerank all observed evidence with this Mixedbread model. Enables strict top-k.",
    )
    args = parser.parse_args()
    strict_top_k = args.strict_top_k or args.rerank_model is not None
    if strict_top_k and ANSWER_MODE == "plain_text":
        parser.error("Strict top-k requires a ranking answer mode.")
    reranker = MixedbreadReranker(args.rerank_model) if args.rerank_model else None

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = json.loads(args.queries.read_text())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    tokenizer_dir = os.environ["AGENT_HARNESS_TOKENIZER"]
    # Budget and truncate against the policy model's real tokenizer.
    agent_harness.ensure_token_counter(tokenizer_dir)

    client = openai.OpenAI(base_url=BASE_URL, api_key="unused", timeout=1800.0, max_retries=0)

    summaries: list[dict[str, Any]] = []
    for item in queries:
        query_id = str(item["query_id"])
        calls: list[dict[str, Any]] = []
        started = time.time()
        error: str | None = None
        result: dict[str, Any] | None = None
        metrics: dict[str, Any] = {}
        try:
            result = agent_harness.run_searcher(
                item["query"],
                store_identifiers=[STORE],
                top_k=None if ANSWER_MODE == "plain_text" else TOP_K,
                strict_top_k=strict_top_k,
                **({"reranker": reranker} if reranker is not None else {}),
                generation_fn=build_generation_fn(client, calls),
                query_id=query_id,
                answer_mode=ANSWER_MODE,
            )
            if ANSWER_MODE != "plain_text" and item.get("relevant_ids") is not None:
                metrics = score(result["retrieval"], item["relevant_ids"])
        except Exception as exc:  # recorded per query, never swallowed
            error = traceback.format_exc()
            print(f"query {query_id} FAILED: {exc}", flush=True)
        wall_s = time.time() - started

        if result is not None:
            (out_dir / f"rollout-{query_id}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
            )
        agent = ((result or {}).get("openai", {}).get("metadata", {}) or {}).get("agent", {}) or {}
        retrieval = (result or {}).get("retrieval", {}) or {}
        summary = {
            "query_id": query_id,
            "error": error,
            "wall_s": wall_s,
            "ranked_ids": list(retrieval.get("ranked_ids") or []),
            **metrics,
            "ranking_strategy": retrieval.get("ranking_strategy"),
            "answer": retrieval.get("answer"),
            "rounds_executed": agent.get("rounds_executed"),
            "forced_ranking": agent.get("forced_ranking"),
            "total_tokens": agent.get("total_tokens"),
            "input_tokens": agent.get("input_tokens"),
            "output_tokens": agent.get("output_tokens"),
            "trace_counts": agent.get("trace_counts"),
            "provider_failure_count": agent.get("provider_failure_count"),
            "generation_calls": calls,
        }
        if result and "reranking" in retrieval:
            summary.update(reranking=retrieval["reranking"], submission=result["submission"])
        summaries.append(summary)
        if error is None:
            print(f"query {query_id} done in {wall_s:.1f}s", flush=True)

    (out_dir / "client_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True, default=str) + "\n"
    )
    return int(any(row["error"] is not None for row in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
