# toast-harness

The harness for the Mixedbread fast search agent. It ships the retrieval tool
surface over a Mixedbread store (`search_corpus`, `grep`, `get_chunks`,
`read_document`, `filter_chunks`, `prune_context`, `submit_ranking`), the agent loop
that drives those tools to a ranked list and/or an answer, and exact token
accounting so context budgeting and payload truncation measure what the model
actually sees. One import name: `agent_harness`.

## Install

```bash
pip install toast-harness
```

## Quickstart

Wired against any OpenAI-compatible served model. `examples/browsecomp/run.py` is
a complete client of this shape.

```python
import openai

import agent_harness
from agent_harness import apply_force_submit

# Optional: rollouts install the policy tokenizer themselves. Naming it here
# loads the checkpoint before the first query rather than inside it.
agent_harness.ensure_token_counter("/models/my-policy-model")

client = openai.OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="unused")


def generate(
    messages, *, tools, completion_config, force_submit=False, forced_tool_name="submit_ranking"
):
    # completion_config also carries harness-side policy (require_tool_calls,
    # model, num_retries) that is not a wire param -- translating it is the
    # caller's job, and apply_force_submit encodes the force-submit rule.
    config = dict(completion_config)
    if force_submit:
        apply_force_submit(config, forced_tool_name)
    return client.chat.completions.create(
        model="my-policy-model",
        messages=messages,
        tools=tools,
        tool_choice=config["tool_choice"],
        parallel_tool_calls=config["parallel_tool_calls"],
        temperature=0.7,
        top_p=0.95,
        max_tokens=4096,
    )


result = agent_harness.run_searcher(
    "Which contract governs the 2019 Acme distribution agreement?",
    store_identifiers=["my-mixedbread-store"],
    top_k=10,
    generation_fn=generate,
)

print(result["retrieval"]["ranked_ids"])  # chunk ids, best first
agent = result["openai"]["metadata"]["agent"]
print(agent["total_tokens"], agent["rounds_executed"])  # exact accounting
```

Both provider seams are injected; the harness bundles neither. `generation_fn`
receives the messages, the tool schemas, and the completion config, and returns a
chat-completion-shaped response. Retrieval resolves a Mixedbread SDK client from
`MXBAI_API_KEY` unless you pass a `client=` of your own -- anything structurally
matching `agent_harness.RetrievalClient`, the SDK client or an in-process
implementation alike.

## Answer modes

`answer_mode` picks how the episode ends. It is accepted by every entry point
(`run_searcher`, `fast_agentic_search`, and the `aio` coroutines).

```python
# 1. Plain ranking (default): the agent ends with a submit_ranking call.
result = agent_harness.run_searcher(query, store_identifiers=[store], generation_fn=generate)
result["retrieval"]["ranked_ids"]

# 2. Ranking plus answer: submit_ranking carries a required `answer` argument.
result = agent_harness.run_searcher(..., answer_mode="submit_ranking")
result["retrieval"]["ranked_ids"], result["answer"]

# 3. Answer only: no submit_ranking tool at all. The agent searches, then ends
#    the episode with a plain-text reply -- that text is the answer.
result = agent_harness.run_searcher(..., answer_mode="plain_text")
result["answer"]  # no ranking, so top_k / strict_top_k must stay unset
```

- `"none"` leaves prompts, tool schemas, and loop behavior byte-identical to the
  harness before the knob existed.
- `"submit_ranking"` makes a submission without an answer a validation error the
  agent gets a correction round for, forced submits included.
- `"plain_text"` withdraws `submit_ranking`; a turn with no tool calls ends the
  run. When the round budget runs out without an answer, the harness forces an
  answer turn the same way it forces a ranking.

Every record carries `answer_mode`; `answer` (top-level and under `retrieval`) is
present only when the mode produced one.

## Async

The loop is async-native; the sync API adapts sync seams onto it per call. Async
services consume the coroutines directly and inject async seams:

```python
from agent_harness import aio

result = await aio.run_fast_agentic_search(
    query,
    store_identifiers=[store],
    client=my_async_store_client,  # agent_harness.AsyncRetrievalClient
    generation_fn=my_async_generate,  # agent_harness.AsyncGenerationFn
    answer_mode="plain_text",
)

async for event in aio.stream_fast_agentic_search(...):  # progress events
    ...
```

Every event stream ends with exactly one terminal event (`RolloutCompleted`,
`RolloutFailed`, or `RolloutCancelled`). Per-rollout knobs travel as
`tuning=HarnessTuning(...)`, and `agent_harness.testing.verify_retrieval_client`
conformance-tests a retrieval binding by driving one real rollout against it.

## Token counting

Every rollout installs the policy tokenizer at its entry point -- from
`AGENT_HARNESS_TOKENIZER`, else the model name in `SEARCHER_AGENT_CONFIG` -- so
budgets count what the model will see. The counter used is recorded on the
record as `openai.metadata.token_counter_mode`; a rollout whose tokenizer will
not load fails unless `AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER=0` allows the
`chars/4` estimate.

## Running BrowseComp-Plus

`examples/browsecomp/` drives BrowseComp-Plus queries through the harness and
scores the resulting rankings. With a served model up:

```bash
export MXBAI_API_KEY=...                       # retrieval
export AGENT_HARNESS_TOKENIZER=/models/my-policy-model
export TOAST_MODEL_BASE_URL=http://127.0.0.1:30000/v1
export TOAST_ANSWER_MODE=none                  # or submit_ranking / plain_text
python examples/browsecomp/run.py out/ queries.json
```

`queries.json` is a list of `{"query_id", "query", "relevant_ids"}` rows.
`examples/browsecomp/scoring.py` scores a rollout's ranking:
`score(rollout["retrieval"], row["relevant_ids"])` returns nDCG@10 and recall@10.

## Configuration

| Variable | Effect |
| --- | --- |
| `MXBAI_API_KEY` | Mixedbread API key (required unless a `client` is injected). `MBREAD_API_KEY` and `MIXEDBREAD_API_KEY` are accepted aliases; pass `api_key_env=` to read some other variable. |
| `MXBAI_BASE_URL` | Point the store client at a non-default Mixedbread deployment (aliases `MBREAD_BASE_URL`, `MIXEDBREAD_BASE_URL`). |
| `AGENT_HARNESS_TOOL_CHOICE` | Override the wire `tool_choice` sent to the model. Per-rollout `HarnessTuning(tool_choice=...)` wins. |
| `KEEP_REASONING_HISTORY` | Round-trip `reasoning_content` back into the message history. Off by default. |
| `AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER` | On by default; set to `0` to allow the `chars/4` estimate when no tokenizer loads. |
| `AGENT_HARNESS_TOKENIZER` | Load this checkpoint as the token counter instead of the model name the rollout carries. |
| `AGENT_HARNESS_TOKEN_COUNTER_BACKEND` | `gigatoken` (default, parity-checked against HF) or `hf`. |
| `AGENT_HARNESS_CORPUS_BACKEND_TOP_K` | Candidates requested from the provider per `search_corpus` call; at least 5. Per-rollout `HarnessTuning(backend_top_k=...)` wins. |
| `AGENT_HARNESS_BRIDGE_TIMING` / `..._TIMING_FILE` | Write a per-phase timing trace. |

## Development

```bash
uv sync --all-extras
uv run pytest
uvx ruff@0.16.1 check . && uvx ruff@0.16.1 format --check .
```

CI runs the suite against the built wheel in a fresh venv, not against `src/`.

## License

Apache-2.0.
