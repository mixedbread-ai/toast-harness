# toast-harness

The harness for the Mixedbread fast search agent. It ships the retrieval tool
surface over a Mixedbread store (`search_corpus`, `grep`, `get_chunks`,
`read_document`, `filter_chunks`, `prune_context`, `submit_ranking`), the agent loop
that drives those tools to a ranked answer, and exact token accounting so context
budgeting and payload truncation measure what the model actually sees rather than a
`chars/4` guess. One import name: `agent_harness`.

## Install

```bash
pip install toast-harness
```

## Quickstart

Wired against any OpenAI-compatible served model. `examples/browsecomp/run.py` is
a complete client of this shape, including the sampling parameters a real run
would set.

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

Both provider seams are injected; the harness bundles neither. Generation is a
required `generation_fn` — it receives the messages, the tool schemas, and the
completion config, and returns a chat-completion-shaped response. Retrieval
resolves a Mixedbread SDK client from `MXBAI_API_KEY` unless you pass a `client`
of your own (see below).

Every rollout installs the policy tokenizer at its entry point — from
`AGENT_HARNESS_TOKENIZER`, else the model name in `SEARCHER_AGENT_CONFIG` — so
context budgeting and payload truncation count what the model will actually see.
Which counter measured a rollout is recorded on it, as
`openai.metadata.token_counter_mode` (`exact-gigatoken`, `exact`, or
`chars-heuristic`); exact counting is required by default: set
`AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER=0` to instead allow a rollout whose
tokenizer would not load to run on the `chars/4` estimate.

Pass `client=` to run the retrieval tools against your own store client instead of
the public API. Anything structurally matching `agent_harness.RetrievalClient` works
— the SDK client does, and so does an in-process implementation, which is how the
harness runs inside a service that already owns the store layer:

```python
result = agent_harness.run_searcher(
    query, store_identifiers=[store], client=my_store_client, generation_fn=generate
)
```

## Async

The agent loop is async-native; the sync API above is a compatibility
surface that adapts sync seams (the SDK client included) onto it per call.
Async services that run the harness in-process on their own event loop
consume the coroutines directly and inject async seams, with no thread
bridging:

```python
from agent_harness import aio

result = await aio.run_fast_agentic_search(
    query,
    store_identifiers=[store],
    client=my_async_store_client,  # agent_harness.AsyncRetrievalClient
    generation_fn=my_async_generate,  # agent_harness.AsyncGenerationFn
)

async for event in aio.stream_fast_agentic_search(...):  # progress events
    ...
```

Every event stream ends with exactly one terminal event -- ``RolloutCompleted``,
``RolloutFailed`` (whose error also re-raises from the iterator) or
``RolloutCancelled`` -- so consumers that account rollouts can key off it.
Cancelling the consuming task cancels the rollout and its in-flight seam
awaits. Parallel tool calls fan out with ``asyncio.gather``; bulk token
counting (history baselines, round truncation) runs off the event loop, while
the small per-call payload budget checks count inline. Per-rollout knobs travel as
``tuning=HarnessTuning(...)`` instead of process-global env vars, and
``agent_harness.testing.verify_retrieval_client`` conformance-tests a
retrieval binding by driving one real rollout against it.

## Running BrowseComp-Plus

`examples/browsecomp/` drives BrowseComp-Plus queries through the harness and
scores the resulting rankings. With a served model up:

```bash
export MXBAI_API_KEY=...                       # retrieval
export AGENT_HARNESS_TOKENIZER=/models/my-policy-model
export TOAST_MODEL_BASE_URL=http://127.0.0.1:30000/v1
python examples/browsecomp/run.py out/ queries.json
```

`queries.json` is a list of `{"query_id", "query", "relevant_ids"}` rows — query text
and evidence-document ids from the BrowseComp-Plus set. Score the rollouts with
`examples/browsecomp/scoring.py`:
`score(rollout["retrieval"], row["relevant_ids"])` returns nDCG@10 and recall@10.

## Configuration

| Variable | Effect |
| --- | --- |
| `MXBAI_API_KEY` | Mixedbread API key (required unless a `client` is injected). `MBREAD_API_KEY` and `MIXEDBREAD_API_KEY` are accepted aliases, tried in that order; pass `api_key_env=` to read some other variable instead. |
| `MXBAI_BASE_URL` | Point the store client at a non-default Mixedbread deployment (aliases `MBREAD_BASE_URL`, `MIXEDBREAD_BASE_URL`). |
| `AGENT_HARNESS_TOOL_CHOICE` | Override the wire `tool_choice` sent to the model. Per-rollout `HarnessTuning(tool_choice=...)` wins over the env var. |
| `KEEP_REASONING_HISTORY` | Round-trip `reasoning_content` back into the message history. Off by default. Per-rollout `HarnessTuning(keep_reasoning_history=...)` wins over the env var. |
| `AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER` | On by default: a rollout whose policy tokenizer could not be installed fails instead of budgeting with the `chars/4` estimate. Set to `0` to allow the heuristic. |
| `AGENT_HARNESS_TOKENIZER` | Load this checkpoint as the token counter instead of the model name the rollout carries (or the one passed to `ensure_token_counter`). Needed when the served name is an alias no tokenizer resolves for. |
| `AGENT_HARNESS_TOKEN_COUNTER_BACKEND` | `gigatoken` (default; exact-parity-checked against HF, falls back to HF on any failure) or `hf` to opt out. gigatoken resolves `tokenizer.json` from a local checkpoint directory, else downloads it via `huggingface_hub`, else the HF tokenizer counts. |
| `AGENT_HARNESS_CORPUS_BACKEND_TOP_K` | Candidates requested from the provider per `search_corpus` call, without changing any agent-visible limit. Read on first use, then frozen for the process; must be at least 5. Per-rollout `HarnessTuning(backend_top_k=...)` wins over the env var. |
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
