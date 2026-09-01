# toast-harness

`toast-1` is Mixedbread's fast search agent: a model trained to drive retrieval
tools over a document store and end with a ranked list, an answer, or both. This
repository is two things:

1. **[Use toast-1](#use-toast-1-the-completions-api)** through the Mixedbread
   Completions API: an OpenAI-compatible endpoint, so the `openai` SDK is the
   client and `MXBAI_API_KEY` the key. Three runnable examples live in
   [`completions/`](completions/).
2. **[The harness it was trained in](#the-harness-agent_harness)**:
   `pip install toast-harness`, import name `agent_harness`. The retrieval tool
   surface, the agent loop, and exact token accounting, wired to any
   OpenAI-compatible served model.

## Use toast-1: the Completions API

`base_url="https://api.mixedbread.com/v1"`, `model="toast-1"`, your
`MXBAI_API_KEY`. No checkpoint, no tokenizer, and no retrieval code once you
declare a hosted store tool:

```python
import os

import openai

client = openai.OpenAI(
    base_url="https://api.mixedbread.com/v1", api_key=os.environ["MXBAI_API_KEY"]
)
completion = client.chat.completions.create(
    model="toast-1",
    messages=[{"role": "user", "content": "Which robot vacuums run for at least 200 minutes?"}],
    tools=[{"type": "store_search", "store_identifiers": ["my-store"]}],
)
print(completion.choices[0].message.content)
```

The hosted `store_search` runs inside the API for as many rounds as the model
wants; none of them come back as tool calls, and the answer arrives as plain
content.

### Three Examples

[`completions/`](completions/) has three example scripts. Each is a different
split of who does the retrieval and how the answer is produced:

| Script | Retrieval | Ends with |
| --- | --- | --- |
| [`hosted_tools.py`](completions/hosted_tools.py) | The API's hosted store tools: one request, no retrieval code | A plain-text answer, printed with the chunks the hosted tools retrieved |
| [`own_harness.py`](completions/own_harness.py) | Your own tools: `bm25_search` and `grep` over a directory of text files | A plain-text answer once the model stops calling tools |
| [`agent_harness_on_api.py`](completions/agent_harness_on_api.py) | The toast-harness loop and its Stores tools | A ranked list of chunks, a plain-text answer, or both ([`--answer-mode`](#answer-modes)) |

The first two need nothing beyond the `openai` SDK: the endpoint is
OpenAI-compatible, so that package is the client, and no OpenAI account is
involved. `agent_harness_on_api.py` also needs `pip install toast-harness`.

```bash
cd completions
export MXBAI_API_KEY=...   # or pass --api-key
python hosted_tools.py --store my-store "Which contract governs the 2019 Acme distribution agreement?"
python own_harness.py "Which firmware does the Nordhavn Water Utility fleet need for Modbus TCP, and what does the Ethernet module cost?"
python agent_harness_on_api.py --store my-store --answer-mode submit_ranking "Which robot vacuums run for at least 200 minutes?"
```

`own_harness.py` ships a thirteen-document sample corpus, so the second command
needs no store at all; the other two take the Mixedbread store to search as
`--store`. Every script reads `MXBAI_API_KEY` from the environment or a `.env`
file, or takes it as `--api-key`, and writes the full record as JSON with
`--out FILE`. `tests/completions/` drives all three against a scripted model;
no key is spent.

### What the API does for you, and what you own

- **Hosted tools are opt-in.** `store_search`, `store_grep`, `store_list_chunks`,
  `store_metadata_facets` and `list_stores` run inside the completion for exactly
  the requests that list them in `tools`; `store_identifiers` pins the store(s),
  or omit it and add `list_stores` to let the model pick one. `hosted_tools.py`
  lists `store_search` and `store_grep`. List none of them to bring your own
  backend: every tool call then comes back to your loop as an ordinary function
  call that you answer with one tool message, which is what `own_harness.py`
  and `agent_harness_on_api.py` do.
- **The answer arrives as plain content.** When the model has enough evidence it
  replies in prose with `finish_reason="stop"`; `hosted_tools.py` and
  `own_harness.py` take that reply as the answer. For a structured ending,
  declare your own terminal function and force it by name on the final turn
  (`tool_choice={"type": "function", "function": {"name": ...}}`), as
  `agent_harness_on_api.py` does with `submit_ranking`.
- **Hosted results are visible on request.** The response carries
  `hosted_tool_calls` (queries, pattern, status); the chunks themselves ride
  along with `extra_body={"include": ["store_search_call.results",
  "store_grep_call.results"]}`, keyed by `filename` and `chunk_index`.
  `hosted_tools.py` prints them as the evidence behind the answer.
- **Completions are stored by default.** With `store=True` a follow-up request
  can name the previous completion as `previous_completion_id` and continue it
  with everything the hosted tools retrieved. The examples send `store=False`:
  each request is complete in itself. `usage.prompt_tokens` counts the hosted
  rounds too.

[`completions/README.md`](completions/README.md) has the per-script details.
The [Completions API docs](https://www.mixedbread.com/docs/agent/chat-completions)
and the [build-your-own-harness guide](https://www.mixedbread.com/docs/agent/build-your-own-harness)
cover the endpoint itself.

## The harness: `agent_harness`

The harness `toast-1` was trained in. It ships the retrieval tool surface over a
Mixedbread store (`search_corpus`, `grep`, `get_chunks`, `read_document`,
`filter_chunks`, `prune_context`, `submit_ranking`), the agent loop that drives
those tools to a ranked list and/or an answer, and exact token accounting so
context budgeting and payload truncation measure what the model actually sees.

```bash
pip install toast-harness
```

### Quickstart

Wired against any OpenAI-compatible served model. `examples/browsecomp/run.py` is
a complete client of this shape against a self-hosted checkpoint;
`completions/agent_harness_on_api.py` is the same client against the
Completions API.

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

### Answer modes

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

### Async

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

### Token counting

Every rollout installs the policy tokenizer at its entry point -- from
`AGENT_HARNESS_TOKENIZER`, else the model name in `SEARCHER_AGENT_CONFIG` -- so
budgets count what the model will see. The counter used is recorded on the
record as `openai.metadata.token_counter_mode`; a rollout whose tokenizer will
not load fails unless `AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER=0` allows the
`chars/4` estimate.

### Running BrowseComp-Plus

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

### Configuration

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

CI runs the harness suite and the recipe tests against the built wheel in a
fresh venv, not against `src/`.

## License

Apache-2.0.
