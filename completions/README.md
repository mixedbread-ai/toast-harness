# Toast-1 over the Mixedbread Completions API

Three ways to run the search model through `https://api.mixedbread.com/v1`.
Each example is a different split of who does the retrieval and how the
answer is produced:

| Script | Retrieval | Ends with |
| --- | --- | --- |
| `hosted_tools.py` | The API's hosted store tools: one request, no retrieval code | A plain-text answer, printed with the chunks the hosted tools retrieved |
| `own_harness.py` | Your own tools: `bm25_search` and `grep` over a directory of text files | A plain-text answer once the model stops calling tools |
| `agent_harness_on_api.py` | The toast-harness loop and its Stores tools | A ranked list of chunks, a plain-text answer, or both (`--answer-mode`) |

```bash
cd completions
export MXBAI_API_KEY=...   # or pass --api-key
python hosted_tools.py --store my-store "Which contract governs the 2019 Acme distribution agreement?"
python own_harness.py "Which firmware does the Nordhavn Water Utility fleet need for Modbus TCP, and what does the Ethernet module cost?"
python agent_harness_on_api.py --store my-store --answer-mode submit_ranking "Which robot vacuums run for at least 200 minutes?"
```

`hosted_tools.py` and `own_harness.py` need only `pip install openai`: the
endpoint is OpenAI-compatible, so that package is the client.
`agent_harness_on_api.py` also needs `pip install toast-harness`. Every script
reads `MXBAI_API_KEY` from the environment or a `.env` file, or takes it as
`--api-key`; `--out FILE` writes the full record as JSON.

## `hosted_tools.py`: the API runs the search

Declare the hosted `search_corpus` and `grep` tools for `--store` and the
API executes them inside the completion: the model searches as often as it
needs and the request returns with the answer. `include` returns the chunks
every hosted call retrieved, which the script prints as the evidence behind
the answer. `context_management` lets the model prune results it no longer
needs when a long search approaches the context window; the script prints
what was cleared.

## `own_harness.py`: your own retrieval

The tool-call loop from the [build-your-own-harness guide](https://www.mixedbread.com/docs/agent/build-your-own-harness)
in one file: `bm25_search` and `grep` over a directory of text files, up to
four search rounds with parallel tool calls, tool failures returned to the
model as data, and a plain-text answer once the model stops calling tools.
The loop continues one stored completion: every request after the first names
the previous completion as `previous_completion_id` and sends only the new
tool results, and `context_management` lets the model prune tool results it
no longer needs on the server, yours included. The tool docstrings and
`Annotated` hints are the schema the model sees, so connecting your own
backend means replacing the two tool bodies in `Tools`.
It ships a thirteen-document sample corpus about a fictional sensor maker,
with distractors such as a sensor that lacks Modbus TCP and a second,
unrelated firmware line, so the answers have to come from retrieval;
`--corpus DIR` points it at your own files.

## `agent_harness_on_api.py`: the toast-harness loop

The full harness (`pip install toast-harness`) on the hosted model: its
search, grep and read tools over a Mixedbread store, its own client-side
context management, and a structured ending. `--answer-mode none` ends with a
ranked list of chunks, `submit_ranking` with the ranking plus an answer, and
`plain_text` with an answer only; `--top-k` sets the length of the ranking.
The hosted model's tokenizer is not available locally, so the script budgets
on the harness's token estimate (`AGENT_HARNESS_TOKENIZER=estimate`); point
that variable at a tokenizer checkpoint to count exactly.

`tests/completions/` drives all three scripts against a scripted model; no
API key is spent.
