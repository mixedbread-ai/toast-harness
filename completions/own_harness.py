"""Bring your own search backend: toast-1 over the Completions API with local tools.

A complete tool-call loop in one file. The model searches a directory of text
files with two local tools, ``bm25_search`` and ``grep``, and ends the episode
with a third, ``submit_answer``, once the evidence is sufficient: a structured
answer, as the toast harness ends on ``submit_ranking``, with the answer in its
``answer`` argument and the model's reasoning in ``chain_of_thought``. Every
turn has to call a tool, and the last round asks for ``submit_answer`` by name:

    python own_harness.py [--corpus DIR] [--out FILE] "question"

The loop continues one stored completion instead of resending the history:
every request after the first names the previous completion as
``previous_completion_id`` and sends only the new tool results. That keeps the
conversation on the server, where ``context_management`` lets the model prune
tool results it no longer needs, including the ones your own tools produced;
each response reports what was cleared as ``context_management.applied_edits``.

``openai`` is the only dependency; ``MXBAI_API_KEY`` comes from the environment,
a ``.env`` file or ``--api-key``. The sample corpus is thirteen documents about
a fictional sensor maker, so every answer has to come from retrieval. To connect
your own backend, replace the two tool bodies in ``Tools``: their docstrings and
``Annotated`` hints are the tool descriptions the model reads.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

MODEL = "toast-1"
BASE_URL = os.environ.get("MXBAI_COMPLETIONS_BASE_URL", "https://api.mixedbread.com/v1")
SAMPLING = {"temperature": 0.7, "top_p": 0.95}
CONTEXT_MANAGEMENT = {"edits": [{"type": "prune_context"}]}
SAMPLE_CORPUS = Path(__file__).parent / "sample_corpus"

MAX_ROUNDS = 6  # rounds of tool calls; the last one asks for submit_answer
MAX_PARALLEL_CALLS = 8  # per-turn fan-out the prompt asks for
TOP_K_MAX = 20
CLIP_CHARS = 2_000  # per chunk in a search result, about 500 tokens

SYSTEM_PROMPT = f"""You are a search agent answering a user query from a document corpus with \
search tools. Search until the evidence is sufficient, then call submit_answer.

- Plan first, then fan out: several bm25_search and grep calls in one turn that chase different \
aspects, entities and wording of the query; at most {MAX_PARALLEL_CALLS} tool calls per turn. \
Follow-up searches pivot to what is still missing, not to paraphrases.
- bm25_search matches keywords: send keyword-heavy queries, not questions. grep matches a \
regular expression against literal text: use it for identifiers, codes, dates and exact phrases. \
bm25_search returns only chunks you have not seen yet; grep also confirms matches in seen ones.
- You have at most {MAX_ROUNDS} rounds of tool calls; from the second round on, a "Search round \
N of {MAX_ROUNDS}." line marks the round, and the last round is marked as the last.
- submit_answer ends the episode: your reasoning in chain_of_thought, and in answer only the \
answer, based on the retrieved evidence alone. If the evidence is insufficient, say so in answer.
"""
LAST_ROUND = (
    f"Search round {MAX_ROUNDS} of {MAX_ROUNDS}: the last one. Call submit_answer now, based "
    "only on the retrieved evidence. If it is insufficient, say so in answer."
)
SUBMIT_ONLY = {"type": "function", "function": {"name": "submit_answer"}}  # by name, last round


# --- the corpus: text files split into chunks with stable handles ----------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    filename: str
    chunk_index: int
    text: str


def load_corpus(directory: Path) -> list[Chunk]:
    """One chunk per paragraph of every .md and .txt file; a heading joins the paragraph below.

    Handles are minted once, in file order, and never change: search and grep
    speak the same ids for the whole episode.
    """
    chunks: list[Chunk] = []
    for path in sorted(p for p in directory.iterdir() if p.suffix in (".md", ".txt")):
        heading, index = "", 0
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
            paragraph = block.strip()
            if not paragraph:
                continue
            if all(line.startswith("#") for line in paragraph.splitlines()):
                heading += paragraph + "\n"
                continue
            chunks.append(Chunk(f"c{len(chunks) + 1}", path.name, index, heading + paragraph))
            heading, index = "", index + 1
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25:
    """Okapi BM25 over the chunks (k1 = 1.5, b = 0.75)."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks, self._k1, self._b = chunks, k1, b
        self._counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._lengths = [sum(counts.values()) for counts in self._counts]
        self._average_length = sum(self._lengths) / max(len(chunks), 1)
        document_frequency = Counter(term for counts in self._counts for term in counts)
        self._idf = {
            term: math.log(1 + (len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str) -> list[tuple[Chunk, float]]:
        """Every chunk with a positive score, best first."""
        terms = tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for chunk, counts, length in zip(self._chunks, self._counts, self._lengths, strict=True):
            norm = self._k1 * (1 - self._b + self._b * length / self._average_length)
            score = sum(
                self._idf[term] * counts[term] * (self._k1 + 1) / (counts[term] + norm)
                for term in terms
                if term in counts
            )
            if score > 0:
                scored.append((chunk, score))
        return sorted(scored, key=lambda hit: -hit[1])


# --- the tools: two searches over one corpus, and the terminal submit_answer ------------


class Tools:
    """The model's tools: every public method is one.

    The docstring is the description the model follows and the ``Annotated``
    strings are the parameter descriptions; ``tool_schema`` reads both off the
    signature. ``seen`` holds every chunk shown so far, by handle:
    ``bm25_search`` skips those chunks, while ``grep`` still reports them.
    ``submit_answer`` is the terminal tool: a valid call to it ends the episode.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._bm25 = BM25(chunks)
        self.seen: dict[str, Chunk] = {}

    def by_name(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "bm25_search": self.bm25_search,
            "grep": self.grep,
            "submit_answer": self.submit_answer,
        }

    def bm25_search(
        self,
        query: Annotated[
            str,
            "Space-separated keywords, no natural-language questions, no boolean operators. "
            "Example: 'jordan international goals caps'",
        ],
        top_k: Annotated[int, f"Number of chunks to return, max {TOP_K_MAX}."] = 5,
        mode: Annotated[
            Literal["chunks", "documents"],
            "chunks ranks every chunk; documents keeps the best chunk of each document.",
        ] = "chunks",
    ) -> dict[str, Any]:
        """Keyword-based BM25 search over the corpus. This tool matches keywords only:
        send keyword-heavy queries, not questions. Use for rare terms, names, codes, and
        exact vocabulary; use grep for regular expressions and literal phrases. Returns
        up to top_k chunks with stable chunk_id handles."""
        _check_top_k(top_k)
        if mode not in ("chunks", "documents"):
            raise ValueError(f"mode must be 'chunks' or 'documents', got {mode!r}")
        hits = [(c, s) for c, s in self._bm25.search(query) if c.chunk_id not in self.seen]
        if mode == "documents":
            best = {}
            for chunk, score in hits:
                best.setdefault(chunk.filename, (chunk, score))
            hits = list(best.values())
        results = [{**self._show(c), "score": round(s, 4)} for c, s in hits[:top_k]]
        return {"query": query, "candidate_count": len(results), "results": results}

    def grep(
        self,
        pattern: Annotated[str, "A regular expression in Python re syntax, case-insensitive."],
        top_k: Annotated[int, f"Number of chunks to return, max {TOP_K_MAX}."] = 10,
    ) -> dict[str, Any]:
        """Find chunks whose literal text matches a regular expression. No semantic
        matching: use it for exact tokens, identifiers, codes, dates and literal phrases,
        and bm25_search for keyword relevance. Chunks with the most matches come first,
        whether or not you have seen them. Returns up to top_k chunks with stable chunk_id
        handles."""
        _check_top_k(top_k)
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc
        hits = [(c, n) for c in self._chunks if (n := len(regex.findall(c.text)))]
        hits.sort(key=lambda hit: -hit[1])
        results = [{**self._show(c), "match_count": n} for c, n in hits[:top_k]]
        return {"pattern": pattern, "candidate_count": len(results), "results": results}

    def submit_answer(
        self,
        chain_of_thought: Annotated[
            str, "Your reasoning: how the retrieved evidence supports the answer."
        ],
        answer: Annotated[
            str,
            "The answer to the user's query as one or two complete sentences of plain text: "
            "the facts asked for, with no account of the evidence or of your search.",
        ],
    ) -> dict[str, Any]:
        """End the episode with your answer, once the evidence is sufficient. Put your
        reasoning about the evidence in chain_of_thought and only the answer to the
        user's query in answer; if the evidence is insufficient, say so in answer."""
        return {"chain_of_thought": chain_of_thought, "answer": answer}

    def _show(self, chunk: Chunk) -> dict[str, Any]:
        """The result entry for ``chunk``, which counts as seen from now on."""
        self.seen[chunk.chunk_id] = chunk
        text = chunk.text[:CLIP_CHARS] + (" ...[truncated]" if len(chunk.text) > CLIP_CHARS else "")
        return {
            "chunk_id": chunk.chunk_id,
            "filename": chunk.filename,
            "chunk_index": chunk.chunk_index,
            "text": text,
        }


def _check_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or not 1 <= top_k <= TOP_K_MAX:
        raise ValueError(f"top_k must be an integer between 1 and {TOP_K_MAX}, got {top_k!r}")


def tool_schema(tool: Callable[..., Any]) -> dict[str, Any]:
    """The function schema for ``tool``: docstring -> description, Annotated -> parameters."""
    hints = get_type_hints(tool, include_extras=True)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in inspect.signature(tool).parameters.items():
        hint, description = get_args(hints[name])[:2]
        properties[name] = {**_type_schema(hint), "description": description}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]["default"] = parameter.default
    return {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": " ".join(inspect.cleandoc(tool.__doc__ or "").split()),
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _type_schema(hint: Any) -> dict[str, Any]:
    if get_origin(hint) is Literal:
        return {"type": "string", "enum": list(get_args(hint))}
    if get_origin(hint) is list:
        return {"type": "array", "items": _type_schema(get_args(hint)[0])}
    return {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}[hint]}


# --- the loop ----------------------------------------------------------------------------


@dataclass
class Episode:
    tools: Tools
    messages: list[dict[str, Any]]  # local transcript, for the record; never resent
    usage: Counter[str] = field(default_factory=Counter)
    generations: int = 0
    completion_id: str | None = None
    context_edits: list[dict[str, Any]] = field(default_factory=list)


def run(client: Any, query: str, *, corpus: Path = SAMPLE_CORPUS) -> dict[str, Any]:
    """Search ``corpus`` for ``query``; returns the submitted answer, usage and the transcript."""
    tools = Tools(load_corpus(corpus))
    offered = [tool_schema(tool) for tool in tools.by_name().values()]
    episode = Episode(tools, [])
    new = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    for round_index in range(1, MAX_ROUNDS + 1):
        last = round_index == MAX_ROUNDS
        if round_index > 1:
            new.append(
                _user(LAST_ROUND if last else f"Search round {round_index} of {MAX_ROUNDS}.")
            )
        # every round has to call a tool, so the model ends by calling submit_answer
        # rather than by falling silent; the last round asks for it by name
        message = _complete(
            client, episode, new, tools=offered, tool_choice=SUBMIT_ONLY if last else "required"
        )
        # in the model's order; a network backend would run them concurrently
        results = [(call, _execute(tools, call)) for call in message.tool_calls or []]
        if (submitted := _submission(results)) is not None:
            return _record(episode, submitted, rounds=round_index)
        new = [_tool_message(call, result) for call, result in results]
    raise RuntimeError("the model did not call submit_answer on the last round")


def _submission(results: list[tuple[Any, dict[str, Any]]]) -> dict[str, Any] | None:
    """The first valid submit_answer result of a turn; a failed one goes back as data."""
    for call, result in results:
        if call.function.name == "submit_answer" and "error" not in result:
            return result
    return None


def _complete(
    client: Any,
    episode: Episode,
    new: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any],
) -> Any:
    """One request continuing the stored completion; only ``new`` messages are sent."""
    extension = {"context_management": CONTEXT_MANAGEMENT}  # extension fields ride in extra_body
    if episode.completion_id is not None:
        extension["previous_completion_id"] = episode.completion_id
    response = client.chat.completions.create(
        model=MODEL,
        messages=new,
        tools=tools,  # local tools only
        tool_choice=tool_choice,
        parallel_tool_calls=tool_choice == "required",  # fan out to search, one call to submit
        extra_body=extension,
        **SAMPLING,
    )
    episode.completion_id = response.id  # the next request continues here
    episode.generations += 1
    if response.usage is not None:
        episode.usage["prompt_tokens"] += response.usage.prompt_tokens
        episode.usage["completion_tokens"] += response.usage.completion_tokens
    applied = (getattr(response, "context_management", None) or {}).get("applied_edits") or []
    episode.context_edits.extend(applied)
    message = response.choices[0].message
    turn = message.model_dump(exclude_none=True)
    turn.pop("reasoning_content", None)  # display narration, not history
    episode.messages.extend([*new, turn])
    return message


def _execute(tools: Tools, call: Any) -> dict[str, Any]:
    """Run one call; every failure comes back as data, so the model can correct it next turn."""
    tool = tools.by_name().get(call.function.name)
    if tool is None:
        return {"error": f"unknown tool {call.function.name}"}
    try:
        return tool(**json.loads(call.function.arguments or "{}"))
    except (TypeError, ValueError) as exc:  # bad JSON, unknown or missing parameters, bad values
        return {"error": str(exc)}


def _record(episode: Episode, submitted: dict[str, Any], *, rounds: int) -> dict[str, Any]:
    return {
        "answer": submitted["answer"],
        "chain_of_thought": submitted["chain_of_thought"],
        "seen": [
            {"chunk_id": c.chunk_id, "filename": c.filename, "chunk_index": c.chunk_index}
            for c in episode.tools.seen.values()
        ],
        "rounds": rounds,
        "generations": episode.generations,
        "completion_id": episode.completion_id,
        "context_edits": episode.context_edits,
        "usage": dict(episode.usage),
        "messages": episode.messages,
    }


def _user(content: str) -> dict[str, Any]:
    return {"role": "user", "content": content}


def _tool_message(call: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(result, ensure_ascii=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--corpus", type=Path, default=SAMPLE_CORPUS, help="a directory of .md/.txt"
    )
    parser.add_argument("--out", type=Path, help="write the full record (with transcript) here")
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
    record = run(client, args.query, corpus=args.corpus)
    print(record["answer"])
    print(
        "chunks shown:",
        len(record["seen"]),
        "generations:",
        record["generations"],
        "rounds:",
        record["rounds"],
        "pruned tokens:",
        sum(edit.get("cleared_input_tokens", 0) for edit in record["context_edits"]),
        "usage:",
        record["usage"],
    )
    if args.out:
        args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
