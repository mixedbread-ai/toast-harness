"""A scripted stand-in for ``openai``'s ``chat.completions.create``."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from copy import deepcopy
from types import SimpleNamespace
from typing import Any


def tool_call(name: str, arguments: dict[str, Any], *, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class Message:
    """Just enough of the SDK's message: attributes plus ``model_dump``."""

    def __init__(
        self,
        content: str | None,
        tool_calls: list[SimpleNamespace] | None,
        reasoning_content: str | None,
    ) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls
        self.reasoning_content = reasoning_content

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        dumped: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in self.tool_calls
            ]
            if self.tool_calls
            else None,
        }
        return {k: v for k, v in dumped.items() if v is not None} if exclude_none else dumped


def response(
    *,
    content: str | None = None,
    tool_calls: Iterable[SimpleNamespace] = (),
    reasoning_content: str | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 10,
    hosted_tool_calls: Iterable[dict[str, Any]] = (),
    context_management: dict[str, Any] | None = None,
    completion_id: str = "cmpl_1",
) -> SimpleNamespace:
    calls = list(tool_calls)
    message = Message(content, calls or None, reasoning_content)
    return SimpleNamespace(
        id=completion_id,
        hosted_tool_calls=list(hosted_tool_calls),
        context_management=context_management,
        choices=[SimpleNamespace(finish_reason="tool_calls" if calls else "stop", message=message)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


Scripted = SimpleNamespace | Callable[[dict[str, Any]], SimpleNamespace]


class ScriptedClient:
    """``create(**request)`` records the request and returns the next scripted response.

    A scripted item may be a callable of the request, for responses that depend
    on what the loop sent (a chunk handle minted at runtime, say).
    """

    def __init__(self, responses: Iterable[Scripted], *, asynchronous: bool = False) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses = list(responses)
        create = self._acreate if asynchronous else self._create
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def _create(self, **request: Any) -> SimpleNamespace:
        snapshot = deepcopy(request)
        self.requests.append(snapshot)
        scripted = self._responses.pop(0)
        return scripted(snapshot) if callable(scripted) else scripted

    async def _acreate(self, **request: Any) -> SimpleNamespace:
        return self._create(**request)
