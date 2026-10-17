"""LLM adapter for servers that implement the OpenAI chat-completions API with tool calls.

Uses only the standard library. Network and server failures that may succeed on retry
(timeouts, connection errors, HTTP 408/429/5xx) raise ``TransientLLMError``; everything else
raises ``LLMError``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from agent_runtime.llm import LLMError, TransientLLMError
from agent_runtime.messages import AssistantTurn, Message, ToolCall, Usage
from agent_runtime.tools import ToolSpec

_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _to_wire(message: Message) -> dict[str, Any]:
    if message.role == "assistant":
        wire: dict[str, Any] = {"role": "assistant", "content": message.content or None}
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in message.tool_calls
            ]
        return wire
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    return {"role": message.role, "content": message.content}


def _parse_tool_call(raw: dict[str, Any], index: int) -> ToolCall:
    function = raw.get("function", {})
    call_id = raw.get("id") or f"call_{index}"
    name = function.get("name", "")
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, dict):  # some servers return an object instead of a JSON string
        return ToolCall(id=call_id, name=name, arguments=arguments)
    try:
        parsed = json.loads(arguments or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("arguments must be a JSON object")
    except ValueError as exc:
        return ToolCall(id=call_id, name=name, parse_error=str(exc))
    return ToolCall(id=call_id, name=name, arguments=parsed)


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_wire(m) for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
        body = self._post("/chat/completions", payload)
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(body)[:200]}") from exc

        usage = body.get("usage") or {}
        return AssistantTurn(
            content=message.get("content") or "",
            tool_calls=[
                _parse_tool_call(c, i) for i, c in enumerate(message.get("tool_calls") or [])
            ],
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode(), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200]
            error_cls = TransientLLMError if exc.code in _TRANSIENT_STATUS else LLMError
            raise error_cls(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise TransientLLMError(f"connection failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"response is not valid JSON: {exc}") from exc
