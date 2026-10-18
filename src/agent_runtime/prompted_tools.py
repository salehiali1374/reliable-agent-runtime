"""Tool calling for models without native tool support.

The tools are described in the system prompt and the model is asked to reply with one JSON
object: ``{"tool": ..., "arguments": {...}}`` to call a tool or ``{"answer": ...}`` to finish.
The reply is parsed back into the same ``AssistantTurn`` a native adapter would return, so the
runtime does not know the difference.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from typing import Any

from agent_runtime.llm import LLMClient
from agent_runtime.messages import AssistantTurn, Message, ToolCall, Usage
from agent_runtime.tools import ToolSpec

PROTOCOL = """\
You can use tools. To use a tool, reply with ONE JSON object and nothing else:
{{"thought": "<why this tool>", "tool": "<tool name>", "arguments": {{...}}}}
To request several independent calls at once, use:
{{"thought": "<why>", "calls": [{{"tool": "<name>", "arguments": {{...}}}}, ...]}}
When you have the final answer, reply with ONE JSON object:
{{"answer": "<your answer>"}}
Tool results arrive in the next message. Use only the tools listed below and follow their \
argument schemas exactly.

Tools:
{tools}"""

REPAIR = (
    "Your last reply was not a valid action. Reply with exactly one JSON object: "
    '{"tool": "<name>", "arguments": {...}} to call a tool, or {"answer": "..."} to finish.'
)


def _describe(spec: ToolSpec) -> str:
    schema = json.dumps(spec.parameters, separators=(",", ":"))
    return f"- {spec.name}: {spec.description}\n  arguments schema: {schema}"


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first top-level JSON object found in ``text`` (fences and prose tolerated)."""
    # strict=False: models often put raw newlines inside JSON strings
    decoder = json.JSONDecoder(strict=False)
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


class PromptedToolsClient:
    def __init__(self, inner: LLMClient, *, max_format_retries: int = 2):
        self.inner = inner
        self.max_format_retries = max_format_retries
        self._ids = itertools.count(1)

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AssistantTurn:
        chat = self._to_text_chat(messages, tools)
        usage = Usage()
        reply = ""
        for attempt in range(self.max_format_retries + 1):
            turn = self.inner.complete(chat, [])
            usage = usage + turn.usage
            reply = turn.content
            parsed = self._parse(reply)
            if parsed is not None:
                parsed.usage = usage
                return parsed
            if attempt < self.max_format_retries:
                chat = [
                    *chat,
                    Message(role="assistant", content=reply),
                    Message(role="user", content=REPAIR),
                ]
        # the model never produced a valid action: surface its text as the answer
        return AssistantTurn(content=reply, usage=usage)

    def _parse(self, reply: str) -> AssistantTurn | None:
        obj = extract_json_object(reply)
        if obj is None:
            # no JSON at all: plain prose is taken as the final answer
            return AssistantTurn(content=reply.strip()) if reply.strip() else None
        thought = str(obj.get("thought", ""))
        if "answer" in obj:
            answer = obj["answer"]
            return AssistantTurn(content=answer if isinstance(answer, str) else json.dumps(answer))
        raw_calls = obj.get("calls") if "calls" in obj else [obj] if "tool" in obj else None
        if not isinstance(raw_calls, list) or not raw_calls:
            return None
        calls = []
        for raw in raw_calls:
            if not isinstance(raw, dict) or not isinstance(raw.get("tool"), str):
                return None
            arguments = raw.get("arguments", {})
            if not isinstance(arguments, dict):
                return None
            calls.append(
                ToolCall(id=f"call_{next(self._ids)}", name=raw["tool"], arguments=arguments)
            )
        return AssistantTurn(content=thought, tool_calls=calls)

    @staticmethod
    def _to_text_chat(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> list[Message]:
        protocol = PROTOCOL.format(tools="\n".join(_describe(t) for t in tools) or "(none)")
        names = {c.id: c.name for m in messages for c in m.tool_calls}
        chat: list[Message] = []
        for message in messages:
            if message.role == "system":
                chat.append(Message(role="system", content=f"{message.content}\n\n{protocol}"))
            elif message.role == "assistant" and message.tool_calls:
                calls = [{"tool": c.name, "arguments": c.arguments} for c in message.tool_calls]
                action: dict[str, Any] = {"thought": message.content} if message.content else {}
                action.update(calls[0] if len(calls) == 1 else {"calls": calls})
                chat.append(Message(role="assistant", content=json.dumps(action)))
            elif message.role == "assistant":
                chat.append(
                    Message(role="assistant", content=json.dumps({"answer": message.content}))
                )
            elif message.role == "tool":
                label = "ERROR from" if message.is_error else "Result of"
                name = names.get(message.tool_call_id or "", "tool")
                chat.append(
                    Message(
                        role="user",
                        content=f"{label} {name} ({message.tool_call_id}):\n{message.content}",
                    )
                )
            else:
                chat.append(message)
        return chat
