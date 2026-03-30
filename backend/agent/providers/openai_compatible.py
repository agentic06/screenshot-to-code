# pyright: reportUnknownVariableType=false
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.providers.base import (
    EventSink,
    ExecutedToolCall,
    ProviderSession,
    ProviderTurn,
    StreamEvent,
)
from agent.providers.token_usage import TokenUsage
from agent.state import ensure_str
from agent.tools import CanonicalToolDefinition, ToolCall, parse_json_arguments


def serialize_chat_tools(
    tools: List[CanonicalToolDefinition],
) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


@dataclass
class ChatParseState:
    assistant_text: str = ""
    tool_calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)


async def _parse_chat_chunk(
    chunk: Any,
    state: ChatParseState,
    on_event: EventSink,
) -> None:
    if not chunk.choices:
        return

    delta = chunk.choices[0].delta

    if delta.content:
        state.assistant_text += delta.content
        await on_event(StreamEvent(type="assistant_delta", text=delta.content))

    if not delta.tool_calls:
        return

    for tc in delta.tool_calls:
        if tc.id:
            state.tool_calls[tc.id] = {
                "id": tc.id,
                "name": tc.function.name or "",
                "arguments": tc.function.arguments or "",
            }
        elif tc.index is not None and tc.function:
            call_id = _find_call_id_by_index(state.tool_calls, tc.index)
            if call_id and call_id in state.tool_calls:
                entry = state.tool_calls[call_id]
                if tc.function.name:
                    entry["name"] = tc.function.name
                if tc.function.arguments:
                    entry["arguments"] += tc.function.arguments
                await on_event(
                    StreamEvent(
                        type="tool_call_delta",
                        tool_call_id=call_id,
                        tool_name=entry["name"],
                        tool_arguments=entry["arguments"],
                    )
                )


def _find_call_id_by_index(
    tool_calls: Dict[str, Dict[str, Any]], index: int
) -> str | None:
    for i, (cid, entry) in enumerate(tool_calls.items()):
        if i == index:
            return cid
    return None


def _extract_chat_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    return TokenUsage(
        input=input_tokens,
        output=output_tokens,
        cache_read=0,
        cache_write=0,
        total=total_tokens,
    )


class OpenAICompatibleProviderSession(ProviderSession):
    def __init__(
        self,
        client: AsyncOpenAI,
        model_name: str,
        prompt_messages: List[ChatCompletionMessageParam],
        tools: List[Dict[str, Any]],
    ):
        self._client = client
        self._model_name = model_name
        self._tools = tools
        self._total_usage = TokenUsage()
        self._messages: List[Dict[str, Any]] = [
            dict(message) for message in prompt_messages
        ]

    async def stream_turn(self, on_event: EventSink) -> ProviderTurn:
        params: Dict[str, Any] = {
            "model": self._model_name,
            "messages": self._messages,
            "tools": self._tools if self._tools else None,
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 50000,
        }

        state = ChatParseState()
        stream = await self._client.chat.completions.create(**params)
        final_response = None
        async for chunk in stream:
            final_response = chunk
            await _parse_chat_chunk(chunk, state, on_event)

        if final_response:
            self._total_usage.accumulate(_extract_chat_usage(final_response))

        tool_calls: List[ToolCall] = []
        for entry in state.tool_calls.values():
            args, error = parse_json_arguments(entry.get("arguments"))
            if error:
                args = {"INVALID_JSON": ensure_str(entry.get("arguments"))}
            tool_calls.append(
                ToolCall(
                    id=entry.get("id") or "",
                    name=entry.get("name") or "unknown_tool",
                    arguments=args,
                )
            )

        assistant_turn: List[Dict[str, Any]] = []
        if tool_calls:
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": state.assistant_text or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            assistant_turn.append(assistant_msg)

        return ProviderTurn(
            assistant_text=state.assistant_text,
            tool_calls=tool_calls,
            assistant_turn=assistant_turn,
        )

    def append_tool_results(
        self,
        turn: ProviderTurn,
        executed_tool_calls: list[ExecutedToolCall],
    ) -> None:
        if turn.assistant_turn:
            self._messages.extend(turn.assistant_turn)

        for executed in executed_tool_calls:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": executed.tool_call.id,
                    "content": json.dumps(executed.result.result),
                }
            )

    async def close(self) -> None:
        u = self._total_usage
        print(
            f"[TOKEN USAGE] provider=openai-compatible model={self._model_name} | "
            f"input={u.input} output={u.output} "
            f"total={u.total}"
        )
        await self._client.close()
