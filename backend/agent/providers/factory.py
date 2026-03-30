from typing import Optional

from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from config import LLM_API_KEY, LLM_BASE_URL
from agent.providers.anthropic import (
    AnthropicProviderSession,
    serialize_anthropic_tools,
)
from agent.providers.base import ProviderSession
from agent.providers.gemini import GeminiProviderSession, serialize_gemini_tools
from agent.providers.openai import OpenAIProviderSession, serialize_openai_tools
from agent.providers.openai_compatible import (
    OpenAICompatibleProviderSession,
    serialize_chat_tools,
)
from agent.tools import canonical_tool_definitions
from llm import Llm, MODEL_PROVIDER


def create_provider_session(
    model: Llm,
    prompt_messages: list[ChatCompletionMessageParam],
    should_generate_images: bool,
) -> ProviderSession:
    from config import LLM_MODEL_NAME

    canonical_tools = canonical_tool_definitions(
        image_generation_enabled=should_generate_images
    )

    if not LLM_API_KEY:
        raise Exception("LLM_API_KEY is missing in backend/.env")

    if model == Llm.CUSTOM:
        client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        return OpenAICompatibleProviderSession(
            client=client,
            model_name=LLM_MODEL_NAME,
            prompt_messages=prompt_messages,
            tools=serialize_chat_tools(canonical_tools),
        )

    provider = MODEL_PROVIDER.get(model, "openai")

    if provider == "openai":
        client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        return OpenAIProviderSession(
            client=client,
            model=model,
            prompt_messages=prompt_messages,
            tools=serialize_openai_tools(canonical_tools),
        )

    if provider == "anthropic":
        client = AsyncAnthropic(api_key=LLM_API_KEY)
        return AnthropicProviderSession(
            client=client,
            model=model,
            prompt_messages=prompt_messages,
            tools=serialize_anthropic_tools(canonical_tools),
        )

    if provider == "gemini":
        client = genai.Client(api_key=LLM_API_KEY)
        return GeminiProviderSession(
            client=client,
            model=model,
            prompt_messages=prompt_messages,
            tools=serialize_gemini_tools(canonical_tools),
        )

    raise ValueError(f"Unsupported model: {model.value}")
