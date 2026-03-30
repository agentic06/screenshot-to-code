from llm import Llm, OPENAI_MODELS, ANTHROPIC_MODELS, GEMINI_MODELS
from agent.runner import Agent
from prompts.create.image import build_image_prompt_messages
from prompts.prompt_types import Stack
from openai.types.chat import ChatCompletionMessageParam
from typing import Any


async def generate_code_for_image(image_url: str, stack: Stack, model: Llm) -> str:
    prompt_messages = build_image_prompt_messages(
        image_data_urls=[image_url],
        stack=stack,
        text_prompt="",
        image_generation_enabled=True,
    )

    async def send_message(
        _: str,
        __: str | None,
        ___: int,
        ____: dict[str, Any] | None = None,
        _____: str | None = None,
    ) -> None:
        # Evals do not stream tool/assistant messages to a frontend.
        return None

    print(f"[EVALS] Using agent runner for model: {model.value}")

    runner = Agent(
        send_message=send_message,
        variant_index=0,
        should_generate_images=True,
        initial_file_state=None,
        option_codes=None,
    )
    return await runner.run(model, prompt_messages)
