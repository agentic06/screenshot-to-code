import asyncio
import traceback
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Literal, Optional

import openai
from config import NUM_VARIANTS, NUM_VARIANTS_VIDEO
from custom_types import InputMode
from llm import Llm
from openai.types.chat import ChatCompletionMessageParam

from routes.generate_code import (
    ExtractedParams,
    ParameterExtractionStage,
    PromptCreationStage,
    ModelSelectionStage,
    AgenticGenerationStage,
)
from prompts.prompt_types import Stack

router = APIRouter()


class PromptContent(BaseModel):
    text: str = ""
    images: List[str] = Field(default_factory=list)
    videos: List[str] = Field(default_factory=list)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str = ""
    images: List[str] = Field(default_factory=list)
    videos: List[str] = Field(default_factory=list)


class FileState(BaseModel):
    path: Optional[str] = None
    content: Optional[str] = None


class GenerateCodeRequest(BaseModel):
    prompt: PromptContent
    generatedCodeConfig: Stack = "html_tailwind"
    inputMode: InputMode = "image"
    generationType: Literal["create", "update"] = "create"
    numVariants: int = Field(default=1, ge=1, le=4)
    history: List[HistoryMessage] = Field(default_factory=list)
    fileState: Optional[FileState] = None
    optionCodes: List[Optional[str]] = Field(default_factory=list)
    isImageGenerationEnabled: bool = False


class VariantResult(BaseModel):
    index: int
    code: str


class GenerateCodeResponse(BaseModel):
    success: bool
    variants: List[VariantResult]
    errors: List[str]


async def _noop_send_message(
    type: str,
    value: Any,
    variant_index: int,
    data: Any,
    event_id: Any,
) -> None:
    pass


@router.post("/api/generate-code", response_model=GenerateCodeResponse)
async def generate_code_rest(request: GenerateCodeRequest):
    errors: List[str] = []
    num_variants = request.numVariants

    params: Dict[str, Any] = {
        "prompt": request.prompt.model_dump(),
        "generatedCodeConfig": request.generatedCodeConfig,
        "inputMode": request.inputMode,
        "generationType": request.generationType,
        "isImageGenerationEnabled": request.isImageGenerationEnabled,
        "history": [h.model_dump() for h in request.history],
        "fileState": request.fileState.model_dump() if request.fileState else None,
        "optionCodes": request.optionCodes,
    }

    async def throw_error(msg: str) -> None:
        errors.append(msg)

    try:
        param_extractor = ParameterExtractionStage(throw_error)
        extracted = await param_extractor.extract_and_validate(params)
    except Exception as e:
        return GenerateCodeResponse(
            success=False, variants=[], errors=errors or [str(e)]
        )

    try:
        prompt_creator = PromptCreationStage(throw_error)
        prompt_messages = await prompt_creator.build_prompt_messages(extracted)
    except Exception as e:
        return GenerateCodeResponse(
            success=False, variants=[], errors=errors or [str(e)]
        )

    try:
        if extracted.input_mode == "video":
            max_variants = NUM_VARIANTS_VIDEO
        elif extracted.generation_type == "update":
            max_variants = 2
        else:
            max_variants = NUM_VARIANTS
        num_variants = min(num_variants, max_variants)

        model_selector = ModelSelectionStage(throw_error)
        all_models = await model_selector.select_models(
            generation_type=extracted.generation_type,
            input_mode=extracted.input_mode,
        )
        variant_models = all_models[:num_variants]
    except Exception as e:
        return GenerateCodeResponse(
            success=False, variants=[], errors=errors or [str(e)]
        )

    generation_stage = AgenticGenerationStage(
        send_message=_noop_send_message,
        should_generate_images=extracted.should_generate_images,
        file_state=extracted.file_state,
        option_codes=extracted.option_codes,
    )

    variant_completions = await generation_stage.process_variants(
        variant_models=variant_models,
        prompt_messages=prompt_messages,
    )

    variants: List[VariantResult] = []
    for i in range(len(variant_models)):
        code = variant_completions.get(i, "")
        variants.append(VariantResult(index=i, code=code))

    return GenerateCodeResponse(
        success=len(variant_completions) > 0,
        variants=variants,
        errors=errors,
    )
