import pytest

from routes.generate_code import ParameterExtractionStage


@pytest.mark.asyncio
async def test_extracts_basic_params() -> None:
    from unittest.mock import AsyncMock

    stage = ParameterExtractionStage(AsyncMock())

    extracted = await stage.extract_and_validate(
        {
            "generatedCodeConfig": "html_tailwind",
            "inputMode": "text",
            "prompt": {"text": "hello"},
        }
    )

    assert extracted.stack == "html_tailwind"
    assert extracted.input_mode == "text"
