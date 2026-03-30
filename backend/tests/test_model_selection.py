import pytest
from unittest.mock import AsyncMock
from routes.generate_code import ModelSelectionStage
from llm import Llm
from config import LLM_MODEL_NAME


class TestModelSelection:
    """Test model selection uses backend config."""

    def setup_method(self):
        """Set up test fixtures."""
        mock_throw_error = AsyncMock()
        self.model_selector = ModelSelectionStage(mock_throw_error)

    @pytest.mark.asyncio
    async def test_uses_config_model(self):
        """Should use the model configured in backend."""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="text",
        )

        expected_model = Llm(LLM_MODEL_NAME)
        expected = [expected_model] * 4
        assert models == expected

    @pytest.mark.asyncio
    async def test_update_uses_two_variants(self):
        """Update mode should use 2 variants."""
        models = await self.model_selector.select_models(
            generation_type="update",
            input_mode="text",
        )

        expected_model = Llm(LLM_MODEL_NAME)
        expected = [expected_model] * 2
        assert models == expected

    @pytest.mark.asyncio
    async def test_video_mode_uses_two_variants(self):
        """Video mode should use 2 variants."""
        models = await self.model_selector.select_models(
            generation_type="create",
            input_mode="video",
        )

        expected_model = Llm(LLM_MODEL_NAME)
        expected = [expected_model] * 2
        assert models == expected
