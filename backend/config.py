import os

NUM_VARIANTS = 4
NUM_VARIANTS_VIDEO = 2

# LLM-related - 后端配置的模型
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", None)
LLM_API_KEY = os.environ.get("LLM_API_KEY", None)
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4.1-2025-04-14")

# Image generation (optional)
REPLICATE_API_KEY = os.environ.get("REPLICATE_API_KEY", None)

# Debugging-related
IS_DEBUG_ENABLED = bool(os.environ.get("IS_DEBUG_ENABLED", False))
DEBUG_DIR = os.environ.get("DEBUG_DIR", "")

# Set to True when running in production (on the hosted version)
# Used as a feature flag to enable or disable certain features
IS_PROD = os.environ.get("IS_PROD", False)
