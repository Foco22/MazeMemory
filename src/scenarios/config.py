from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "anthropic", "openai", "google"
    model: str     # LiteLLM model string used for API calls
    version: str   # human-readable version label for tracking
    api_base: str | None = None
    api_key: str | None = None


# Ready-to-use configs
ANTHROPIC_SONNET = ModelConfig(
    provider="anthropic",
    model="claude-sonnet-4-6",
    version="claude-sonnet-4-6",
)
ANTHROPIC_HAIKU = ModelConfig(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    version="claude-haiku-4-5",
)
OPENAI_GPT4O = ModelConfig(
    provider="openai",
    model="gpt-4o",
    version="gpt-4o",
)
OPENAI_GPT4O_MINI = ModelConfig(
    provider="openai",
    model="gpt-4o-mini",
    version="gpt-4o-mini",
)
GOOGLE_GEMINI_FLASH = ModelConfig(
    provider="google",
    model="gemini/gemini-2.0-flash",
    version="gemini-2.0-flash",
)