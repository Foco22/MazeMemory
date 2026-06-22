from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    provider: str  # "anthropic", "openai", "google"
    model: str     # LiteLLM model string used for API calls
    version: str   # human-readable version label for tracking
    api_base: str | None = None
    api_key: str | None = None
    rate_limit: int | None = None  # max requests per minute; None = unlimited


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
ANTHROPIC_OPUS48 = ModelConfig(
    provider="anthropic",
    model="claude-opus-4-8",
    version="claude-opus-4-8",
)
OPENAI_GPT5 = ModelConfig(
    provider="openai",
    model="gpt-5",
    version="gpt-5",
)
OPENAI_GPT54 = ModelConfig(
    provider="openai",
    model="gpt-5.4",
    version="gpt-5.4",
)
OPENAI_GPT55 = ModelConfig(
    provider="openai",
    model="gpt-5.5",
    version="gpt-5.5",
)
GOOGLE_GEMINI_31_PRO = ModelConfig(
    provider="google",
    model="gemini/gemini-3.1-pro-preview",
    version="gemini-3.1-pro-preview",
    rate_limit=20,  # free tier quota: 25 req/min; 20 leaves headroom
)