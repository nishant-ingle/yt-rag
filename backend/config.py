from pathlib import Path
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_PLAYLISTS: list[dict[str, Any]] = [
    {
        "id": "ask-abhijit",
        "name": "#AskAbhijit",
        "url": "https://www.youtube.com/playlist?list=PLSw9nXpYo3m6Bb_MVBgd0jRE3U4hJvZkd",
    },
    {
        "id": "abhijit-chavda-podcast",
        "name": "Abhijit Chavda Podcast",
        "url": "https://www.youtube.com/playlist?list=PLSw9nXpYo3m7CUa1yvPZmxYCLkchGa9-e",
    },
]

CHAT_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-2.5-flash"},
    "claude": {"base_url": "https://api.anthropic.com/v1/", "model": "claude-sonnet-4-6"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(ROOT_DIR / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LLM_BASE_URL: str = Field(default="https://api.openai.com/v1")
    LLM_API_KEY: str = Field(default="")
    CHAT_MODEL: str = Field(default="gpt-4o-mini")

    EMBED_BASE_URL: str = Field(default="https://api.openai.com/v1")
    EMBED_API_KEY: str = Field(default="")
    EMBED_MODEL: str = Field(default="text-embedding-3-small")
    EMBED_DIM: int = Field(default=1536)

    READ_ONLY_MODE: bool = Field(default=False)
    ADMIN_TOKEN: str = Field(default="")

    DB_PATH: str = Field(default=str(ROOT_DIR / "data" / "rag.db"))

    PLAYLISTS: list[dict[str, Any]] = Field(default_factory=lambda: list(DEFAULT_PLAYLISTS))
    CHAT_PROVIDERS: dict[str, dict[str, str]] = Field(default_factory=lambda: dict(CHAT_PROVIDERS))


settings = Settings()
PLAYLISTS = settings.PLAYLISTS
