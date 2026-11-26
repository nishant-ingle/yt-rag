from pathlib import Path
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_PLAYLISTS: list[dict[str, Any]] = [
    {
        "id": "ask-abhijit",
        "name": "#AskAbhijit Show",
        "url": "https://www.youtube.com/playlist?list=PLSw9nXpYo3m6Bb_MVBgd0jRE3U4hJvZkd",
    },
    {
        "id": "abhijit-chavda-podcast",
        "name": "Abhijit Chavda Podcast",
        "url": "https://www.youtube.com/playlist?list=PLSw9nXpYo3m7CUa1yvPZmxYCLkchGa9-e",
    },

    {
        "id": "geopolitical-videos",
        "name": "Geopolitical Videos",
        "url": "https://youtube.com/playlist?list=PLSw9nXpYo3m5B0oy2aMBlmWZGvb35Vv7T&si=3zR_ldctm_meWdVK",
    },
    {
        "id": "geostrategy-with-abhijit-chavda-podcast",
        "name": "Geostrategy with Abhijit Chavda",
        "url": "https://youtube.com/playlist?list=PLSw9nXpYo3m5gO6wo-CwCFQe7hos3MC_B&si=ymhbxEzN4Op-xI85",
    },
    {
        "id": "the-indian-interest-podcast",
        "name": "The #IndianInterest Podcast",
        "url": "https://youtube.com/playlist?list=PLSw9nXpYo3m6nR38Csp5Ea61pnLdRxdn-&si=yUkpeF83-vJHBELf",
    },
    {
        "id": "understanding-america-w-abhijit-chavda",
        "name": "Understanding America w/ Abhijit Chavda",
        "url": "https://youtube.com/playlist?list=PLSw9nXpYo3m432nSt6uOp9rWYlwCtcBVq&si=uqqbzRCExd1LoqWU",
    },

    {
        "id": "collaborations",
        "name": "Collaborations",
        "url": "https://youtube.com/playlist?list=PLSw9nXpYo3m7AJ1XdjXwAxsuWUf7hL3dR&si=nG1eumfIeJJ465ll",
    }
]

CHAT_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-3.8-flash"},
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
