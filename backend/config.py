from pathlib import Path
from typing import Any
from pydantic import Field, field_validator
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
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.2:3b"},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(ROOT_DIR / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LLM_BASE_URL: str = Field(default="http://localhost:11434/v1")
    LLM_API_KEY: str = Field(default="ollama")
    CHAT_MODEL: str = Field(default="llama3.2:3b")

    EMBED_BASE_URL: str = Field(default="http://localhost:11434/v1")
    EMBED_API_KEY: str = Field(default="ollama")
    EMBED_MODEL: str = Field(default="nomic-embed-text")
    EMBED_DIM: int = Field(default=768)

    READ_ONLY_MODE: bool = Field(default=False)
    ADMIN_TOKEN: str = Field(default="")

    DB_PATH: str = Field(default=str(ROOT_DIR / "data" / "rag.db"))

    PLAYLISTS: list[dict[str, Any]] = Field(default_factory=lambda: list(DEFAULT_PLAYLISTS))
    CHAT_PROVIDERS: dict[str, dict[str, str]] = Field(default_factory=lambda: dict(CHAT_PROVIDERS))

    @field_validator(
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "CHAT_MODEL",
        "EMBED_BASE_URL",
        "EMBED_API_KEY",
        "EMBED_MODEL",
        "ADMIN_TOKEN",
        mode="before",
    )
    @classmethod
    def _strip_quotes_and_whitespace(cls, v: Any) -> Any:
        """Strip whitespace and matching surrounding quotes (e.g. from Docker --env-file)."""
        if isinstance(v, str):
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1].strip()
        return v


settings = Settings()
PLAYLISTS = settings.PLAYLISTS
