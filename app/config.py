"""
Application configuration and threshold parameters.
Loads values from environment variables or .env file with safe defaults.
"""

from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.60,
        description="Minimum confidence before fallback to FALLBACK_ROUTE",
    )
    WRITE_THRESHOLD: float = Field(
        default=0.40,
        description="Sensitivity threshold to trigger DB_WRITE when clarification is not stronger",
    )
    CLARIFY_THRESHOLD: float = Field(
        default=0.40,
        description="Sensitivity threshold to prioritize CLARIFICATION_REQUIRED over speculative writes",
    )
    FALLBACK_ROUTE: str = Field(
        default="CLARIFICATION_REQUIRED",
        description="Default route when prediction confidence is below CONFIDENCE_THRESHOLD",
    )

    # Ollama configuration
    OLLAMA_BASE_URL: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for Ollama service",
    )
    OLLAMA_MODEL: str = Field(
        default="intent-router-qwen2.5-0.5b-instruct:v1",
        description="Fine-tuned Ollama classifier model name (with optional :version tag)",
    )
    OLLAMA_TIMEOUT_SEC: float = Field(
        default=15.0,
        description="HTTP timeout for Ollama /api/chat calls (allows cold model load)",
    )

    # Server
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        extra="ignore",
    )


settings = Settings()
