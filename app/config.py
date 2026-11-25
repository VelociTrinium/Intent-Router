"""
Application configuration and threshold parameters.
Loads values from environment variables or .env file with safe defaults.
"""

import os
from pathlib import Path
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field

    class Settings(BaseSettings):
        CONFIDENCE_THRESHOLD: float = Field(
            default=0.60,
            description="Minimum confidence before fallback to GENERAL_CHAT"
        )
        WRITE_THRESHOLD: float = Field(
            default=0.40,
            description="Sensitivity threshold to trigger DB_WRITE route"
        )
        FALLBACK_ROUTE: str = Field(
            default="GENERAL_CHAT",
            description="Default route when prediction is uncertain"
        )
        ROUTER_BACKEND: Literal["setfit", "ollama", "baseline"] = Field(
            default="ollama",
            description="Active classifier backend (ollama, setfit, or baseline)"
        )

        # Backbone for SetFit
        SETFIT_BACKBONE: str = Field(
            default="BAAI/bge-small-en-v1.5",
            description="Compact English backbone for SetFit"
        )

        # Paths
        MODEL_PATH: Path = Field(
            default=BASE_DIR / "models" / "intent_router",
            description="Path to fine-tuned SetFit model artifacts"
        )
        BASELINE_MODEL_PATH: Path = Field(
            default=BASE_DIR / "models" / "baseline_tfidf" / "baseline_model.joblib",
            description="Path to baseline TF-IDF model artifact"
        )

        # Ollama configuration
        OLLAMA_BASE_URL: str = Field(
            default="http://127.0.0.1:11434",
            description="Base URL for Ollama service"
        )
        OLLAMA_MODEL: str = Field(
            default="intent-router",
            description="Fine-tuned Ollama classifier model name"
        )
        OLLAMA_MODE: Literal["single_token", "json_schema"] = Field(
            default="single_token",
            description="Ollama inference mode: 1-token logprobs classifier or legacy JSON schema"
        )
        OLLAMA_TIMEOUT_SEC: float = Field(
            default=15.0,
            description="HTTP timeout for Ollama /api/chat calls (allows cold model load)"
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

except ImportError:
    class FallbackSettings:
        def __init__(self):
            self.CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
            self.WRITE_THRESHOLD: float = float(os.getenv("WRITE_THRESHOLD", "0.40"))
            self.FALLBACK_ROUTE: str = os.getenv("FALLBACK_ROUTE", "GENERAL_CHAT")
            self.ROUTER_BACKEND: str = os.getenv("ROUTER_BACKEND", "ollama").lower()
            self.SETFIT_BACKBONE: str = os.getenv("SETFIT_BACKBONE", "BAAI/bge-small-en-v1.5")
            self.MODEL_PATH: Path = BASE_DIR / "models" / "intent_router"
            self.BASELINE_MODEL_PATH: Path = BASE_DIR / "models" / "baseline_tfidf" / "baseline_model.joblib"
            self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            self.OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "intent-router")
            self.OLLAMA_MODE: str = os.getenv("OLLAMA_MODE", "single_token").lower()
            self.OLLAMA_TIMEOUT_SEC: float = float(os.getenv("OLLAMA_TIMEOUT_SEC", "15.0"))
            self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
            self.API_PORT: int = int(os.getenv("API_PORT", "8000"))
            self.DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1")

    settings = FallbackSettings()
