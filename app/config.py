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
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        CONFIDENCE_THRESHOLD: float = Field(default=0.60, description="Minimum confidence before fallback to GENERAL_CHAT")
        WRITE_THRESHOLD: float = Field(default=0.40, description="Sensitivity threshold to trigger WRITE intent")
        ROUTER_BACKEND: Literal["setfit", "ollama", "baseline"] = Field(default="setfit", description="Active classifier backend")

        # Paths
        MODEL_PATH: Path = Field(default=BASE_DIR / "models" / "intent_router")
        BASELINE_MODEL_PATH: Path = Field(default=BASE_DIR / "models" / "baseline_tfidf" / "baseline_model.joblib")

        # Ollama configuration
        OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
        OLLAMA_MODEL: str = Field(default="llama3.2")

        # Server
        API_HOST: str = Field(default="0.0.0.0")
        API_PORT: int = Field(default=8000)
        DEBUG: bool = Field(default=False)

        class Config:
            env_file = str(BASE_DIR / ".env")
            extra = "ignore"

    settings = Settings()

except ImportError:
    # Graceful fallback if pydantic-settings is not yet installed
    class FallbackSettings:
        def __init__(self):
            self.CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
            self.WRITE_THRESHOLD: float = float(os.getenv("WRITE_THRESHOLD", "0.40"))
            self.ROUTER_BACKEND: str = os.getenv("ROUTER_BACKEND", "setfit").lower()
            self.MODEL_PATH: Path = BASE_DIR / "models" / "intent_router"
            self.BASELINE_MODEL_PATH: Path = BASE_DIR / "models" / "baseline_tfidf" / "baseline_model.joblib"
            self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
            self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
            self.API_PORT: int = int(os.getenv("API_PORT", "8000"))
            self.DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1")

    settings = FallbackSettings()
