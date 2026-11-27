"""
FastAPI service exposing production endpoints for 4-class Intent Router:
- GET /health: Service readiness, active backend, and threshold configuration
- POST /route: Fast intent routing endpoint with safety guardrails & probabilities
"""

import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import BASE_DIR, settings
from app.schemas import RouteRequest, RouteResponse, HealthResponse, VALID_ROUTES
from app.router_engine import RouterEngine

logger = logging.getLogger("intent_router")
_REVIEW_LOG_LOCK = threading.Lock()

# Global router engine instance
engine: Optional[RouterEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if engine is None:
        engine = RouterEngine()
    yield


app = FastAPI(
    title="4-Class NLP Intent Router",
    description=(
        "Production-grade intent classification router classifying user prompts into "
        "GENERAL_CHAT, DB_READ, DB_WRITE, or CLARIFICATION_REQUIRED with calibrated "
        "probabilities and safety guardrails."
    ),
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Returns server status, active backend readiness, and routing thresholds."""
    global engine
    if engine is None:
        engine = RouterEngine()

    return HealthResponse(
        status="healthy",
        backend=engine.backend_name,
        model_loaded=engine.router.is_ready(),
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        write_threshold=settings.WRITE_THRESHOLD,
        clarify_threshold=settings.CLARIFY_THRESHOLD,
        fallback_route=settings.FALLBACK_ROUTE,
        classes=VALID_ROUTES
    )


@app.post("/route", response_model=RouteResponse, tags=["Routing"])
def route_prompt(request: RouteRequest):
    """
    Classifies a user prompt into GENERAL_CHAT, DB_READ, DB_WRITE, or CLARIFICATION_REQUIRED.
    Applies clarification-priority, write-sensitivity, and confidence-fallback guardrails.
    """
    global engine
    if engine is None:
        engine = RouterEngine()

    cleaned_prompt = request.prompt.strip()
    if not cleaned_prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Prompt cannot be empty or whitespace only."
        )

    try:
        response = engine.route(
            prompt=cleaned_prompt,
            chat_history=request.chat_history
        )

        # Log low-confidence or fallback-triggered predictions when enabled
        if settings.ENABLE_REVIEW_LOGGING and (
            response.confidence < settings.REVIEW_CONFIDENCE_THRESHOLD or response.fallback_triggered
        ):
            _log_for_review(cleaned_prompt, request.chat_history, response)

        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intent routing failed: {str(e)}"
        )


def _log_for_review(
    prompt: str,
    chat_history: Optional[list],
    response: RouteResponse,
) -> None:
    """Appends low-confidence or fallback predictions to REVIEW_LOG_PATH for human review."""
    try:
        log_path = Path(settings.REVIEW_LOG_PATH)
        if not log_path.is_absolute():
            log_path = BASE_DIR / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt": prompt,
            "chat_history": chat_history or [],
            "route": response.route,
            "raw_prediction": response.raw_prediction,
            "confidence": response.confidence,
            "probabilities": response.probabilities,
            "fallback_triggered": response.fallback_triggered,
            "backend": response.backend,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _REVIEW_LOG_LOCK:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        logger.warning(f"Failed to log review entry: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
