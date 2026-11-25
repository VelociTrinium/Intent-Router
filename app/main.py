"""
FastAPI service exposing production endpoints for 3-class Intent Router:
- GET /health: Service readiness, active backend, and threshold configuration
- POST /route: Fast intent routing endpoint with safety guardrails & probabilities
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import RouteRequest, RouteResponse, HealthResponse, VALID_ROUTES
from app.router_engine import RouterEngine

# Global router engine instance
engine: RouterEngine = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if engine is None:
        engine = RouterEngine()
    yield


app = FastAPI(
    title="3-Class NLP Intent Router",
    description=(
        "Production-grade intent classification router classifying user prompts into "
        "GENERAL_CHAT, DB_READ, or DB_WRITE with calibrated probabilities and safety guardrails."
    ),
    version="1.0.0",
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
        classes=VALID_ROUTES
    )


@app.post("/route", response_model=RouteResponse, tags=["Routing"])
def route_prompt(request: RouteRequest):
    """
    Classifies a user prompt into GENERAL_CHAT, DB_READ, or DB_WRITE.
    Applies write sensitivity guardrail and confidence fallback rules.
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
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intent routing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
