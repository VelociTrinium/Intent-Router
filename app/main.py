"""
FastAPI entrypoint exposing:
- GET /health: Healthcheck, active backend, model status, and threshold config
- POST /route: Intent classification routing endpoint
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import RouteRequest, RouteResponse, HealthResponse
from app.router_engine import RouterEngine

app = FastAPI(
    title="Intent Router Microservice",
    description="Low-latency multi-backend intent routing API for LLM agents.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global router engine instance
engine = RouterEngine()


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Healthcheck endpoint returning server status and model readiness."""
    return HealthResponse(
        status="healthy",
        backend=settings.ROUTER_BACKEND,
        model_loaded=engine.router.is_ready(),
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        write_threshold=settings.WRITE_THRESHOLD
    )


@app.post("/route", response_model=RouteResponse, tags=["Routing"])
def route_intent(request: RouteRequest):
    """
    Classifies and routes user query into READ, WRITE, or GENERAL_CHAT,
    applying safety thresholds and fallback mechanisms.
    """
    if not request.prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Prompt cannot be empty or whitespace only."
        )

    try:
        response = engine.route(
            prompt=request.prompt,
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
