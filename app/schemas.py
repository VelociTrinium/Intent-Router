"""
Pydantic request and response schemas for 3-class NLP intent routing API.
Classes:
- GENERAL_CHAT: General conversation, conceptual questions, syntax help, hypothetical discussions.
- DB_READ: Fetch, search, filter, count, aggregate, or inspect live records/schema.
- DB_WRITE: Insert, update, delete, modify, or mutate records/state in a database.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

RouteName = Literal["GENERAL_CHAT", "DB_READ", "DB_WRITE"]
IntentType = RouteName  # Backward-compatibility alias

VALID_ROUTES: List[RouteName] = ["GENERAL_CHAT", "DB_READ", "DB_WRITE"]


class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Incoming English user prompt to route")
    chat_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional preceding conversation turns for future multi-turn support"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Show me all active pursuits with deal value greater than $1,000,000.",
                "chat_history": []
            }
        }
    }


class RouteResponse(BaseModel):
    route: RouteName = Field(..., description="Final routed classification decision")
    raw_prediction: RouteName = Field(..., description="Raw top prediction from underlying model before guardrails")
    confidence: float = Field(..., description="Confidence score associated with the routing decision")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across GENERAL_CHAT, DB_READ, and DB_WRITE"
    )
    fallback_triggered: bool = Field(
        default=False,
        description="True if confidence fallback or safety guardrail overrode the raw prediction"
    )
    backend: Optional[str] = Field(default=None, description="Classifier backend used (setfit, baseline, or ollama)")
    latency_ms: Optional[float] = Field(default=None, description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_loaded: bool
    confidence_threshold: float
    write_threshold: float
    classes: List[str]
