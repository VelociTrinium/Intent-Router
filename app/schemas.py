"""
Pydantic request and response schemas for intent routing API.
"""

from typing import List, Optional, Dict, Literal

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def dict(self):
            return self.__dict__

        def model_dump(self):
            return self.__dict__

    def Field(*args, **kwargs):  # type: ignore
        return kwargs.get("default", None)

IntentType = Literal["READ", "WRITE", "GENERAL_CHAT"]


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of message author (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="Message text content")


class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Incoming user query to route")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional preceding conversation turns for contextual intent routing"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Can you delete order #4912 from the database?",
                "chat_history": [
                    {"role": "user", "content": "Hello, I made a mistake on my last order."},
                    {"role": "assistant", "content": "I can help with that. What order was it?"}
                ]
            }
        }
    }


class RouteResponse(BaseModel):
    intent: IntentType = Field(..., description="Final routed intent decision")
    confidence: float = Field(..., description="Confidence score associated with the final routing decision")
    raw_intent: str = Field(..., description="Raw top intent predicted by the underlying model before threshold rules")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution across READ, WRITE, GENERAL_CHAT")
    backend: str = Field(..., description="Name of backend used (setfit, baseline, or ollama)")
    routed_by_rule: str = Field(..., description="Explanation of which rule governed the decision")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    status: str
    backend: str
    model_loaded: bool
    confidence_threshold: float
    write_threshold: float
