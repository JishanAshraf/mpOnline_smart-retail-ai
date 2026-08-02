"""Pydantic schemas for request and response models across all endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# --- Vision Schemas ---

class FaceRecognitionResponse(BaseModel):
    customer_id: str = Field(..., json_schema_extra={"example": "CUST_10492"})
    status: str = Field(..., json_schema_extra={"example": "recognized"})  # "recognized" or "unknown"
    confidence: float = Field(..., json_schema_extra={"example": 0.95})


class ProductClassificationResponse(BaseModel):
    category: str = Field(..., json_schema_extra={"example": "Electronics/Headphones"})
    confidence: float = Field(..., json_schema_extra={"example": 0.92})


# --- NLP Schemas ---

class SentimentRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "I love the fast delivery and quality of this product!"})


class SentimentResponse(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "I love the fast delivery and quality of this product!"})
    sentiment: str = Field(..., json_schema_extra={"example": "positive"})  # "positive", "negative", or "neutral"
    confidence: float = Field(..., json_schema_extra={"example": 0.98})


# --- Chatbot & Dashboard Schemas ---

class ChatbotRequest(BaseModel):
    message: str = Field(..., json_schema_extra={"example": "Where is my order?"})
    user_id: Optional[str] = Field(None, json_schema_extra={"example": "user_123"})


class ChatbotResponse(BaseModel):
    message: str = Field(..., json_schema_extra={"example": "You can track your order by visiting the Order Status page."})
    intent: str = Field(..., json_schema_extra={"example": "order_status"})
    match_type: str = Field(..., json_schema_extra={"example": "rule_based"})  # "rule_based" or "ml_classifier"


class DashboardStatsResponse(BaseModel):
    total_visits: int = Field(..., json_schema_extra={"example": 1245})
    sentiment_counts: Dict[str, int] = Field(
        ...,
        json_schema_extra={"example": {"positive": 850, "negative": 120, "neutral": 275}}
    )
    top_intents: List[Dict[str, Any]] = Field(
        ...,
        json_schema_extra={"example": [
            {"intent": "order_status", "count": 412},
            {"intent": "return_policy", "count": 289},
            {"intent": "shipping_info", "count": 194}
        ]}
    )
