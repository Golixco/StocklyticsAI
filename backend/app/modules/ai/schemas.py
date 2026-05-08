"""
AI Module – Pydantic schemas.

Defines validated request payloads and typed response shapes for
the AI chat endpoints per api_contracts.md §8.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Payload for POST /api/v1/ai/chat."""

    store_id: str = Field(..., description="The store scoped to this request.")
    chat_session_id: str = Field(..., description="Unique identifier for the chat session.")
    query: str = Field(..., min_length=1, description="The business question asked by the user.")
    model_id: Optional[str] = Field(
        default=None,
        description="Optional generation model override. Example: google/gemma-4-4b-it or google/gemma-4-2b-it.",
    )


class EmbedSyncRequest(BaseModel):
    """Payload for POST /api/v1/ai/embed-sync."""

    store_id: str = Field(..., description="The store whose embeddings should be rebuilt.")


# ---------------------------------------------------------------------------
# Response / shape models
# ---------------------------------------------------------------------------

class GroundingInfo(BaseModel):
    """Grounding metadata returned with every AI answer."""

    analytics_used: bool
    alerts_used: list[str] = Field(default_factory=list)
    inventory_products_used: list[str] = Field(default_factory=list)
    rag_products_used: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """A single message entry in chat session history."""

    role: str  # 'user' | 'assistant'
    text: str
    created_at: str
