"""
AI Module router.

Owned by: AI Module developer.
Base path: /api/v1/ai

Endpoints:
    POST  /api/v1/ai/chat
    GET   /api/v1/ai/chat/sessions/{chat_session_id}
    POST  /api/v1/ai/embed-sync

Key rules (ai_implementation.md):
    - AI context must come only from: analytics metadata, alerts, inventory snapshot.
    - AI responses must be grounded in available system data.
    - If analytics data is stale, the response must clearly mention it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.common.auth import AuthenticatedUser, require_auth
from app.common.exceptions import ValidationError
from app.common.responses import success_response
from app.modules.ai.schemas import ChatRequest, EmbedSyncRequest
from app.modules.ai.service import AIService

router = APIRouter()
_service = AIService()


@router.post("/chat", status_code=200)
async def post_chat(
    body: ChatRequest,
    user: AuthenticatedUser = Depends(require_auth),
) -> None:
    """
    Answer a business question using structured store data and Gemma.

    - Requires: valid auth token, store_id in body matching the authenticated store scope.
    - Returns: natural language answer with grounding metadata and freshness info.
    - Error 503 AI_CONTEXT_NOT_READY: analytics metadata not available.
    - Error 503 AI_PROVIDER_ERROR: Gemma API call failed.
    """
    if body.store_id != user.store_id:
        raise ValidationError(
            "store_id in request must match authenticated store scope.",
            details={
                "request_store_id": body.store_id,
                "auth_store_id": user.store_id,
            },
        )

    result = await _service.chat(
        store_id=user.store_id,
        user_id=user.user_id,
        chat_session_id=body.chat_session_id,
        query=body.query,
        model_id=body.model_id,
    )
    return success_response(result, status_code=200)


@router.get("/chat/sessions/{chat_session_id}", status_code=200)
async def get_chat_session(
    chat_session_id: str,
    user: AuthenticatedUser = Depends(require_auth),
) -> None:
    """
    Retrieve the full message history for a chat session.

    - Requires: valid auth token.
    - Returns: ordered list of user and assistant messages.
    - Error 404 CHAT_SESSION_NOT_FOUND: the session ID does not exist.
    """
    result = await _service.get_session_history(
        store_id=user.store_id,
        chat_session_id=chat_session_id,
    )
    return success_response(result, status_code=200)


@router.post("/embed-sync", status_code=202)
async def post_embed_sync(
    body: EmbedSyncRequest,
    user: AuthenticatedUser = Depends(require_auth),
) -> None:
    """
    Trigger on-demand product embedding sync for the authenticated store.

    Fetches all products from Firestore and regenerates their vector embeddings
    in BigQuery. Safe to call at any time; replaces only this store's rows.

    - Requires: valid auth token.
    - Returns 202: { embedded: int, product_count: int }
    - Embedding failure returns 500 with error detail.
    """
    if body.store_id != user.store_id:
        raise ValidationError(
            "store_id in request must match authenticated store scope.",
            details={
                "request_store_id": body.store_id,
                "auth_store_id": user.store_id,
            },
        )

    result = await _service.sync_embeddings(store_id=user.store_id)
    return success_response(result, status_code=202)
