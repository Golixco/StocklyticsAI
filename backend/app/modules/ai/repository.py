"""
AI Module – Repository layer.

Isolates all Firestore read/write operations for the AI chat module.
The service layer calls only these functions; routes never touch Firestore directly.

Collections:
    ai_chat_sessions        – parent document per chat session
    ai_chat_sessions/{id}/messages – ordered message subcollection
    analytics_metadata      – freshness metadata (read-only by AI module)
    alerts                  – active alerts snapshot (read-only by AI module)
    products                – inventory snapshot (read-only by AI module)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from google.cloud import bigquery
from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentSnapshot

from app.common.config import get_settings
from app.common.google_clients import (
    create_bigquery_client,
    create_firestore_async_client,
    get_default_gcp_project,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Firestore client (lazy singleton)
# ---------------------------------------------------------------------------

_db: Optional[firestore.AsyncClient] = None
_bq: Optional[bigquery.Client] = None


def _get_db() -> firestore.AsyncClient:
    """Return a cached Firestore async client, initialising on first call."""
    global _db
    if _db is None:
        settings = get_settings()
        project = settings.firestore_project_id or None
        _db = create_firestore_async_client(project=project)
    return _db


def _get_bq() -> bigquery.Client:
    """Return a cached BigQuery client, initialising on first call."""
    global _bq
    if _bq is None:
        settings = get_settings()
        project = get_default_gcp_project(settings)
        _bq = create_bigquery_client(project=project)
    return _bq


# ---------------------------------------------------------------------------
# Collection constants
# ---------------------------------------------------------------------------

SESSIONS_COLLECTION = "ai_chat_sessions"
MESSAGES_SUBCOLLECTION = "messages"
ANALYTICS_METADATA_COLLECTION = "analytics_metadata"
ALERTS_COLLECTION = "alerts"
PRODUCTS_COLLECTION = "products"
CUSTOMERS_COLLECTION = "customers"
TRANSACTIONS_COLLECTION = "transactions"


# ---------------------------------------------------------------------------
# Analytics metadata (freshness)
# ---------------------------------------------------------------------------

async def get_analytics_metadata(store_id: str) -> Optional[dict[str, Any]]:
    """
    Read analytics_metadata for the given store.
    Returns None if no metadata document exists yet.
    """
    db = _get_db()
    metadata_id = f"{store_id}_dashboard"
    doc: DocumentSnapshot = await db.collection(ANALYTICS_METADATA_COLLECTION).document(metadata_id).get()
    if not doc.exists:
        return None
    data: dict[str, Any] = doc.to_dict() or {}
    return _normalise_timestamps(data)


# ---------------------------------------------------------------------------
# Context fetchers (read-only consumers from other modules' collections)
# ---------------------------------------------------------------------------

async def get_active_alerts_snapshot(store_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Return up to `limit` ACTIVE alerts for the store to provide context to Gemma.
    Reads directly from the alerts collection (owned by Alerts module).
    """
    db = _get_db()
    query = db.collection(ALERTS_COLLECTION).where("store_id", "==", store_id)
    results: list[dict[str, Any]] = []
    async for doc in query.stream():
        data: dict[str, Any] = doc.to_dict() or {}
        if "alert_id" not in data:
            data["alert_id"] = doc.id
        if data.get("status") == "ACTIVE":
            results.append(data)
        if len(results) >= limit:
            break
    return results


async def get_relevant_alerts_snapshot(store_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Return alerts relevant for AI grounding.

    Active alerts are always preferred. If there is room remaining, recent
    acknowledged alerts are also included so the AI can still reference issues
    that may matter to the current question.
    """
    active_alerts = await get_active_alerts_snapshot(store_id=store_id, limit=limit)
    if len(active_alerts) >= limit:
        return active_alerts

    db = _get_db()
    acknowledged_query = db.collection(ALERTS_COLLECTION).where("store_id", "==", store_id)
    acknowledged_alerts: list[dict[str, Any]] = []
    async for doc in acknowledged_query.stream():
        data: dict[str, Any] = doc.to_dict() or {}
        if "alert_id" not in data:
            data["alert_id"] = doc.id
        if data.get("status") == "ACKNOWLEDGED":
            acknowledged_alerts.append(data)
        if len(acknowledged_alerts) >= limit - len(active_alerts):
            break

    return active_alerts + acknowledged_alerts


async def get_inventory_snapshot(
    store_id: str,
    limit: int = 10,
    product_ids: Optional[list[str]] = None,
    query_text: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Return up to `limit` products for the store to ground the AI in current inventory reality.
    Reads directly from the products collection (owned by Inventory module).
    """
    db = _get_db()
    fetch_limit = max(limit * 5, 25)
    query = (
        db.collection(PRODUCTS_COLLECTION)
        .where("store_id", "==", store_id)
        .limit(fetch_limit)
    )
    results: list[dict[str, Any]] = []
    async for doc in query.stream():
        data: dict[str, Any] = doc.to_dict() or {}
        if "product_id" not in data:
            data["product_id"] = doc.id
        results.append(data)

    requested_ids = set(product_ids or [])
    query_tokens = {
        token
        for token in _normalise_query_tokens(query_text or "")
        if len(token) >= 3
    }

    focused: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for item in results:
        product_id = str(item.get("product_id", ""))
        product_name = str(item.get("name", "")).lower()
        if product_id in requested_ids or any(token in product_name for token in query_tokens):
            focused.append(item)
        else:
            fallback.append(item)

    ordered = focused + fallback
    return ordered[:limit]


async def get_products_by_ids(
    store_id: str,
    product_ids: list[str],
) -> list[dict[str, Any]]:
    """
    Fetch specific products by ID for the given store.

    Used by AI retrieval to hydrate vector matches with full inventory facts.
    Returns only products that belong to the provided store scope.
    """
    if not product_ids:
        return []

    db = _get_db()
    results: list[dict[str, Any]] = []
    for product_id in list(dict.fromkeys(product_ids))[:20]:
        doc: DocumentSnapshot = await db.collection(PRODUCTS_COLLECTION).document(product_id).get()
        if not doc.exists:
            continue
        data: dict[str, Any] = doc.to_dict() or {}
        if str(data.get("store_id", "")) != store_id:
            continue
        data.setdefault("product_id", doc.id)
        results.append(data)
    return _normalise_timestamps_on_list(results)


async def get_customer_snapshot(
    store_id: str,
    limit: int = 10,
    query_text: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Return up to `limit` customers for the store to ground customer-focused answers.
    Reads directly from the customers collection.
    """
    db = _get_db()
    fetch_limit = max(limit * 5, 25)
    query = (
        db.collection(CUSTOMERS_COLLECTION)
        .where("store_id", "==", store_id)
        .limit(fetch_limit)
    )
    results: list[dict[str, Any]] = []
    async for doc in query.stream():
        data: dict[str, Any] = doc.to_dict() or {}
        if "customer_id" not in data:
            data["customer_id"] = doc.id
        results.append(data)

    query_tokens = {
        token
        for token in _normalise_query_tokens(query_text or "")
        if len(token) >= 3
    }

    focused: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for item in results:
        haystacks = (
            str(item.get("customer_id", "")).lower(),
            str(item.get("name", "")).lower(),
            str(item.get("phone", "")).lower(),
        )
        if any(token in haystack for token in query_tokens for haystack in haystacks):
            focused.append(item)
        else:
            fallback.append(item)

    fallback.sort(
        key=lambda item: (
            float(item.get("total_spend", 0) or 0),
            int(item.get("visit_count", 0) or 0),
        ),
        reverse=True,
    )
    ordered = focused + fallback
    return ordered[:limit]


async def get_recent_transactions_snapshot(
    store_id: str,
    limit: int = 10,
    query_text: Optional[str] = None,
    customer_ids: Optional[list[str]] = None,
    product_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """
    Return recent transactions for the store to ground billing and customer questions.
    Reads directly from the transactions collection.
    """
    db = _get_db()
    fetch_limit = max(limit * 5, 25)
    query = (
        db.collection(TRANSACTIONS_COLLECTION)
        .where("store_id", "==", store_id)
        .limit(fetch_limit)
    )
    results: list[dict[str, Any]] = []
    async for doc in query.stream():
        data: dict[str, Any] = doc.to_dict() or {}
        data.setdefault("transaction_id", doc.id)
        results.append(data)

    query_tokens = {
        token
        for token in _normalise_query_tokens(query_text or "")
        if len(token) >= 3
    }
    wanted_customer_ids = set(customer_ids or [])
    wanted_product_ids = set(product_ids or [])

    focused: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for item in results:
        item_customer_id = str(item.get("customer_id", ""))
        item_transaction_id = str(item.get("transaction_id", ""))
        items = item.get("items") or []
        item_product_ids = {
            str(line.get("product_id", ""))
            for line in items
            if line.get("product_id")
        }
        matches = (
            item_customer_id in wanted_customer_ids
            or bool(item_product_ids & wanted_product_ids)
            or any(
                token in item_transaction_id.lower() or token in item_customer_id.lower()
                for token in query_tokens
            )
        )
        if matches:
            focused.append(item)
        else:
            fallback.append(item)

    ordered = focused + sorted(
        fallback,
        key=lambda item: item.get("sale_timestamp") or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    compact: list[dict[str, Any]] = []
    for item in ordered[:limit]:
        compact.append(
            {
                "transaction_id": item.get("transaction_id"),
                "customer_id": item.get("customer_id"),
                "payment_method": item.get("payment_method"),
                "total_amount": item.get("total_amount"),
                "sale_timestamp": item.get("sale_timestamp"),
                "items": [
                    {
                        "product_id": line.get("product_id"),
                        "quantity": line.get("quantity"),
                        "line_total": line.get("line_total"),
                    }
                    for line in (item.get("items") or [])[:5]
                ],
            }
        )
    return _normalise_timestamps_on_list(compact)


# ---------------------------------------------------------------------------
# BigQuery mart readers (analytics context for AI)
# ---------------------------------------------------------------------------

async def get_analytics_context(
    store_id: str,
    include_customer_insights: bool = False,
    product_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Fetch a compact analytics context from BigQuery mart tables.

    The AI service uses this to build the strict `analytics_summary` text field
    required by the AI system design.
    """
    settings = get_settings()
    project = settings.bigquery_project_id
    mart = settings.bigquery_dataset_mart
    if not project:
        raise RuntimeError("BIGQUERY_PROJECT_ID is not configured.")
    bq = _get_bq()

    dashboard_sql = f"""
        SELECT
            snapshot_date,
            today_sales,
            today_transactions,
            active_alert_count,
            low_stock_count,
            top_selling_product,
            analytics_last_updated_at
        FROM `{project}.{mart}.dashboard_summary`
        WHERE store_id = @store_id
        ORDER BY snapshot_date DESC
        LIMIT 1
    """

    sales_sql = f"""
        SELECT
            sales_date,
            total_sales,
            transaction_count,
            average_basket_value
        FROM `{project}.{mart}.sales_daily`
        WHERE store_id = @store_id
        ORDER BY sales_date DESC
        LIMIT @limit
    """

    if product_ids:
        product_sql = f"""
            WITH ranked_products AS (
                SELECT
                    product_id,
                    product_name,
                    quantity_sold,
                    revenue,
                    sales_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY product_id
                        ORDER BY sales_date DESC
                    ) AS row_num
                FROM `{project}.{mart}.product_sales_daily`
                WHERE store_id = @store_id
                  AND product_id IN UNNEST(@product_ids)
            )
            SELECT product_id, product_name, quantity_sold, revenue, sales_date
            FROM ranked_products
            WHERE row_num = 1
            ORDER BY revenue DESC, quantity_sold DESC
            LIMIT @limit
        """
        product_parameters = [
            bigquery.ScalarQueryParameter("store_id", "STRING", store_id),
            bigquery.ArrayQueryParameter("product_ids", "STRING", product_ids),
            bigquery.ScalarQueryParameter("limit", "INT64", 3),
        ]
    else:
        product_sql = f"""
            WITH ranked_products AS (
                SELECT
                    product_id,
                    product_name,
                    quantity_sold,
                    revenue,
                    sales_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY product_id
                        ORDER BY sales_date DESC
                    ) AS row_num
                FROM `{project}.{mart}.product_sales_daily`
                WHERE store_id = @store_id
            )
            SELECT product_id, product_name, quantity_sold, revenue, sales_date
            FROM ranked_products
            WHERE row_num = 1
            ORDER BY revenue DESC, quantity_sold DESC
            LIMIT @limit
        """
        product_parameters = [
            bigquery.ScalarQueryParameter("store_id", "STRING", store_id),
            bigquery.ScalarQueryParameter("limit", "INT64", 3),
        ]

    customer_sql = f"""
        SELECT
            customer_id,
            customer_name,
            lifetime_spend,
            visit_count,
            last_purchase_at
        FROM `{project}.{mart}.customer_summary`
        WHERE store_id = @store_id
        ORDER BY lifetime_spend DESC, visit_count DESC
        LIMIT @limit
    """

    dashboard_rows = await _run_bigquery(
        bq,
        dashboard_sql,
        [
            bigquery.ScalarQueryParameter("store_id", "STRING", store_id),
        ],
    )
    sales_rows = await _run_bigquery(
        bq,
        sales_sql,
        [
            bigquery.ScalarQueryParameter("store_id", "STRING", store_id),
            bigquery.ScalarQueryParameter("limit", "INT64", 7),
        ],
    )
    product_rows = await _run_bigquery(bq, product_sql, product_parameters)

    customer_rows: list[dict[str, Any]] = []
    if include_customer_insights:
        customer_rows = await _run_bigquery(
            bq,
            customer_sql,
            [
                bigquery.ScalarQueryParameter("store_id", "STRING", store_id),
                bigquery.ScalarQueryParameter("limit", "INT64", 3),
            ],
        )

    return {
        "dashboard_summary": dashboard_rows[0] if dashboard_rows else None,
        "sales_trends": sales_rows,
        "product_performance": product_rows,
        "customer_insights": customer_rows,
    }


async def vector_search_products(
    store_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-K semantically relevant products for a query using BigQuery VECTOR_SEARCH.

    The embedding values are inlined into the SQL literal because BigQuery does not
    support parameterized ARRAY<FLOAT64> in the VECTOR_SEARCH sub-select context.
    Float values from our own embedding model carry no injection risk.

    Returns an empty list on any failure — the caller handles degraded context gracefully.

    We intentionally force brute-force mode here so RAG still works before a
    dedicated BigQuery vector index has been created in production.
    """
    settings = get_settings()
    project = settings.bigquery_project_id
    mart = settings.bigquery_dataset_mart
    if not project or not query_embedding:
        return []

    bq = _get_bq()
    # Inline float literals for VECTOR_SEARCH compatibility
    vector_literal = ", ".join(repr(float(v)) for v in query_embedding)

    sql = f"""
        SELECT
            base.product_id,
            base.product_name,
            base.category,
            base.embedding_text,
            distance
        FROM VECTOR_SEARCH(
            (
                SELECT product_id, product_name, category, embedding_text, embedding
                FROM `{project}.{mart}.product_embeddings`
                WHERE store_id = '{store_id}'
            ),
            'embedding',
            (SELECT [{vector_literal}] AS embedding),
            top_k => {int(top_k)},
            distance_type => 'COSINE',
            options => '{{"use_brute_force": true}}'
        )
        ORDER BY distance ASC
    """
    try:
        return await _run_bigquery(bq, sql, [])
    except Exception as exc:
        logger.warning(
            "VECTOR_SEARCH query failed; returning empty RAG results",
            exc_info=exc,
            extra={"store_id": store_id},
        )
        return []


# ---------------------------------------------------------------------------
# Chat session persistence
# ---------------------------------------------------------------------------

async def get_chat_session(chat_session_id: str) -> Optional[dict[str, Any]]:
    """Fetch the session document. Returns None if the session does not exist."""
    db = _get_db()
    doc: DocumentSnapshot = await db.collection(SESSIONS_COLLECTION).document(chat_session_id).get()
    if not doc.exists:
        return None
    return _normalise_timestamps(doc.to_dict() or {})


async def ensure_chat_session(
    chat_session_id: str,
    store_id: str,
    user_id: str,
    updated_at: datetime,
) -> None:
    """
    Create the session parent document if it does not already exist.
    Uses set with merge=True so replays are safe.
    """
    db = _get_db()
    doc_ref = db.collection(SESSIONS_COLLECTION).document(chat_session_id)
    existing_doc = await doc_ref.get()
    if existing_doc.exists:
        await doc_ref.set(
            {
                "last_query_at": updated_at,
                "store_id": store_id,
                "user_id": user_id,
            },
            merge=True,
        )
        return

    await doc_ref.set(
        {
            "chat_session_id": chat_session_id,
            "store_id": store_id,
            "user_id": user_id,
            "created_at": updated_at,
            "last_query_at": updated_at,
        }
    )


async def append_message(
    chat_session_id: str,
    message_id: str,
    role: str,
    text: str,
    created_at: datetime,
) -> None:
    """
    Append a single message document to the session's messages subcollection.

    Args:
        chat_session_id: Parent session identifier.
        message_id:      Unique ID for this message document.
        role:            'user' or 'assistant'.
        text:            The message content.
        created_at:      UTC datetime of the message.
    """
    db = _get_db()
    msg_ref = (
        db.collection(SESSIONS_COLLECTION)
        .document(chat_session_id)
        .collection(MESSAGES_SUBCOLLECTION)
        .document(message_id)
    )
    await msg_ref.set({
        "role": role,
        "text": text,
        "created_at": created_at,
    })
    logger.info(
        "Chat message appended",
        extra={"chat_session_id": chat_session_id, "role": role},
    )


async def list_messages(chat_session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """
    Return the most recent messages for a session, ordered by creation time ascending.

    Args:
        chat_session_id: The session to fetch messages from.
        limit:           Maximum number of messages to return.
    """
    db = _get_db()
    query = (
        db.collection(SESSIONS_COLLECTION)
        .document(chat_session_id)
        .collection(MESSAGES_SUBCOLLECTION)
        .order_by("created_at")
        .limit(limit)
    )
    results: list[dict[str, Any]] = []
    async for doc in query.stream():
        data: dict[str, Any] = doc.to_dict() or {}
        results.append(_normalise_timestamps(data))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_timestamps(data: dict[str, Any]) -> dict[str, Any]:
    """Convert Firestore DatetimeWithNanoseconds values to ISO-8601 UTC strings."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


def _normalise_timestamps_on_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalise_timestamps(item) for item in items]


async def _run_bigquery(
    bq: bigquery.Client,
    sql: str,
    params: list[bigquery.ScalarQueryParameter | bigquery.ArrayQueryParameter],
) -> list[dict[str, Any]]:
    """Execute a BigQuery query off the event loop and normalise rows."""

    def _query() -> list[dict[str, Any]]:
        config = bigquery.QueryJobConfig(query_parameters=params)
        job = bq.query(sql, job_config=config)
        rows = job.result()
        return [_normalise_bigquery_row(dict(row.items())) for row in rows]

    return await asyncio.to_thread(_query)


def _normalise_bigquery_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert BigQuery row values into API-safe Python primitives."""
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            result[key] = value.isoformat()
        elif isinstance(value, date):
            result[key] = value.isoformat()
        elif isinstance(value, Decimal):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def _normalise_query_tokens(query_text: str) -> set[str]:
    return {
        token.strip(".,!?-_/").lower()
        for token in query_text.split()
        if token.strip(".,!?-_/")
    }
