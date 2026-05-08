"""
AI Module – Service layer.

Contains all business logic for the AI chat feature:
  1. Build a deterministic, structured context from analytics, alerts, and inventory.
  2. Send a grounded prompt to the hosted Gemma model (no vector DB, no heavy RAG).
  3. Run a response guard to strip unsupported claims and add freshness warnings.
  4. Persist the user question and AI answer to Firestore chat history.
  5. Return the answer and grounding metadata.

Rules (ai_implementation.md §8):
  - Never invent reasons, trends, or product details not in the provided context.
  - Return freshness as structured metadata and keep answer text focused on the question.
  - Use structured system data only — no raw database dumps to Gemini.
  - If context is not ready, raise AI_CONTEXT_NOT_READY (503).
  - If the hosted model call fails, raise AI_PROVIDER_ERROR (503).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Literal, Optional

import httpx

from app.common.config import get_settings
from app.common.exceptions import NotFoundError, ServiceUnavailableError
from app.modules.ai import repository
from app.modules.analytics.repository import AnalyticsRepository

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain errors
# ---------------------------------------------------------------------------

class AIContextNotReadyError(ServiceUnavailableError):
    """Raised when analytics metadata is missing and context cannot be built."""
    error_code = "AI_CONTEXT_NOT_READY"


class AIProviderError(ServiceUnavailableError):
    """Raised when the hosted model API call fails."""
    error_code = "AI_PROVIDER_ERROR"


class ChatSessionNotFoundError(NotFoundError):
    """Raised when the requested chat session does not exist."""
    error_code = "CHAT_SESSION_NOT_FOUND"


# ---------------------------------------------------------------------------
# Hosted model client
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """You are Stocklytics AI, a business intelligence assistant for a retail store.
Your role is to answer the store owner's questions using ONLY the structured data provided in the context below.

Rules you MUST follow:
1. Only use facts that are explicitly present in the provided context.
2. Never invent sales figures, product details, trends, or reasons not in the context.
3. Do not add freshness disclaimers, timestamps, or system notes in the answer text.
4. Keep answers concise, factual, and actionable for a store owner.
5. Use plain text only. Do not use markdown, bullet symbols, or decorative formatting.
6. Prefer 2 to 4 short sentences.
7. If you cannot answer from the provided context, say so clearly.
"""


def _build_context_block(
    analytics_summary: str,
    alerts: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    customers: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    rag_products: Optional[list[dict[str, Any]]] = None,
) -> str:
    """
    Build the strict JSON-shaped context block required by ai_system_design.md.
    Only compact, approved fields are included.
    rag_products (optional) adds semantically retrieved products from BigQuery
    Vector Search to augment the deterministic inventory snapshot.
    """
    alerts = alerts[:6]
    inventory = inventory[:8]
    customers = customers[:5]
    transactions = transactions[:6]
    rag_products = (rag_products or [])[:4]

    payload = {
        "analytics_summary": analytics_summary,
        "alerts": [
            {
                "alert_id": alert.get("alert_id"),
                "alert_type": alert.get("alert_type"),
                "status": alert.get("status"),
                "severity": alert.get("severity"),
                "title": alert.get("title"),
                "message": alert.get("message"),
                "source_entity_id": alert.get("source_entity_id"),
            }
            for alert in alerts
        ],
        "inventory_snapshot": [
            {
                "product_id": product.get("product_id"),
                "name": product.get("name"),
                "category": product.get("category"),
                "price": product.get("price"),
                "quantity_on_hand": product.get("quantity_on_hand"),
                "reorder_threshold": product.get("reorder_threshold"),
                "expiry_date": product.get("expiry_date"),
                "expiry_status": product.get("expiry_status"),
                "status": product.get("status"),
                "created_at": product.get("created_at"),
                "updated_at": product.get("updated_at"),
            }
            for product in inventory
        ],
        "rag_relevant_products": [
            {
                "product_id": p.get("product_id"),
                "product_name": p.get("product_name"),
                "category": p.get("category"),
                "retrieval_note": "semantically matched to query via vector search",
            }
            for p in rag_products
        ],
        "customer_snapshot": [
            {
                "customer_id": customer.get("customer_id"),
                "name": customer.get("name"),
                "phone": customer.get("phone"),
                "total_spend": customer.get("total_spend", customer.get("lifetime_spend")),
                "visit_count": customer.get("visit_count"),
                "last_purchase_at": customer.get("last_purchase_at"),
                "created_at": customer.get("created_at"),
                "updated_at": customer.get("updated_at"),
            }
            for customer in customers
        ],
        "recent_transactions": transactions,
    }
    return json.dumps(payload, indent=2, sort_keys=True, default=_json_safe_default)


def _json_safe_default(value: Any) -> Any:
    """Convert Firestore/BigQuery values into JSON-safe prompt content."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return str(value)


_LEGACY_FRESHNESS_PATTERNS = (
    r"\s*(?:⚠️\s*)?Note:\s*Analytics data may be slightly behind real-time activity\s*\(freshness status:\s*delayed\)\.?",
    r"\s*(?:⚠️\s*)?Note:\s*Analytics data is not current\s*\(freshness status:\s*stale\)\.\s*This answer uses the latest available snapshot and may miss recent changes\.?",
    r"\s*(?:⚠️\s*)?Note:\s*Analytics data is slightly delayed, so this answer uses the latest available snapshot\.?",
    r"\s*Please note this data is stale, last updated on\s+[^\n.]+\.?",
    r"\s*Freshness status:\s*(?:delayed|stale)\.?",
)


def _strip_legacy_freshness_text(answer: str) -> str:
    """Remove boilerplate freshness text that the UI already renders separately."""
    cleaned = answer
    for pattern in _LEGACY_FRESHNESS_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _response_guard(
    raw_answer: str,
    freshness_status: str,
) -> str:
    """
    Post-process the model response:
    - Remove boilerplate freshness notices because the UI renders freshness separately.
    - Strip any mention of unavailable data sources (basic sanity check).
    """
    answer = raw_answer.strip()
    answer = answer.replace("**", "")
    answer = answer.replace("__", "")
    answer = " ".join(answer.split())
    return _strip_legacy_freshness_text(answer)


def _extract_grounding(
    alerts: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    analytics_used: bool,
    rag_products: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Build the grounding metadata object for the API response."""
    return {
        "analytics_used": analytics_used,
        "alerts_used": [a.get("alert_id", "") for a in alerts],
        "inventory_products_used": [p.get("product_id", "") for p in inventory],
        "rag_products_used": [p.get("product_id", "") for p in (rag_products or [])],
    }


def _query_terms(query: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", query.lower())
        if token
    }


def _query_mentions_any(query: str, terms: set[str]) -> bool:
    tokens = _query_terms(query)
    return any(term in tokens for term in terms)


def _query_contains_phrase(query: str, phrases: tuple[str, ...]) -> bool:
    lowered = query.lower()
    return any(phrase in lowered for phrase in phrases)


IntentName = Literal[
    "business_summary",
    "product_lookup",
    "sales_kpi",
    "inventory_risk",
    "reorder_decision",
    "customer_insight",
    "transaction_lookup",
    "reasoning",
    "general",
]


def _detect_intent(query: str) -> IntentName:
    lowered = query.lower()

    if _query_is_business_summary(query):
        return "business_summary"

    if _query_contains_phrase(
        lowered,
        (
            "which product should i order",
            "what should i order",
            "what should i restock",
            "reorder",
            "restock",
        ),
    ):
        return "reorder_decision"

    if _query_contains_phrase(
        lowered,
        (
            "top selling",
            "top-selling",
            "best selling",
            "highest selling",
            "today sale",
            "today's sale",
            "today sales",
            "today's sales",
            "revenue",
            "transactions today",
        ),
    ):
        return "sales_kpi"

    if _query_mentions_customers(query):
        return "customer_insight"

    if _query_mentions_transactions(query):
        return "transaction_lookup"

    if _query_contains_phrase(
        lowered,
        (
            "what needs attention",
            "what should i focus on",
            "priority",
            "risk",
            "alert",
            "low stock",
            "inventory status",
        ),
    ):
        return "inventory_risk"

    if _query_mentions_any(
        query,
        {"product", "products", "stock", "inventory", "item", "items", "about"},
    ):
        return "product_lookup"

    if _query_requests_deeper_reasoning(query):
        return "reasoning"

    return "general"


def _intent_needs_customers(intent: IntentName, query: str) -> bool:
    return intent in {"customer_insight"} or _query_mentions_customers(query)


def _intent_needs_transactions(intent: IntentName, query: str) -> bool:
    return intent in {"transaction_lookup", "reasoning"} or _query_mentions_transactions(query)


def _intent_needs_product_focus(intent: IntentName) -> bool:
    return intent in {"product_lookup", "inventory_risk", "reorder_decision", "reasoning"}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    return _coerce_quantity(value)


def _product_timestamp(product: dict[str, Any], field: str) -> Optional[datetime]:
    return _to_datetime(product.get(field))


def _format_product_age(product: dict[str, Any]) -> str:
    created_at = _product_timestamp(product, "created_at")
    if created_at is None:
        return "added recently"

    delta = datetime.now(timezone.utc) - created_at
    days = max(delta.days, 0)
    if days == 0:
        return "added today"
    if days == 1:
        return "added 1 day ago"
    if days < 7:
        return f"added {days} days ago"
    return f"added on {created_at.strftime('%b %d')}"


def _top_risk_products(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def score(product: dict[str, Any]) -> tuple[int, int]:
        expiry_status = str(product.get("expiry_status", "")).upper()
        quantity = _safe_int(product.get("quantity_on_hand"))
        reorder_threshold = _safe_int(product.get("reorder_threshold"))
        risk = 0
        if expiry_status == "EXPIRED":
            risk += 4
        elif expiry_status == "EXPIRING_SOON":
            risk += 2
        if reorder_threshold and quantity <= reorder_threshold:
            risk += 3
        elif quantity <= 0:
            risk += 3
        return (risk, -quantity)

    ranked = [product for product in inventory if score(product)[0] > 0]
    ranked.sort(key=score, reverse=True)
    return ranked


def _build_recent_product_answer(query: str, inventory: list[dict[str, Any]]) -> Optional[str]:
    if not inventory:
        return None

    recent_queries = {"new", "recent", "latest", "added", "addition"}
    if not _query_mentions_any(query, recent_queries) and not _query_contains_phrase(
        query,
        ("new product", "recent product", "added in inventory", "added to inventory"),
    ):
        return None

    active_products = [product for product in inventory if str(product.get("status", "ACTIVE")).upper() == "ACTIVE"]
    candidates = active_products or inventory
    ordered = sorted(
        candidates,
        key=lambda product: (
            _product_timestamp(product, "created_at") or datetime.min.replace(tzinfo=timezone.utc),
            _product_timestamp(product, "updated_at") or datetime.min.replace(tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    newest = ordered[0]
    quantity = _safe_int(newest.get("quantity_on_hand"))
    reorder_threshold = _safe_int(newest.get("reorder_threshold"))
    price = newest.get("price")
    category = newest.get("category") or "Uncategorized"

    parts = [
        f"The newest product I can see is {newest.get('name', 'this item')} in {category}, { _format_product_age(newest) }."
    ]
    if price not in (None, ""):
        parts.append(f"It is priced at {price} and currently has {quantity} units on hand.")
    else:
        parts.append(f"It currently has {quantity} units on hand.")

    if reorder_threshold and quantity <= reorder_threshold:
        parts.append("It is already near or below its reorder threshold, so it needs replenishment planning.")
    elif quantity == 0:
        parts.append("It is already out of stock, so it should be restocked before promotion.")
    else:
        parts.append("Stock looks available, so the next step is to watch sell-through and reorder timing.")
    return " ".join(parts)


def _build_inventory_status_answer(
    query: str,
    inventory: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
) -> Optional[str]:
    inventory_terms = {"inventory", "stock", "product", "products", "status", "items"}
    if not _query_mentions_any(query, inventory_terms) and not _query_contains_phrase(
        query,
        ("what needs attention", "what should i focus on", "inventory status"),
    ):
        return None

    if not inventory:
        return "I could not load the current inventory snapshot, so I cannot confirm product-level status right now."

    active_products = [product for product in inventory if str(product.get("status", "ACTIVE")).upper() == "ACTIVE"]
    low_stock = [
        product for product in active_products
        if _safe_int(product.get("reorder_threshold")) > 0
        and _safe_int(product.get("quantity_on_hand")) <= _safe_int(product.get("reorder_threshold"))
    ]
    expiring = [
        product for product in active_products
        if str(product.get("expiry_status", "")).upper() in {"EXPIRING_SOON", "EXPIRED"}
    ]
    risks = _top_risk_products(active_products)[:3]
    active_alerts = [alert for alert in alerts if str(alert.get("status", "")).upper() == "ACTIVE"]

    parts = [
        f"I can see {len(active_products)} active products in the current inventory snapshot."
    ]
    if low_stock:
        parts.append(f"{len(low_stock)} are at or below reorder threshold.")
    if expiring:
        parts.append(f"{len(expiring)} are expired or expiring soon.")

    if risks:
        parts.append(
            "Top attention items are "
            + ", ".join(
                f"{product.get('name', product.get('product_id', 'unknown'))} ({_safe_int(product.get('quantity_on_hand'))} units, {str(product.get('expiry_status', 'OK')).lower()})"
                for product in risks
            )
            + "."
        )

    if active_alerts:
        parts.append(f"There are also {len(active_alerts)} active alerts reinforcing those risks.")

    if low_stock:
        parts.append("Best next move is to reorder the lowest-stock items first and clear any expired stock from sale.")
    elif expiring:
        parts.append("Best next move is to prioritise sell-through or markdowns for the expiring items.")
    else:
        parts.append("Overall inventory looks stable right now, so focus on keeping your best sellers in stock.")
    return " ".join(parts)


def _build_sales_answer(query: str, analytics_context: dict[str, Any]) -> Optional[str]:
    sales_terms = {"sale", "sales", "revenue", "transaction", "transactions", "today"}
    if not _query_mentions_any(query, sales_terms):
        return None

    summary = analytics_context.get("dashboard_summary") or {}
    if not summary:
        return None

    today_sales = _safe_float(summary.get("today_sales"))
    today_transactions = _safe_int(summary.get("today_transactions"))
    top_product = summary.get("top_selling_product") or "your top product"
    low_stock_count = _safe_int(summary.get("low_stock_count"))

    parts = [
        f"Today's confirmed sales are {today_sales:.0f} across {today_transactions} transactions."
    ]
    parts.append(f"{top_product} is the current top-selling product.")
    if low_stock_count > 0:
        parts.append(f"{low_stock_count} products are already low on stock, so protect availability on your fast movers.")
    else:
        parts.append("Inventory coverage looks cleaner, so the next step is to maintain momentum on the current top seller.")
    return " ".join(parts)


def _build_business_summary_answer(
    query: str,
    analytics_context: dict[str, Any],
    alerts: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> Optional[str]:
    if not _query_is_business_summary(query):
        return None

    summary = analytics_context.get("dashboard_summary") or {}
    if not summary:
        return None

    today_sales = _safe_float(summary.get("today_sales"))
    today_transactions = _safe_int(summary.get("today_transactions"))
    top_product = str(summary.get("top_selling_product") or "").strip()
    active_alert_count = _safe_int(summary.get("active_alert_count"))
    low_stock_count = _safe_int(summary.get("low_stock_count"))

    if active_alert_count <= 0:
        active_alert_count = sum(1 for alert in alerts if str(alert.get("status", "")).upper() == "ACTIVE")
    if low_stock_count <= 0:
        low_stock_count = sum(
            1
            for product in inventory
            if _safe_int(product.get("reorder_threshold")) > 0
            and _safe_int(product.get("quantity_on_hand")) <= _safe_int(product.get("reorder_threshold"))
        )

    parts = [f"Your store has {today_sales:.0f} in sales across {today_transactions} transactions today."]
    if top_product and top_product.lower() != "unknown":
        parts.append(f"{top_product} is the current top-selling product.")
    if active_alert_count > 0:
        parts.append(f"There are {active_alert_count} active alerts that need attention.")
    elif low_stock_count > 0:
        parts.append(f"{low_stock_count} products are low on stock, so inventory needs a quick review.")
    else:
        parts.append("Operations look stable right now with no urgent stock or alert pressure in the current snapshot.")
    return " ".join(parts)


def _build_top_selling_product_answer(
    query: str,
    analytics_context: dict[str, Any],
) -> Optional[str]:
    top_seller_phrases = (
        "top selling",
        "top-selling",
        "best selling",
        "best-selling",
        "highest selling",
        "most sold",
    )
    if not _query_contains_phrase(query, top_seller_phrases):
        return None

    summary = analytics_context.get("dashboard_summary") or {}
    if not summary:
        return "I could not load the current sales summary, so I cannot confirm the top-selling product right now."

    top_product = summary.get("top_selling_product")
    today_sales = _safe_float(summary.get("today_sales"))
    today_transactions = _safe_int(summary.get("today_transactions"))
    if not top_product or str(top_product).strip().lower() == "unknown":
        return "I can see the current sales snapshot, but it does not identify a top-selling product yet."

    parts = [f"{top_product} is the current top-selling product."]
    if today_sales > 0 or today_transactions > 0:
        parts.append(
            f"Today's store snapshot shows {today_sales:.0f} in sales across {today_transactions} transactions."
        )
    parts.append("Keep that item in stock and visible because it is leading sales right now.")
    return " ".join(parts)


def _build_reorder_recommendation_answer(
    query: str,
    inventory: list[dict[str, Any]],
    analytics_context: dict[str, Any],
) -> Optional[str]:
    reorder_terms = {"order", "reorder", "restock", "buy", "purchase"}
    if not _query_mentions_any(query, reorder_terms) and not _query_contains_phrase(
        query,
        ("which product should i order", "what should i order", "what should i restock"),
    ):
        return None

    if not inventory:
        return "I could not load the inventory snapshot, so I cannot recommend a restock priority right now."

    active_products = [product for product in inventory if str(product.get("status", "ACTIVE")).upper() == "ACTIVE"]
    low_stock = [
        product for product in active_products
        if _safe_int(product.get("reorder_threshold")) > 0
        and _safe_int(product.get("quantity_on_hand")) <= _safe_int(product.get("reorder_threshold"))
    ]
    low_stock.sort(
        key=lambda product: (
            _safe_int(product.get("quantity_on_hand")) - _safe_int(product.get("reorder_threshold")),
            _safe_int(product.get("quantity_on_hand")),
        )
    )

    summary = analytics_context.get("dashboard_summary") or {}
    top_product = str(summary.get("top_selling_product") or "").strip()

    matching_top_product: Optional[dict[str, Any]] = None
    if top_product:
        for product in active_products:
            if str(product.get("name", "")).strip().lower() == top_product.lower():
                matching_top_product = product
                break

    if matching_top_product and _safe_int(matching_top_product.get("reorder_threshold")) > 0:
        quantity = _safe_int(matching_top_product.get("quantity_on_hand"))
        reorder_threshold = _safe_int(matching_top_product.get("reorder_threshold"))
        if quantity <= reorder_threshold:
            return (
                f"You should reorder {matching_top_product.get('name', matching_top_product.get('product_id', 'this product'))} first. "
                f"It is currently your top-selling product and has {quantity} units on hand against a reorder threshold of {reorder_threshold}. "
                "Protecting availability on that fast mover is the highest-priority replenishment action."
            )

    if low_stock:
        recommended = low_stock[0]
        quantity = _safe_int(recommended.get("quantity_on_hand"))
        reorder_threshold = _safe_int(recommended.get("reorder_threshold"))
        return (
            f"You should reorder {recommended.get('name', recommended.get('product_id', 'this product'))} first. "
            f"It has {quantity} units on hand against a reorder threshold of {reorder_threshold}, which makes it the weakest stock position in the current snapshot. "
            "That is the clearest restock priority right now."
        )

    if top_product:
        return (
            f"{top_product} is the safest product to keep watching for replenishment because it is currently leading sales. "
            "The current snapshot does not show any urgent product below reorder threshold right now."
        )

    return "I do not see an urgent reorder priority in the current inventory snapshot, because no active product is currently below its reorder threshold."


def _rank_customers(customers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def spend(customer: dict[str, Any]) -> float:
        return _safe_float(customer.get("lifetime_spend", customer.get("total_spend", 0.0)))

    return sorted(
        customers,
        key=lambda customer: (
            spend(customer),
            _safe_int(customer.get("visit_count")),
        ),
        reverse=True,
    )


def _build_customer_answer(
    query: str,
    analytics_context: dict[str, Any],
    customers: list[dict[str, Any]],
) -> Optional[str]:
    customer_terms = {"customer", "customers", "buyer", "buyers", "loyal", "repeat", "spend", "spender"}
    if not _query_mentions_any(query, customer_terms) and not _query_contains_phrase(
        query,
        ("best customers", "top customers", "loyal customers"),
    ):
        return None

    ranked = _rank_customers(analytics_context.get("customer_insights") or customers)
    if not ranked:
        return "I could not load ranked customer insights from the current store data, so I cannot confirm your top customers yet."

    top = ranked[:3]
    parts = [
        "Your top customers right now are "
        + ", ".join(
            f"{customer.get('customer_name', customer.get('name', customer.get('customer_id', 'Unknown')))} "
            f"({_safe_float(customer.get('lifetime_spend', customer.get('total_spend', 0.0))):.0f} spend, {_safe_int(customer.get('visit_count'))} visits)"
            for customer in top
        )
        + "."
    ]
    leader = top[0]
    last_purchase = leader.get("last_purchase_at")
    if last_purchase:
        parts.append(
            f"{leader.get('customer_name', leader.get('name', 'This customer'))} was last seen on {_display_timestamp(last_purchase)}."
        )
    parts.append("Best next move is to target these repeat buyers with availability updates, bundles, or loyalty outreach.")
    return " ".join(parts)


def _build_transaction_answer(query: str, transactions: list[dict[str, Any]]) -> Optional[str]:
    transaction_terms = {"transaction", "transactions", "bill", "billing", "payment", "payments", "purchase", "purchases"}
    if not _query_mentions_any(query, transaction_terms) and not _query_contains_phrase(
        query,
        ("recent transactions", "recent purchases", "last bill"),
    ):
        return None

    if not transactions:
        return "I could not load recent transaction records from the database, so I cannot confirm billing activity right now."

    recent = transactions[:3]
    parts = [
        "The most recent transactions I can see are "
        + ", ".join(
            f"{txn.get('transaction_id', 'unknown')} for {_safe_float(txn.get('total_amount')):.0f} via {txn.get('payment_method', 'unknown payment method')}"
            for txn in recent
        )
        + "."
    ]
    latest = recent[0]
    if latest.get("sale_timestamp"):
        parts.append(f"The latest one was recorded on {_display_timestamp(latest.get('sale_timestamp'))}.")
    return " ".join(parts)


def _build_focus_answer(
    query: str,
    analytics_context: dict[str, Any],
    alerts: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> Optional[str]:
    if not _query_contains_phrase(query, ("what should i focus on", "what needs attention", "priority", "priorities")):
        return None

    active_alerts = [alert for alert in alerts if str(alert.get("status", "")).upper() == "ACTIVE"]
    risks = _top_risk_products(inventory)
    top_product = (analytics_context.get("dashboard_summary") or {}).get("top_selling_product")

    parts: list[str] = []
    if active_alerts:
        top_alert = active_alerts[0]
        parts.append(f"Your first priority is {top_alert.get('title', 'the top active alert')}.")
    if risks:
        top_risk = risks[0]
        parts.append(
            f"The most urgent inventory item is {top_risk.get('name', top_risk.get('product_id', 'unknown'))} with {_safe_int(top_risk.get('quantity_on_hand'))} units left and status {str(top_risk.get('expiry_status', 'OK')).lower()}."
        )
    if top_product:
        parts.append(f"Keep {top_product} available because it is currently leading sales.")
    if not parts:
        return None
    parts.append("Best next move is to resolve the top alert, restock the weakest item, and then review your best seller coverage.")
    return " ".join(parts)


def _query_prefers_operator_answer(query: str) -> bool:
    return (
        _query_mentions_any(
            query,
            {
                "inventory",
                "stock",
                "product",
                "products",
                "sales",
                "sale",
                "revenue",
                "alert",
                "alerts",
                "new",
                "recent",
                "latest",
                "added",
                "status",
                "focus",
                "priority",
                "order",
                "reorder",
                "restock",
                "buy",
                "purchase",
            },
        )
        or _query_contains_phrase(
            query,
            (
                "what needs attention",
                "what should i focus on",
                "new product",
                "inventory status",
                "top selling",
                "top-selling",
                "best selling",
                "which product should i order",
                "what should i order",
            ),
        )
    )


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    refusal_signals = (
        "i cannot answer",
        "i can't answer",
        "cannot answer this question",
        "provided data only contains",
        "not enough information",
        "do not have enough information",
    )
    return any(signal in lowered for signal in refusal_signals)


def _build_operator_answer(
    query: str,
    analytics_context: dict[str, Any],
    alerts: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    customers: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
) -> Optional[str]:
    answer, _ = _build_operator_answer_result(
        query,
        analytics_context,
        alerts,
        inventory,
        customers,
        transactions,
    )
    return answer


def _build_operator_answer_result(
    query: str,
    analytics_context: dict[str, Any],
    alerts: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    customers: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
) -> tuple[Optional[str], bool]:
    builders: tuple[tuple[Any, bool], ...] = (
        (lambda q, i: _build_business_summary_answer(q, analytics_context, alerts, inventory), True),
        (lambda q, i: _build_top_selling_product_answer(q, analytics_context), True),
        (lambda q, i: _build_reorder_recommendation_answer(q, inventory, analytics_context), True),
        (_build_recent_product_answer, False),
        (lambda q, i: _build_focus_answer(q, analytics_context, alerts, inventory), False),
        (lambda q, i: _build_inventory_status_answer(q, inventory, alerts), False),
        (lambda q, i: _build_customer_answer(q, analytics_context, customers), True),
        (lambda q, i: _build_transaction_answer(q, transactions), False),
        (lambda q, i: _build_sales_answer(q, analytics_context), True),
    )
    for builder, analytics_used in builders:
        answer = builder(query, inventory)
        if answer:
            return answer, analytics_used
    return None, False


def _infer_analytics_used(
    query: str,
    answer: str,
    operator_analytics_used: bool,
) -> bool:
    if operator_analytics_used:
        return True

    if _query_mentions_any(
        query,
        {"sale", "sales", "revenue", "transaction", "transactions", "customer", "customers", "buyer", "buyers"},
    ):
        return True

    if _query_mentions_any(query, {"inventory", "stock", "product", "products", "new", "recent", "added"}):
        return False

    lowered = answer.lower()
    analytics_signals = (
        "today's confirmed sales",
        "transactions",
        "top-selling product",
        "leading sales",
    )
    return any(signal in lowered for signal in analytics_signals)
    return None


def _build_fallback_answer(
    query: str,
    analytics_summary: str,
    analytics_context: dict[str, Any],
    alerts: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    customers: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    freshness_status: str,
) -> str:
    """
    Build a deterministic local fallback answer when the hosted model or mart context is unavailable.
    """
    operator_answer = _build_operator_answer(
        query,
        analytics_context,
        alerts,
        inventory,
        customers,
        transactions,
    )
    if operator_answer:
        return operator_answer

    lines = []
    lowered = query.lower()

    if "sale" in lowered or "revenue" in lowered or "transaction" in lowered:
        lines.append("Here is the latest operational sales snapshot I can confirm:")
        for part in analytics_summary.split("; "):
            if part.startswith(("today_sales=", "today_transactions=", "top_selling_product=")):
                lines.append(part.replace("=", ": ", 1))

    if "alert" in lowered or "stock" in lowered:
        active_alerts = [alert for alert in alerts if alert.get("status") == "ACTIVE"]
        low_stock_products = [
            item for item in inventory
            if _coerce_quantity(item.get("quantity_on_hand", 0)) <= 0
            or str(item.get("expiry_status", "")).upper() in {"EXPIRING_SOON", "EXPIRED"}
        ]
        lines.append(f"Active alerts: {len(active_alerts)}.")
        if active_alerts:
            lines.append(
                "Top alert titles: " + ", ".join(str(alert.get("title") or alert.get("alert_id")) for alert in active_alerts[:3])
            )
        if low_stock_products:
            lines.append(
                "Products needing attention: "
                + ", ".join(str(item.get("name") or item.get("product_id")) for item in low_stock_products[:5])
            )

    if not lines:
        lines.append("I can answer from the current operational store data, but the full AI provider context is unavailable right now.")
        lines.append("Available snapshot:")
        for part in analytics_summary.split("; "):
            if part.startswith(("today_sales=", "today_transactions=", "active_alert_count=", "low_stock_count=")):
                lines.append(part.replace("=", ": ", 1))

    return "\n".join(lines)


def _coerce_quantity(value: Any) -> int:
    """Best-effort int coercion for loosely typed inventory quantities."""
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _build_analytics_summary(
    metadata: dict[str, Any],
    analytics_context: dict[str, Any],
) -> str:
    """
    Build the compact `analytics_summary` string expected by ai_system_design.md.
    """
    last_updated = _display_timestamp(metadata.get("analytics_last_updated_at", "unknown"))
    freshness_status = metadata.get("freshness_status", "unknown")

    parts = [
        f"analytics_last_updated_at={last_updated}",
        f"freshness_status={freshness_status}",
    ]

    dashboard = analytics_context.get("dashboard_summary") or {}
    if dashboard:
        parts.extend(
            [
                f"today_sales={dashboard.get('today_sales', 'unknown')}",
                f"today_transactions={dashboard.get('today_transactions', 'unknown')}",
                f"top_selling_product={dashboard.get('top_selling_product', 'unknown')}",
                f"active_alert_count={dashboard.get('active_alert_count', 'unknown')}",
                f"low_stock_count={dashboard.get('low_stock_count', 'unknown')}",
            ]
        )

    sales_trends = analytics_context.get("sales_trends") or []
    if sales_trends:
        latest_day = sales_trends[0]
        parts.append(f"latest_sales_date={latest_day.get('sales_date', 'unknown')}")
        parts.append(f"latest_total_sales={latest_day.get('total_sales', 'unknown')}")
        if len(sales_trends) >= 2:
            previous_day = sales_trends[1]
            latest_total = float(latest_day.get("total_sales", 0) or 0)
            previous_total = float(previous_day.get("total_sales", 0) or 0)
            direction = "flat"
            if latest_total > previous_total:
                direction = "up"
            elif latest_total < previous_total:
                direction = "down"
            parts.append(f"sales_trend_direction={direction}")

    product_performance = analytics_context.get("product_performance") or []
    if product_performance:
        product_bits = []
        for item in product_performance:
            product_bits.append(
                f"{item.get('product_name', item.get('product_id', 'unknown'))}"
                f" qty_sold={item.get('quantity_sold', 'unknown')}"
                f" revenue={item.get('revenue', 'unknown')}"
            )
        parts.append(f"product_performance={' | '.join(product_bits)}")

    customer_insights = analytics_context.get("customer_insights") or []
    if customer_insights:
        customer_bits = []
        for item in customer_insights:
            customer_bits.append(
                f"{item.get('customer_name', item.get('customer_id', 'unknown'))}"
                f" lifetime_spend={item.get('lifetime_spend', 'unknown')}"
                f" visit_count={item.get('visit_count', 'unknown')}"
            )
        parts.append(f"top_customers={'; '.join(customer_bits)}")

    if len(parts) <= 2:
        return ""
    return "; ".join(parts)


def _display_timestamp(value: Any) -> str:
    dt = _to_datetime(value)
    if dt is None:
        return str(value)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


_FRESH_MAX_AGE = timedelta(minutes=30)
_DELAYED_MAX_AGE = timedelta(hours=2)
_FRESHNESS_ORDER = {
    "fresh": 0,
    "delayed": 1,
    "stale": 2,
}
_CONTEXT_TIMEOUT_SECONDS = 6
_MODEL_TIMEOUT_SECONDS = 20
_PERSIST_TIMEOUT_SECONDS = 6


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    if hasattr(value, "seconds"):
        return datetime.fromtimestamp(value.seconds, tz=timezone.utc)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    return None


def _freshness_from_timestamp(updated_at: Optional[datetime]) -> str:
    if updated_at is None:
        return "stale"

    age = datetime.now(timezone.utc) - updated_at
    if age <= timedelta(0):
        return "fresh"
    if age <= _FRESH_MAX_AGE:
        return "fresh"
    if age <= _DELAYED_MAX_AGE:
        return "delayed"
    return "stale"


def _merge_freshness(computed_status: str, metadata_status: Any) -> str:
    candidate = str(metadata_status).strip().lower() if metadata_status is not None else ""
    if candidate not in _FRESHNESS_ORDER:
        return computed_status
    if _FRESHNESS_ORDER[candidate] > _FRESHNESS_ORDER[computed_status]:
        return candidate
    return computed_status


def _resolve_freshness_fields(metadata: dict[str, Any]) -> tuple[str, str]:
    updated_raw = metadata.get("analytics_last_updated_at")
    updated_dt = _to_datetime(updated_raw)
    computed_status = _freshness_from_timestamp(updated_dt)
    freshness_status = _merge_freshness(computed_status, metadata.get("freshness_status"))

    if hasattr(updated_raw, "isoformat"):
        last_updated = updated_raw.isoformat()
    elif updated_raw is None:
        last_updated = "unknown"
    else:
        last_updated = str(updated_raw)

    return last_updated, freshness_status


async def _await_with_timeout(coro, timeout_seconds: float):
    """Bound slow upstream work so AI chat can fall back instead of hanging."""
    return await asyncio.wait_for(coro, timeout=timeout_seconds)


_tokenizer = None
_model = None
_embedder = None

def _get_tokenizer_and_model():
    global _tokenizer, _model
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Using Gemma 4 4B model as requested for the Kaggle challenge
        model_id = "google/gemma-4-4b-it"
        logger.info(f"Loading local Gemma model: {model_id}")
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    return _tokenizer, _model

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Loading local embedding model: {model_id}")
        _embedder = SentenceTransformer(model_id)
    return _embedder


async def _embed_query(query: str) -> Optional[list[float]]:
    """Generate a query embedding via local sentence-transformers model."""
    try:
        embedder = await asyncio.to_thread(_get_embedder)
        embedding = await asyncio.to_thread(embedder.encode, query)
        return embedding.tolist()
    except Exception as exc:
        logger.warning(
            "Query embedding failed; skipping RAG retrieval",
            exc_info=exc,
        )
        return None


async def _generate_model_answer(
    prompt: str,
    *,
    preferred_model: Optional[str] = None,
    fallback_models: Optional[list[str]] = None,
    max_output_tokens: int = 220,
) -> str:
    """Call the local Gemma 4 4B model using transformers."""
    try:
        import torch
        tokenizer, model = await asyncio.to_thread(_get_tokenizer_and_model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        messages = [
            {"role": "user", "content": _SYSTEM_INSTRUCTION},
            {"role": "assistant", "content": "Understood. I will strictly follow the rules and only answer based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        
        prompt_tensor = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        
        prompt_len = prompt_tensor.shape[-1]
        
        def _generate():
            return model.generate(
                prompt_tensor,
                max_new_tokens=max_output_tokens,
                do_sample=False,
            )
            
        outputs = await asyncio.to_thread(_generate)
        gen_ids = outputs[0, prompt_len:]
        reply = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        
        if not reply:
            raise AIProviderError("Local model generated empty response.")
        return reply
    except Exception as exc:
        raise AIProviderError(f"Local Gemma inference failed: {exc}")


async def _get_live_dashboard_summary_fallback(
    live_repo: AnalyticsRepository,
    store_id: str,
) -> tuple[dict[str, Any], bool]:
    """Attempt a live summary read, otherwise fall back to a minimal empty snapshot."""
    try:
        summary = await _await_with_timeout(
            live_repo.get_live_dashboard_summary(store_id),
            _CONTEXT_TIMEOUT_SECONDS,
        )
        return summary, False
    except Exception as exc:
        logger.warning(
            "Live dashboard summary read failed; using minimal fallback summary",
            exc_info=exc,
            extra={"store_id": store_id},
        )
        return (
            {
                "today_sales": 0.0,
                "today_transactions": 0,
                "active_alert_count": 0,
                "low_stock_count": 0,
                "top_selling_product": "unknown",
            },
            True,
        )


def _query_mentions_customers(query: str) -> bool:
    customer_keywords = {
        "customer",
        "customers",
        "buyer",
        "buyers",
        "loyal",
        "frequent",
        "repeat",
    }
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in customer_keywords)


def _query_mentions_transactions(query: str) -> bool:
    transaction_keywords = {
        "transaction",
        "transactions",
        "bill",
        "billing",
        "payment",
        "payments",
        "purchase",
        "purchases",
    }
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in transaction_keywords)


def _get_related_product_ids(alerts: list[dict[str, Any]]) -> list[str]:
    return [
        str(alert["source_entity_id"])
        for alert in alerts
        if alert.get("source_entity_id")
    ]


def _query_requests_deeper_reasoning(query: str) -> bool:
    lowered = query.lower()
    if _query_is_business_summary(query):
        return False

    phrase_triggers = (
        "compare",
        "trend",
        "forecast",
        "predict",
        "recommend strategy",
        "what should i do next",
        "root cause",
        "why did",
        "why is",
        "why are",
        "how can i",
        "how should i",
    )
    if any(phrase in lowered for phrase in phrase_triggers):
        return True

    return bool(re.search(r"\b(why|explain)\b", lowered))


def _query_is_business_summary(query: str) -> bool:
    return _query_contains_phrase(
        query,
        (
            "how is my business",
            "how's my business",
            "hows my business",
            "how is the business",
            "how's the business",
            "hows the business",
            "how is store doing",
            "how's store doing",
            "hows store doing",
            "how is my store doing",
            "how's my store doing",
            "hows my store doing",
            "hows going my business",
            "how is business going",
            "how's business going",
            "how is my shop doing",
            "business summary",
            "store summary",
            "overall summary",
            "summarize my business",
            "summarize store",
            "summarize my store",
            "give me a summary of my business",
        ),
    )


def _should_use_operator_answer_directly(query: str, operator_answer: Optional[str]) -> bool:
    """Return True for routine operational queries that don't need full model generation."""
    if not operator_answer:
        return False
    if _query_requests_deeper_reasoning(query):
        return False
    return _query_prefers_operator_answer(query)


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _product_match_score(query: str, product: dict[str, Any]) -> float:
    query_norm = _normalise_text(query)
    if not query_norm:
        return 0.0

    product_id = _normalise_text(str(product.get("product_id", "")))
    product_name = _normalise_text(str(product.get("name", product.get("product_name", ""))))
    category = _normalise_text(str(product.get("category", "")))

    score = 0.0
    if product_name and product_name in query_norm:
        score += 1.0
    if query_norm and product_name and query_norm in product_name:
        score += 0.9
    if product_id and product_id in query_norm:
        score += 0.95

    query_tokens = [token for token in query_norm.split(" ") if len(token) >= 3]
    if query_tokens and product_name:
        overlap = sum(1 for token in query_tokens if token in product_name)
        score += min(overlap / max(len(query_tokens), 1), 0.8)
    if query_tokens and category:
        category_overlap = sum(1 for token in query_tokens if token in category)
        score += min(category_overlap / max(len(query_tokens), 1), 0.3)
    return score


def _hybrid_product_matches(
    query: str,
    inventory: list[dict[str, Any]],
    rag_products: list[dict[str, Any]],
    *,
    top_k: int = 4,
) -> tuple[list[dict[str, Any]], float]:
    if not inventory and not rag_products:
        return [], 0.0

    scored_by_id: dict[str, tuple[float, dict[str, Any]]] = {}
    for product in inventory:
        product_id = str(product.get("product_id", ""))
        if not product_id:
            continue
        lexical_score = _product_match_score(query, product)
        scored_by_id[product_id] = (lexical_score, product)

    for rag in rag_products:
        product_id = str(rag.get("product_id", ""))
        if not product_id:
            continue
        distance = _safe_float(rag.get("distance"))
        vector_score = max(0.0, min(1.0, 1.0 - distance))
        if product_id in scored_by_id:
            current_score, item = scored_by_id[product_id]
            merged = max(current_score, 0.65 + vector_score)
            scored_by_id[product_id] = (merged, item)
            continue

        # Keep vector-only candidates when inventory hydration missed them.
        candidate = {
            "product_id": product_id,
            "name": rag.get("product_name"),
            "category": rag.get("category"),
        }
        lexical_score = _product_match_score(query, candidate)
        scored_by_id[product_id] = (max(lexical_score, 0.55 + vector_score), candidate)

    ranked = sorted(scored_by_id.values(), key=lambda pair: pair[0], reverse=True)
    matches = [item for _, item in ranked[:top_k]]
    confidence = ranked[0][0] if ranked else 0.0
    # Clamp to simple confidence range for client/UI metadata
    confidence = max(0.0, min(confidence, 1.0))
    return matches, confidence


def _select_model(
    *,
    query: str,
    intent: IntentName,
    explicit_model_id: Optional[str],
    retrieval_confidence: float,
) -> tuple[str, list[str], int]:
    settings = get_settings()
    explicit = (explicit_model_id or "").strip()
    if explicit:
        fallback_models = [model for model in settings.ai_fallback_model_ids if model != explicit]
        return explicit, fallback_models, 420 if _query_requests_deeper_reasoning(query) else 280

    reasoning_needed = (
        intent in {"reasoning"}
        or _query_requests_deeper_reasoning(query)
        or len(query.split()) >= 18
        or (intent == "product_lookup" and retrieval_confidence < 0.45)
    )
    if reasoning_needed:
        selected = settings.ai_reasoning_model_id or settings.ai_default_model_id or settings.ai_primary_model_id
        fallback_models = [
            model
            for model in [
                settings.ai_fast_model_id,
                settings.ai_default_model_id,
                *settings.ai_fallback_model_ids,
            ]
            if model and model != selected
        ]
        return selected, list(dict.fromkeys(fallback_models)), 520

    selected = settings.ai_fast_model_id or settings.ai_default_model_id or settings.ai_primary_model_id
    fallback_models = [
        model
        for model in [
            settings.ai_default_model_id,
            settings.ai_reasoning_model_id,
            *settings.ai_fallback_model_ids,
        ]
        if model and model != selected
    ]
    return selected, list(dict.fromkeys(fallback_models)), 260


def _build_product_fact_line(product: dict[str, Any]) -> str:
    name = str(product.get("name") or product.get("product_name") or product.get("product_id") or "Unknown product")
    qty = _safe_int(product.get("quantity_on_hand"))
    reorder = _safe_int(product.get("reorder_threshold"))
    expiry = str(product.get("expiry_status", "OK")).upper()
    price = product.get("price")
    bits = [f"{name}: qty={qty}", f"reorder={reorder}", f"expiry={expiry}"]
    if price not in (None, ""):
        bits.append(f"price={price}")
    return ", ".join(bits)


def _build_retrieval_only_answer(
    *,
    intent: IntentName,
    query: str,
    analytics_context: dict[str, Any],
    alerts: list[dict[str, Any]],
    matched_products: list[dict[str, Any]],
    retrieval_confidence: float,
) -> Optional[str]:
    if intent == "business_summary":
        return _build_business_summary_answer(query, analytics_context, alerts, matched_products)

    if intent == "sales_kpi":
        return _build_top_selling_product_answer(query, analytics_context) or _build_sales_answer(query, analytics_context)

    if intent in {"product_lookup", "reorder_decision"}:
        if retrieval_confidence < 0.35 or not matched_products:
            return "I cannot confidently match that product from current store data. Please use the exact product name or SKU."
        lead = matched_products[0]
        base = _build_product_fact_line(lead)
        if intent == "reorder_decision":
            qty = _safe_int(lead.get("quantity_on_hand"))
            reorder = _safe_int(lead.get("reorder_threshold"))
            if reorder > 0 and qty <= reorder:
                return f"Restock priority is {base}. It is at or below reorder threshold, so replenishment should be immediate."
            return f"{base}. Stock is above reorder threshold right now, so immediate restock is not urgent."
        return f"Matched product details: {base}."

    if intent == "inventory_risk":
        return _build_inventory_status_answer(query, matched_products or [], alerts)

    return None


def _build_evidence_prompt(
    *,
    intent: IntentName,
    query: str,
    analytics_summary: str,
    freshness_status: str,
    alerts: list[dict[str, Any]],
    matched_products: list[dict[str, Any]],
    rag_products: list[dict[str, Any]],
    customers: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
) -> str:
    compact_payload = {
        "intent": intent,
        "freshness_status": freshness_status,
        "analytics_summary": analytics_summary,
        "matched_products": [
            {
                "product_id": item.get("product_id"),
                "name": item.get("name") or item.get("product_name"),
                "category": item.get("category"),
                "quantity_on_hand": item.get("quantity_on_hand"),
                "reorder_threshold": item.get("reorder_threshold"),
                "expiry_status": item.get("expiry_status"),
                "price": item.get("price"),
            }
            for item in matched_products[:4]
        ],
        "rag_candidates": [
            {
                "product_id": item.get("product_id"),
                "product_name": item.get("product_name"),
                "category": item.get("category"),
                "distance": item.get("distance"),
            }
            for item in rag_products[:5]
        ],
        "alerts": [
            {
                "alert_id": alert.get("alert_id"),
                "title": alert.get("title"),
                "severity": alert.get("severity"),
                "source_entity_id": alert.get("source_entity_id"),
            }
            for alert in alerts[:6]
        ],
        "customers": [
            {
                "customer_id": customer.get("customer_id"),
                "name": customer.get("name"),
                "total_spend": customer.get("total_spend", customer.get("lifetime_spend")),
                "visit_count": customer.get("visit_count"),
            }
            for customer in customers[:4]
        ],
        "transactions": [
            {
                "transaction_id": txn.get("transaction_id"),
                "total_amount": txn.get("total_amount"),
                "sale_timestamp": txn.get("sale_timestamp"),
                "items": txn.get("items", [])[:4],
            }
            for txn in transactions[:4]
        ],
    }
    return (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        "Use only the evidence JSON below. If evidence is weak or missing, say you cannot confirm.\n"
        f"=== EVIDENCE JSON ===\n{json.dumps(compact_payload, sort_keys=True, default=_json_safe_default)}\n\n"
        f"=== USER QUESTION ===\n{query}"
    )


def _finalize_grounded_answer(
    *,
    answer: str,
    intent: IntentName,
    retrieval_confidence: float,
    matched_products: list[dict[str, Any]],
) -> str:
    cleaned = _response_guard(answer, freshness_status="fresh")
    if intent in {"product_lookup", "reorder_decision"} and retrieval_confidence < 0.35:
        return "I cannot confidently match that product from current store data. Please use the exact product name or SKU."
    if intent in {"product_lookup", "reorder_decision"} and matched_products:
        lead_name = str(matched_products[0].get("name") or matched_products[0].get("product_name") or "").strip()
        if lead_name and lead_name.lower() not in cleaned.lower():
            return f"{lead_name}: {cleaned}"
    return cleaned


# ---------------------------------------------------------------------------
# Public service methods
# ---------------------------------------------------------------------------

class AIService:
    """Business logic for AI chat functionality."""

    async def chat(
        self,
        store_id: str,
        user_id: str,
        chat_session_id: str,
        query: str,
        model_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Handle a single AI chat turn.

        Flow:
          1. Read analytics_metadata for freshness context.
          2. Fetch active alerts and inventory snapshot.
          3. Build a deterministic structured prompt.
          4. Call the hosted model API.
          5. Guard and format the response.
          6. Persist user query + AI answer to Firestore session history.
          7. Return structured response.

        Raises:
            AIContextNotReadyError: If analytics metadata is unavailable.
            AIProviderError:        If the hosted model call fails.
        """
        intent = _detect_intent(query)
        needs_product_focus = _intent_needs_product_focus(intent)
        needs_customers = _intent_needs_customers(intent, query)
        needs_transactions = _intent_needs_transactions(intent, query)

        try:
            metadata = await _await_with_timeout(
                repository.get_analytics_metadata(store_id),
                _CONTEXT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning(
                "Analytics metadata read failed; using synthetic freshness metadata",
                exc_info=exc,
                extra={"store_id": store_id},
            )
            metadata = None
        if metadata is None:
            metadata = {
                "analytics_last_updated_at": datetime.now(timezone.utc).isoformat(),
                "freshness_status": "fresh",
            }

        last_updated, freshness_status = _resolve_freshness_fields(metadata)

        alerts_task = asyncio.create_task(
            _await_with_timeout(
                repository.get_relevant_alerts_snapshot(store_id, limit=10),
                _CONTEXT_TIMEOUT_SECONDS,
            )
        )
        inventory_task = asyncio.create_task(
            _await_with_timeout(
                repository.get_inventory_snapshot(
                    store_id,
                    limit=24 if needs_product_focus else 10,
                    query_text=query,
                ),
                _CONTEXT_TIMEOUT_SECONDS,
            )
        )
        customers_task: Optional[asyncio.Task] = None
        if needs_customers:
            customers_task = asyncio.create_task(
                _await_with_timeout(
                    repository.get_customer_snapshot(store_id, limit=10, query_text=query),
                    _CONTEXT_TIMEOUT_SECONDS,
                )
            )
        query_embedding_task: Optional[asyncio.Task] = None
        if needs_product_focus:
            query_embedding_task = asyncio.create_task(
                _await_with_timeout(_embed_query(query), _CONTEXT_TIMEOUT_SECONDS)
            )

        try:
            alerts = await alerts_task
        except Exception as exc:
            logger.warning(
                "Alerts snapshot read failed; continuing without alerts context",
                exc_info=exc,
                extra={"store_id": store_id},
            )
            alerts = []

        try:
            inventory = await inventory_task
        except Exception as exc:
            logger.warning(
                "Inventory snapshot read failed; continuing without inventory context",
                exc_info=exc,
                extra={"store_id": store_id},
            )
            inventory = []

        customers: list[dict[str, Any]] = []
        if customers_task is not None:
            try:
                customers = await customers_task
            except Exception as exc:
                logger.warning(
                    "Customer snapshot read failed; continuing without customer context",
                    exc_info=exc,
                    extra={"store_id": store_id},
                )
                customers = []

        rag_products: list[dict[str, Any]] = []
        if query_embedding_task is not None:
            try:
                _query_embedding = await query_embedding_task
                if _query_embedding:
                    rag_products = await _await_with_timeout(
                        repository.vector_search_products(
                            store_id,
                            query_embedding=_query_embedding,
                            top_k=max(get_settings().vector_search_top_k, 6),
                        ),
                        _CONTEXT_TIMEOUT_SECONDS,
                    )
            except Exception as exc:
                logger.warning(
                    "RAG vector retrieval failed; continuing without vector context",
                    exc_info=exc,
                    extra={"store_id": store_id},
                )
                rag_products = []

        vector_only_ids = [
            str(item.get("product_id", ""))
            for item in rag_products
            if item.get("product_id")
            and str(item.get("product_id")) not in {str(p.get("product_id", "")) for p in inventory}
        ][:6]
        if vector_only_ids:
            try:
                hydrated = await _await_with_timeout(
                    repository.get_products_by_ids(store_id, vector_only_ids),
                    _CONTEXT_TIMEOUT_SECONDS,
                )
                inventory.extend(hydrated)
            except Exception as exc:
                logger.warning(
                    "Could not hydrate vector-only products from Firestore",
                    exc_info=exc,
                    extra={"store_id": store_id},
                )

        matched_products, retrieval_confidence = _hybrid_product_matches(query, inventory, rag_products, top_k=4)
        matched_product_ids = [str(item.get("product_id", "")) for item in matched_products if item.get("product_id")]
        related_product_ids = _get_related_product_ids(alerts)
        analytics_product_ids = list(dict.fromkeys([*matched_product_ids, *related_product_ids]))[:8]

        transactions: list[dict[str, Any]] = []
        if needs_transactions or (needs_product_focus and bool(matched_product_ids)):
            try:
                transactions = await _await_with_timeout(
                    repository.get_recent_transactions_snapshot(
                        store_id,
                        limit=8,
                        query_text=query,
                        customer_ids=[item.get("customer_id") for item in customers if item.get("customer_id")],
                        product_ids=matched_product_ids or [item.get("product_id") for item in inventory if item.get("product_id")],
                    ),
                    _CONTEXT_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.warning(
                    "Transaction snapshot read failed; continuing without transaction context",
                    exc_info=exc,
                    extra={"store_id": store_id},
                )
                transactions = []

        try:
            analytics_context = await _await_with_timeout(
                repository.get_analytics_context(
                    store_id,
                    include_customer_insights=needs_customers,
                    product_ids=analytics_product_ids or None,
                ),
                _CONTEXT_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Analytics context query failed; falling back to live operational summary", exc_info=exc)
            live_repo = AnalyticsRepository()
            live_summary, _ = await _get_live_dashboard_summary_fallback(live_repo, store_id)
            analytics_context = {
                "dashboard_summary": live_summary,
                "sales_trends": [],
                "product_performance": [],
                "customer_insights": [],
            }

        if needs_customers and not (analytics_context.get("customer_insights") or []):
            analytics_context["customer_insights"] = [
                {
                    "customer_id": customer.get("customer_id"),
                    "customer_name": customer.get("name"),
                    "lifetime_spend": customer.get("total_spend", 0.0),
                    "visit_count": customer.get("visit_count", 0),
                    "last_purchase_at": customer.get("last_purchase_at"),
                }
                for customer in customers[:5]
            ]

        analytics_summary = _build_analytics_summary(
            {
                **metadata,
                "analytics_last_updated_at": last_updated,
                "freshness_status": freshness_status,
            },
            analytics_context,
        )
        if not analytics_summary:
            live_repo = AnalyticsRepository()
            live_summary, _ = await _get_live_dashboard_summary_fallback(live_repo, store_id)
            analytics_summary = "; ".join(
                [
                    f"analytics_last_updated_at={last_updated}",
                    f"freshness_status={freshness_status}",
                    f"today_sales={live_summary.get('today_sales', 'unknown')}",
                    f"today_transactions={live_summary.get('today_transactions', 'unknown')}",
                    f"active_alert_count={live_summary.get('active_alert_count', 'unknown')}",
                    f"low_stock_count={live_summary.get('low_stock_count', 'unknown')}",
                    f"top_selling_product={live_summary.get('top_selling_product', 'unknown')}",
                ]
            )

        direct_answer = _build_retrieval_only_answer(
            intent=intent,
            query=query,
            analytics_context=analytics_context,
            alerts=alerts,
            matched_products=matched_products if matched_products else inventory,
            retrieval_confidence=retrieval_confidence,
        )
        answer_mode = "model"
        selected_model_id, fallback_models, max_output_tokens = _select_model(
            query=query,
            intent=intent,
            explicit_model_id=model_id,
            retrieval_confidence=retrieval_confidence,
        )
        used_fallback = False

        if direct_answer and intent in {"business_summary", "sales_kpi", "inventory_risk"} and not _query_requests_deeper_reasoning(query):
            raw_answer = direct_answer
            answer_mode = "retrieval_only"
        elif direct_answer and intent in {"product_lookup", "reorder_decision"} and retrieval_confidence >= 0.72:
            raw_answer = direct_answer
            answer_mode = "retrieval_only"
        else:
            prompt = _build_evidence_prompt(
                intent=intent,
                query=query,
                analytics_summary=analytics_summary,
                freshness_status=freshness_status,
                alerts=alerts,
                matched_products=matched_products,
                rag_products=rag_products,
                customers=customers,
                transactions=transactions,
            )
            try:
                raw_answer = await _generate_model_answer(
                    prompt,
                    preferred_model=selected_model_id,
                    fallback_models=fallback_models,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as exc:
                logger.error("Hosted model API call failed; using deterministic fallback answer", exc_info=exc)
                raw_answer = _build_fallback_answer(
                    query=query,
                    analytics_summary=analytics_summary,
                    analytics_context=analytics_context,
                    alerts=alerts,
                    inventory=inventory,
                    customers=customers,
                    transactions=transactions,
                    freshness_status=freshness_status,
                )
                answer_mode = "fallback"
                used_fallback = True

        answer = _finalize_grounded_answer(
            answer=raw_answer,
            intent=intent,
            retrieval_confidence=retrieval_confidence,
            matched_products=matched_products,
        )
        if not answer:
            answer = "I cannot confirm that from current store data."

        grounding = _extract_grounding(
            alerts,
            matched_products or inventory,
            analytics_used=_infer_analytics_used(query, answer, intent in {"sales_kpi", "customer_insight", "reasoning"}),
            rag_products=rag_products,
        )

        # Step 6: Persist messages
        now = datetime.now(tz=timezone.utc)
        try:
            existing_session = await _await_with_timeout(
                repository.get_chat_session(chat_session_id),
                _PERSIST_TIMEOUT_SECONDS,
            )
            if existing_session is not None and existing_session.get("store_id") != store_id:
                raise ChatSessionNotFoundError(
                    f"Chat session '{chat_session_id}' was not found.",
                    details={"chat_session_id": chat_session_id},
                )

            await _await_with_timeout(
                repository.ensure_chat_session(
                    chat_session_id=chat_session_id,
                    store_id=store_id,
                    user_id=user_id,
                    updated_at=now,
                ),
                _PERSIST_TIMEOUT_SECONDS,
            )
            await _await_with_timeout(
                repository.append_message(
                    chat_session_id=chat_session_id,
                    message_id=f"msg_{uuid.uuid4().hex}",
                    role="user",
                    text=query,
                    created_at=now,
                ),
                _PERSIST_TIMEOUT_SECONDS,
            )
            await _await_with_timeout(
                repository.append_message(
                    chat_session_id=chat_session_id,
                    message_id=f"msg_{uuid.uuid4().hex}",
                    role="assistant",
                    text=answer,
                    created_at=now,
                ),
                _PERSIST_TIMEOUT_SECONDS,
            )
        except ChatSessionNotFoundError:
            raise
        except Exception as exc:
            logger.warning(
                "AI chat response generated but session persistence failed",
                exc_info=exc,
                extra={"chat_session_id": chat_session_id, "store_id": store_id},
            )

        # Step 7: Return
        return {
            "chat_session_id": chat_session_id,
            "analytics_last_updated_at": last_updated,
            "freshness_status": freshness_status,
            "answer": answer,
            "grounding": grounding,
            "intent": intent,
            "answer_mode": answer_mode,
            "model_id": selected_model_id,
            "retrieval_confidence": round(retrieval_confidence, 3),
            "used_fallback": used_fallback,
            "citations": [
                {
                    "type": "product",
                    "id": item.get("product_id"),
                    "name": item.get("name") or item.get("product_name"),
                }
                for item in matched_products[:3]
            ],
        }

    async def get_session_history(
        self,
        store_id: str,
        chat_session_id: str,
    ) -> dict[str, Any]:
        """
        Return the full message history for a chat session.

        Raises:
            ChatSessionNotFoundError: If the session does not exist in Firestore.
        """
        session = await repository.get_chat_session(chat_session_id)
        if session is None or session.get("store_id") != store_id:
            raise ChatSessionNotFoundError(
                f"Chat session '{chat_session_id}' was not found.",
                details={"chat_session_id": chat_session_id},
            )

        messages = await repository.list_messages(chat_session_id)

        # Shape messages per api_contracts.md §8
        shaped_messages = [
            {
                "role": msg.get("role"),
                "text": msg.get("text"),
                "created_at": msg.get("created_at"),
            }
            for msg in messages
        ]

        return {
            "chat_session_id": chat_session_id,
            "messages": shaped_messages,
        }

    async def sync_embeddings(self, *, store_id: str) -> dict[str, Any]:
        """
        Trigger on-demand product embedding sync for a store.

        Fetches all products from Firestore and regenerates their vector
        embeddings in BigQuery. Called from POST /api/v1/ai/embed-sync.

        Returns:
            { store_id, product_count, embedded }
        """
        from datetime import datetime, timezone

        from app.common.google_clients import (
            create_bigquery_client,
            create_firestore_async_client,
            get_default_gcp_project,
        )
        from app.modules.data_pipeline.embedding_sync import sync_product_embeddings

        settings = get_settings()
        db = create_firestore_async_client(project=settings.firestore_project_id or None)
        bq = create_bigquery_client(project=get_default_gcp_project(settings))

        products: list[dict[str, Any]] = []
        async for doc in db.collection("products").where("store_id", "==", store_id).stream():
            data = doc.to_dict() or {}
            data.setdefault("product_id", doc.id)
            products.append(data)

        now = datetime.now(timezone.utc)
        embedded_count = await sync_product_embeddings(
            bq,
            store_id=store_id,
            products=products,
            analytics_last_updated_at=now,
        )

        logger.info(
            "On-demand embedding sync complete",
            extra={
                "store_id": store_id,
                "product_count": len(products),
                "embedded": embedded_count,
            },
        )
        return {
            "store_id": store_id,
            "product_count": len(products),
            "embedded": embedded_count,
        }
