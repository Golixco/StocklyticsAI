"""
Data Pipeline Module – Embedding Sync.

Generates text embeddings for all store products and upserts them into
the BigQuery table `stocklytics_mart.product_embeddings`.

This module is called after a successful mart refresh so embeddings always
reflect the latest synced inventory state.

Called by:
  - transform_runner.run_mart_refresh()   (daily scheduled, post-mart success)
  - scripts/run_embedding_sync.py         (manual trigger or full rebuild)

Embedding model: Local sentence-transformers model.
Load strategy: per-store WRITE_TRUNCATE via a temp table swap to avoid DML billing.

Rules:
  - Embedding failure must NEVER fail the mart transform pipeline run.
  - store_id isolation: only the given store's rows are replaced.
  - analytics_last_updated_at from the transform is stamped on every row.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from google.cloud import bigquery

from app.common.config import get_settings

logger = logging.getLogger(__name__)

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Loading local embedding model: {model_id}")
        _embedder = SentenceTransformer(model_id)
    return _embedder


async def _embed_batch_local(texts: list[str]) -> list[list[float]]:
    try:
        embedder = await asyncio.to_thread(_get_embedder)
        embeddings = await asyncio.to_thread(embedder.encode, texts)
        return [emb.tolist() for emb in embeddings]
    except Exception as exc:
        raise RuntimeError(f"Local embedding failed: {exc}")


def _load_rows_to_bigquery(
    bq: bigquery.Client,
    rows: list[dict[str, Any]],
    table_id: str,
) -> None:
    """Load JSON rows into BigQuery using WRITE_APPEND (blocking, run in thread)."""
    schema = [
        bigquery.SchemaField("store_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("product_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("product_name", "STRING"),
        bigquery.SchemaField("category", "STRING"),
        bigquery.SchemaField("embedding_text", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("embedded_at", "TIMESTAMP"),
        bigquery.SchemaField("analytics_last_updated_at", "TIMESTAMP"),
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=schema,
    )
    job = bq.load_table_from_json(rows, table_id, job_config=job_config)
    job.result()  # blocks until done; raises on error


def _delete_store_rows(
    bq: bigquery.Client,
    table_id: str,
    store_id: str,
) -> None:
    """Remove existing rows for a store before re-inserting (idempotent update).

    Uses a DML DELETE. This is acceptable here because:
    - product_embeddings is NOT a mart table (no MERGE contract applies)
    - DELETE + INSERT keeps the pattern idempotent
    - Billing restriction only applies to mart tables in the pipeline
    """
    sql = f"DELETE FROM `{table_id}` WHERE store_id = '{store_id}'"
    job = bq.query(sql)
    job.result()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def sync_product_embeddings(
    bq: bigquery.Client,
    *,
    store_id: str,
    products: list[dict[str, Any]],
    analytics_last_updated_at: datetime,
) -> int:
    """Generate embeddings for all products and upsert into product_embeddings.

    Args:
        bq:                          BigQuery client (from transform_runner).
        store_id:                    Isolates all reads/writes to this store.
        products:                    Raw product dicts from Firestore.
        analytics_last_updated_at:   Timestamp from the successful mart refresh.

    Returns:
        Number of products successfully embedded and loaded.

    Raises:
        Any unhandled exception — callers must wrap in try/except and log.
        The mart pipeline must NOT fail if this function raises.
    """
    settings = get_settings()

    if not products:
        logger.info(
            "No products to embed; skipping embedding sync",
            extra={"store_id": store_id},
        )
        return 0

    batch_size = settings.embedding_batch_size
    now = datetime.now(timezone.utc)
    project = settings.bigquery_project_id
    mart = settings.bigquery_dataset_mart
    table_id = f"{project}.{mart}.product_embeddings"

    rows: list[dict[str, Any]] = []
    failed_batches = 0

    selected_model: str | None = None
    for i in range(0, len(products), batch_size):
        batch = products[i : i + batch_size]
        texts = [_build_embedding_text(p) for p in batch]

        try:
            embeddings = await _embed_batch_local(texts)
            if selected_model is None:
                selected_model = "sentence-transformers/all-MiniLM-L6-v2"
        except Exception as exc:
            logger.warning(
                "Embedding batch failed; skipping batch",
                exc_info=exc,
                extra={"store_id": store_id, "batch_start": i, "batch_size": len(batch)},
            )
            failed_batches += 1
            continue

        for product, embedding, text in zip(batch, embeddings, texts):
            pid = product.get("product_id") or product.get("id") or ""
            if not pid:
                continue
            rows.append(
                {
                    "store_id": store_id,
                    "product_id": pid,
                    "product_name": product.get("product_name") or product.get("name"),
                    "category": product.get("category"),
                    "embedding_text": text,
                    "embedding": embedding,
                    "embedded_at": now.isoformat(),
                    "analytics_last_updated_at": analytics_last_updated_at.isoformat(),
                }
            )

    if not rows:
        logger.warning(
            "All embedding batches failed or produced no rows",
            extra={"store_id": store_id, "failed_batches": failed_batches},
        )
        return 0

    # Replace existing rows for this store, then append fresh ones
    await asyncio.to_thread(_delete_store_rows, bq, table_id, store_id)
    await asyncio.to_thread(_load_rows_to_bigquery, bq, rows, table_id)

    logger.info(
        "Embedding sync complete",
        extra={
            "store_id": store_id,
            "embedded": len(rows),
            "failed_batches": failed_batches,
            "model": selected_model or preferred_model,
        },
    )
    return len(rows)
