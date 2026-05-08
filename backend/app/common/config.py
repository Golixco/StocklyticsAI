"""
Application configuration loaded from environment variables.
Central place for all config values. Import this module wherever settings are needed.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.common.logging_config import configure_logging


_BACKEND_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_BACKEND_ENV_PATH)


class Settings:
    """Holds all application configuration values."""

    app_env: str = os.getenv("APP_ENV", "local")
    api_port: int = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    cors_allow_origins_raw: str = os.getenv("CORS_ALLOW_ORIGINS", "")

    # Firebase / Auth
    firebase_project_id: str = os.getenv("FIREBASE_PROJECT_ID", "")
    firebase_client_email: str = os.getenv("FIREBASE_CLIENT_EMAIL", "")
    firebase_private_key: str = os.getenv("FIREBASE_PRIVATE_KEY", "")

    # Firestore
    firestore_project_id: str = os.getenv("FIRESTORE_PROJECT_ID", "")

    # BigQuery
    bigquery_project_id: str = os.getenv("BIGQUERY_PROJECT_ID", "")
    bigquery_dataset_raw: str = os.getenv("BIGQUERY_DATASET_RAW", "stocklytics_raw")
    bigquery_dataset_mart: str = os.getenv("BIGQUERY_DATASET_MART", "stocklytics_mart")

    # Gemma AI (Local)
    gemma_model_id: str = os.getenv("GEMMA_MODEL_ID", "google/gemma-4-4b-it")
    gemma_embedding_model: str = os.getenv("GEMMA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    vector_search_top_k: int = int(os.getenv("VECTOR_SEARCH_TOP_K", "5"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
    demo_auth_bypass_enabled: bool = os.getenv("DEMO_AUTH_BYPASS_ENABLED", "false").lower() == "true"

    @property
    def is_local(self) -> bool:
        return self.app_env == "local"

    @property
    def cors_allow_origins(self) -> list[str]:
        if not self.cors_allow_origins_raw.strip():
            return []
        return [
            origin.strip()
            for origin in self.cors_allow_origins_raw.split(",")
            if origin.strip()
        ]



    def validate_runtime(self) -> None:
        """Fail fast on invalid production configuration."""
        if self.is_local:
            return

        missing: list[str] = []
        required_pairs: list[tuple[str, Any]] = [
            ("FIREBASE_PROJECT_ID", self.firebase_project_id),
            ("FIRESTORE_PROJECT_ID", self.firestore_project_id),
            ("BIGQUERY_PROJECT_ID", self.bigquery_project_id),
        ]
        for name, value in required_pairs:
            if not str(value).strip():
                missing.append(name)

        if missing:
            raise RuntimeError(
                "Missing required production environment variables: "
                + ", ".join(sorted(missing))
            )

        if not self.cors_allow_origins:
            raise RuntimeError(
                "CORS_ALLOW_ORIGINS must be configured in production."
            )



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    settings = Settings()
    settings.validate_runtime()
    return settings


def setup_logging() -> None:
    """
    Backward-compatible logging bootstrap for scripts.

    Some Cloud Run job entrypoints import setup_logging from this module.
    Keep this helper as a thin wrapper around the shared logging config.
    """
    configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
