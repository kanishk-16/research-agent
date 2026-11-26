"""
search_agent/cache_manager.py

Multi-Tier SHA-256 Disk/SQLite Cache:
1. Search Query Cache: Keyed by sha256(provider + query_text) with a 14-30 day TTL (default 21 days).
2. Parsed PDF Cache: Keyed by sha256(doi or arxiv_id or url) to eliminate repeated PDF downloads/parsing.
3. Embedding Cache: Keyed by sha256(passage_text) so text embeddings are never computed twice.
"""

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, List, Optional


DEFAULT_SEARCH_TTL_SECONDS = 21 * 86400  # 21 days (within 14-30 day window)
DEFAULT_PDF_TTL_SECONDS = 90 * 86400     # 90 days for immutable academic PDFs
DEFAULT_EMBEDDING_TTL_SECONDS = 180 * 86400


def compute_sha256_key(*parts: str) -> str:
    """Compute a deterministic SHA-256 hex digest from normalized string parts."""
    joined = "::".join(str(p or "").strip() for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class MultiTierCache:
    """
    Persistent SQLite-backed multi-tier cache supporting Search Queries, Parsed PDFs,
    and Passage Embeddings with per-entry TTL expiration.
    """

    def __init__(self, cache_db_path: str = "data/ara_cache.sqlite"):
        self.cache_db_path = cache_db_path
        if cache_db_path != ":memory:":
            dirname = os.path.dirname(cache_db_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        self._conn = sqlite3.connect(cache_db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                tier TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (tier, cache_key)
            )
            """
        )
        self._conn.commit()

    def _get(self, tier: str, cache_key: str) -> Optional[Any]:
        now = time.time()
        row = self._conn.execute(
            "SELECT value_json, expires_at FROM cache_entries WHERE tier = ? AND cache_key = ?",
            (tier, cache_key),
        ).fetchone()
        if not row:
            return None
        value_json, expires_at = row
        if expires_at < now:
            self._conn.execute(
                "DELETE FROM cache_entries WHERE tier = ? AND cache_key = ?",
                (tier, cache_key),
            )
            self._conn.commit()
            return None
        try:
            return json.loads(value_json)
        except Exception:
            return None

    def _set(self, tier: str, cache_key: str, value: Any, ttl_seconds: float) -> str:
        now = time.time()
        expires_at = now + float(ttl_seconds)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache_entries (tier, cache_key, value_json, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (tier, cache_key, json.dumps(value), expires_at, now),
        )
        self._conn.commit()
        return cache_key

    # ------------------------------------------------------------------
    # Tier 1: Search Query Cache (sha256(provider + query_text), 14-30d TTL)
    # ------------------------------------------------------------------
    @staticmethod
    def search_cache_key(provider: str, query_text: str) -> str:
        return compute_sha256_key(provider.lower(), query_text.lower())

    def get_search_results(self, provider: str, query_text: str) -> Optional[List[dict]]:
        key = self.search_cache_key(provider, query_text)
        return self._get("search_query", key)

    def set_search_results(
        self,
        provider: str,
        query_text: str,
        results: List[dict],
        ttl_seconds: float = DEFAULT_SEARCH_TTL_SECONDS,
    ) -> str:
        key = self.search_cache_key(provider, query_text)
        return self._set("search_query", key, results, ttl_seconds)

    # ------------------------------------------------------------------
    # Tier 2: Parsed PDF Cache (sha256(doi or arxiv_id or url))
    # ------------------------------------------------------------------
    @staticmethod
    def pdf_cache_key(identifier: str) -> str:
        return compute_sha256_key(identifier.lower())

    def get_parsed_pdf(self, identifier: str) -> Optional[str]:
        key = self.pdf_cache_key(identifier)
        return self._get("parsed_pdf", key)

    def set_parsed_pdf(
        self,
        identifier: str,
        extracted_text: str,
        ttl_seconds: float = DEFAULT_PDF_TTL_SECONDS,
    ) -> str:
        key = self.pdf_cache_key(identifier)
        return self._set("parsed_pdf", key, extracted_text, ttl_seconds)

    # ------------------------------------------------------------------
    # Tier 3: Embedding Cache (sha256(passage_text))
    # ------------------------------------------------------------------
    @staticmethod
    def embedding_cache_key(passage_text: str) -> str:
        return compute_sha256_key(passage_text)

    def get_embedding(self, passage_text: str) -> Optional[List[float]]:
        key = self.embedding_cache_key(passage_text)
        return self._get("embedding", key)

    def set_embedding(
        self,
        passage_text: str,
        vector: List[float],
        ttl_seconds: float = DEFAULT_EMBEDDING_TTL_SECONDS,
    ) -> str:
        key = self.embedding_cache_key(passage_text)
        return self._set("embedding", key, list(vector), ttl_seconds)

    def close(self) -> None:
        self._conn.close()
