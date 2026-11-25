"""
search_agent/database.py

Relational and Vector Storage Engine (SQLite + vector column similarity search,
with PostgreSQL + pgvector compatible schema definitions) for:
- users
- projects
- runs
- sources
- findings
- passages (350-word passage chunks with dense/sparse vector embeddings)
"""

import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .semantic_indexer import cosine_similarity


POSTGRES_PGVECTOR_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(user_id),
    name TEXT NOT NULL,
    description TEXT,
    created_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(project_id),
    topic TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_json JSONB,
    report_json JSONB,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    id SERIAL PRIMARY KEY,
    run_id TEXT REFERENCES runs(run_id),
    source_id TEXT NOT NULL,
    title TEXT,
    url TEXT,
    doi TEXT,
    venue TEXT,
    publication_year INTEGER,
    source_type TEXT,
    ranking_score DOUBLE PRECISION,
    content_status TEXT,
    metadata_json JSONB
);
CREATE TABLE IF NOT EXISTS findings (
    id SERIAL PRIMARY KEY,
    run_id TEXT REFERENCES runs(run_id),
    finding_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    claim TEXT NOT NULL,
    stance TEXT NOT NULL,
    confidence TEXT,
    source_ids_json JSONB,
    evidence_json JSONB,
    quantitative_json JSONB
);
CREATE TABLE IF NOT EXISTS passages (
    passage_id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES runs(run_id),
    source_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(128)
);
CREATE INDEX IF NOT EXISTS passages_embedding_hnsw_idx ON passages USING hnsw (embedding vector_cosine_ops);
"""


class ResearchDatabase:
    """
    Relational + Vector database manager for users, projects, runs, sources, findings,
    and 350-word passage embeddings.
    """

    def __init__(self, db_path: str = "data/ara_database.sqlite"):
        self.db_path = db_path
        if db_path != ":memory:":
            dirname = os.path.dirname(db_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                project_id TEXT,
                topic TEXT NOT NULL,
                status TEXT NOT NULL,
                plan_json TEXT,
                report_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            );

            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT,
                url TEXT,
                doi TEXT,
                venue TEXT,
                publication_year INTEGER,
                source_type TEXT,
                ranking_score REAL,
                content_status TEXT,
                metadata_json TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                finding_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                claim TEXT NOT NULL,
                stance TEXT NOT NULL,
                confidence TEXT,
                source_ids_json TEXT,
                evidence_json TEXT,
                quantitative_json TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS passages (
                passage_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sources_run_id ON sources(run_id);
            CREATE INDEX IF NOT EXISTS idx_findings_run_id ON findings(run_id);
            CREATE INDEX IF NOT EXISTS idx_passages_run_id ON passages(run_id);
            """
        )
        self._conn.commit()

    def create_user(self, email: str, name: str = "", user_id: Optional[str] = None) -> str:
        uid = user_id or f"usr_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO users (user_id, email, name, created_at) VALUES (?, ?, ?, ?)",
            (uid, email.strip().lower(), name, now),
        )
        self._conn.commit()
        return uid

    def create_project(self, user_id: str, name: str, description: str = "", project_id: Optional[str] = None) -> str:
        pid = project_id or f"prj_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO projects (project_id, user_id, name, description, created_at) VALUES (?, ?, ?, ?, ?)",
            (pid, user_id, name, description, now),
        )
        self._conn.commit()
        return pid

    def create_run(
        self,
        topic: str,
        project_id: Optional[str] = None,
        status: str = "running",
        plan: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs (run_id, project_id, topic, status, plan_json, report_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (rid, project_id, topic, status, json.dumps(plan or {}), None, now, now),
        )
        self._conn.commit()
        return rid

    def update_run(self, run_id: str, status: str, report: Optional[Dict[str, Any]] = None) -> None:
        now = time.time()
        self._conn.execute(
            "UPDATE runs SET status = ?, report_json = COALESCE(?, report_json), updated_at = ? WHERE run_id = ?",
            (status, json.dumps(report) if report is not None else None, now, run_id),
        )
        self._conn.commit()

    def save_sources(self, run_id: str, sources: List[Dict[str, Any]]) -> int:
        self._conn.execute("DELETE FROM sources WHERE run_id = ?", (run_id,))
        rows = []
        for s in sources:
            rows.append((
                run_id,
                str(s.get("source_id", "")),
                str(s.get("title", "")),
                str(s.get("url", "")),
                s.get("doi"),
                s.get("venue"),
                s.get("publication_year"),
                s.get("source_type", "other"),
                float(s.get("ranking_score", 0.0) or 0.0),
                s.get("content_status", "snippet_only"),
                json.dumps(s),
            ))
        self._conn.executemany(
            """
            INSERT INTO sources (
                run_id, source_id, title, url, doi, venue, publication_year,
                source_type, ranking_score, content_status, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

    def save_findings(self, run_id: str, findings: List[Dict[str, Any]]) -> int:
        self._conn.execute("DELETE FROM findings WHERE run_id = ?", (run_id,))
        rows = []
        for f in findings:
            rows.append((
                run_id,
                str(f.get("finding_id", "")),
                str(f.get("question_id", "")),
                str(f.get("claim", "")),
                str(f.get("stance", "context")),
                str(f.get("confidence", "medium")),
                json.dumps(f.get("source_ids", [])),
                json.dumps(f.get("evidence", [])),
                json.dumps(f.get("quantitative_evidence", [])),
            ))
        self._conn.executemany(
            """
            INSERT INTO findings (
                run_id, finding_id, question_id, claim, stance, confidence,
                source_ids_json, evidence_json, quantitative_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

    def store_passage_embeddings(
        self,
        run_id: str,
        source_id: str,
        passages: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store 350-word passage chunks alongside their vector embeddings for similarity queries.
        """
        rows = []
        for idx, (p, emb) in enumerate(zip(passages, embeddings), start=1):
            pid = f"{run_id}:{source_id}:P{idx}"
            content = str(p.get("content", ""))
            word_count = int(p.get("word_count", len(content.split())))
            rows.append((
                pid,
                run_id,
                source_id,
                idx,
                word_count,
                content,
                json.dumps(list(emb)),
            ))
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO passages (
                passage_id, run_id, source_id, chunk_index, word_count, content, embedding_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()
        return len(rows)

    def search_similar_passages(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search across stored 350-word passage embeddings.
        """
        if run_id:
            cur = self._conn.execute("SELECT * FROM passages WHERE run_id = ?", (run_id,))
        else:
            cur = self._conn.execute("SELECT * FROM passages")

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for row in cur.fetchall():
            vec = json.loads(row["embedding_json"])
            sim = cosine_similarity(query_embedding, vec)
            scored.append((
                sim,
                {
                    "passage_id": row["passage_id"],
                    "run_id": row["run_id"],
                    "source_id": row["source_id"],
                    "chunk_index": row["chunk_index"],
                    "word_count": row["word_count"],
                    "content": row["content"],
                    "similarity": round(sim, 4),
                },
            ))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        sources_rows = self._conn.execute("SELECT * FROM sources WHERE run_id = ?", (run_id,)).fetchall()
        findings_rows = self._conn.execute("SELECT * FROM findings WHERE run_id = ?", (run_id,)).fetchall()
        return {
            "run_id": row["run_id"],
            "project_id": row["project_id"],
            "topic": row["topic"],
            "status": row["status"],
            "plan": json.loads(row["plan_json"]) if row["plan_json"] else {},
            "report": json.loads(row["report_json"]) if row["report_json"] else None,
            "sources": [json.loads(r["metadata_json"]) for r in sources_rows],
            "findings": [
                {
                    "finding_id": r["finding_id"],
                    "question_id": r["question_id"],
                    "claim": r["claim"],
                    "stance": r["stance"],
                    "confidence": r["confidence"],
                    "source_ids": json.loads(r["source_ids_json"] or "[]"),
                    "evidence": json.loads(r["evidence_json"] or "[]"),
                    "quantitative_evidence": json.loads(r["quantitative_json"] or "[]"),
                }
                for r in findings_rows
            ],
        }

    def close(self) -> None:
        self._conn.close()
