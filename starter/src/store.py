"""PostgreSQL + pgvector store: one table, rows keyed by (strategy, chunk_id), exact cosine search."""
from __future__ import annotations

import json
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .chunking import content_hash
from .models import Chunk

# Only these metadata columns may be used in filters (whitelist -> no SQL injection via keys).
FILTER_COLUMNS = {"document_id", "regulation", "part", "chapter", "article", "annex", "year"}


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, table: str = "chunks") -> None:
        if not table.replace("_", "").isalnum():
            raise ValueError("bad table name")
        self.table = table
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.url = database_url
        self.dimensions = dimensions

    def _conn(self):
        return psycopg.connect(self.url, autocommit=True)

    def create_schema(self) -> None:
        with self._conn() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    strategy     text    NOT NULL,
                    chunk_id     text    NOT NULL,
                    document_id  text    NOT NULL,
                    source       text    NOT NULL,
                    page         int     NOT NULL,
                    page_end     int     NOT NULL,
                    regulation   text,
                    year         int,
                    part         text,
                    chapter      text,
                    article      text,
                    annex        text,
                    text         text    NOT NULL,
                    content_sha  text    NOT NULL,
                    metadata     jsonb   NOT NULL DEFAULT '{{}}',
                    embedding    vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            c.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_filter ON {self.table} (strategy, part, regulation)")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Insert/update by stable ID; delete chunks of the same (strategy, document) that disappeared."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"vector has {len(v)} dims, store expects {self.dimensions}")
        stats = {"inserted_or_updated": 0, "deleted_stale": 0}
        by_doc: dict[str, set[str]] = {}
        with self._conn() as c, c.cursor() as cur:
            for ch, v in zip(chunks, vectors):
                m = ch.metadata
                cur.execute(f"""
                    INSERT INTO {self.table} (strategy, chunk_id, document_id, source, page, page_end, regulation, year,
                                        part, chapter, article, annex, text, content_sha, metadata, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id=EXCLUDED.document_id, source=EXCLUDED.source, page=EXCLUDED.page,
                        page_end=EXCLUDED.page_end, regulation=EXCLUDED.regulation, year=EXCLUDED.year,
                        part=EXCLUDED.part, chapter=EXCLUDED.chapter, article=EXCLUDED.article,
                        annex=EXCLUDED.annex, text=EXCLUDED.text, content_sha=EXCLUDED.content_sha,
                        metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding
                    WHERE {self.table}.content_sha <> EXCLUDED.content_sha""",
                    (strategy, ch.chunk_id, ch.document_id, ch.source, ch.page, m.get("page_end", ch.page),
                     m.get("regulation"), m.get("year"), m.get("part"), m.get("chapter"), m.get("article"),
                     m.get("annex"), ch.text, content_hash(ch.text), json.dumps(m, ensure_ascii=False), _vec(v)))
                stats["inserted_or_updated"] += cur.rowcount
                by_doc.setdefault(ch.document_id, set()).add(ch.chunk_id)
            for doc, ids in by_doc.items():
                cur.execute(f"DELETE FROM {self.table} WHERE strategy=%s AND document_id=%s AND NOT (chunk_id = ANY(%s))",
                            (strategy, doc, list(ids)))
                stats["deleted_stale"] += cur.rowcount
        return stats

    def count(self, strategy: str) -> int:
        with self._conn() as c:
            return c.execute(f"SELECT count(*) FROM {self.table} WHERE strategy=%s", (strategy,)).fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter in SQL first, then exact cosine ranking (no ANN index -> no post-filter recall loss)."""
        where, params = ["strategy = %s"], [strategy]
        for key, value in (filters or {}).items():
            if key not in FILTER_COLUMNS:
                raise ValueError(f"filter on '{key}' not allowed")
            where.append(f"{key} = %s")
            params.append(value)
        sql = f"""SELECT chunk_id, document_id, source, page, page_end, regulation, part, chapter, article, annex,
                         text, 1 - (embedding <=> %s::vector) AS score
                  FROM {self.table} WHERE {' AND '.join(where)}
                  ORDER BY embedding <=> %s::vector LIMIT %s"""
        qv = _vec(query_vector)
        with self._conn() as c, c.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, [qv, *params, qv, top_k])
            return [dict(r) for r in cur.fetchall()]
