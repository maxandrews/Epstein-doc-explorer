#!/usr/bin/env python3
"""
MCP Server for Epstein Document RAG.

Provides semantic search + keyword search over the document corpus,
with RAG capabilities for answering questions.

Uses PGVector for efficient vector similarity search (PostgreSQL)
or fallback to numpy calculations (SQLite).

Features:
- PGVector semantic search with IVFFlat index (via materialized view)
- PostgreSQL full-text search with tsvector + GIN index
- MRR (Mean Reciprocal Rank) scoring
- Hybrid search combining semantic + keyword + MRR
"""

import os
import sqlite3
import json
import time
import numpy as np
import httpx
from pathlib import Path
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")
SQLITE_PATH = Path(__file__).parent / "document_analysis.db"
MODEL_NAME = "sentence-transformers/all-minilm-l6-v2"
EMBEDDING_DIM = 384

# OpenRouter API configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"

# Initialize MCP server
mcp = FastMCP("epstein-docs")

# Stats cache (1 week TTL)
_stats_cache = {"data": None, "timestamp": 0}
STATS_CACHE_TTL = 7 * 24 * 60 * 60  # 1 week in seconds

# Embeddings cache (1 hour TTL, max 1000 entries)
_embeddings_cache: dict[str, tuple[np.ndarray, float]] = {}
EMBEDDINGS_CACHE_TTL = 60 * 60  # 1 hour
EMBEDDINGS_CACHE_MAX_SIZE = 1000

def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Get embeddings via OpenRouter API with caching.

    Args:
        texts: List of texts to embed

    Returns:
        numpy array of embeddings (shape: [n_texts, 384])
    """
    global _embeddings_cache
    now = time.time()

    # Check cache for single text queries (most common case)
    if len(texts) == 1:
        cache_key = texts[0].strip().lower()
        if cache_key in _embeddings_cache:
            cached_embedding, cached_time = _embeddings_cache[cache_key]
            if now - cached_time < EMBEDDINGS_CACHE_TTL:
                return cached_embedding.reshape(1, -1)
            else:
                del _embeddings_cache[cache_key]

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://epstein-agent-api.local",
    }

    payload = {
        "model": MODEL_NAME,
        "input": texts
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            OPENROUTER_EMBEDDING_URL,
            headers=headers,
            json=payload
        )
        response.raise_for_status()

    result = response.json()

    # Extract embeddings from response
    embeddings = [item["embedding"] for item in result["data"]]
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Cache single text queries
    if len(texts) == 1:
        cache_key = texts[0].strip().lower()
        # Evict old entries if cache is full
        if len(_embeddings_cache) >= EMBEDDINGS_CACHE_MAX_SIZE:
            oldest_key = min(_embeddings_cache, key=lambda k: _embeddings_cache[k][1])
            del _embeddings_cache[oldest_key]
        _embeddings_cache[cache_key] = (embeddings_array[0], now)

    return embeddings_array


_pg_pool = None


def _init_pg_pool():
    """Initialize PostgreSQL connection pool."""
    global _pg_pool
    if _pg_pool is None and DATABASE_URL:
        import psycopg2.pool
        _pg_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=DATABASE_URL
        )
    return _pg_pool


class PooledConnection:
    """Wrapper that returns connection to pool on close()."""

    def __init__(self, pool, conn):
        self._pool = pool
        self._conn = conn
        import psycopg2.extras
        self._conn.cursor_factory = psycopg2.extras.RealDictCursor
        # Set IVFFlat probes for better semantic search quality
        with self._conn.cursor() as cur:
            cur.execute("SET ivfflat.probes = 10")

    def cursor(self):
        return self._conn.cursor()

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        """Return connection to pool instead of closing."""
        if self._pool and self._conn:
            self._pool.putconn(self._conn)
            self._conn = None


def get_db():
    """Get database connection (SQLite or PostgreSQL)."""
    if DATABASE_URL:
        return _get_postgres_conn()
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def _get_postgres_conn():
    """Get PostgreSQL connection from pool."""
    pool = _init_pg_pool()
    if pool:
        conn = pool.getconn()
        return PooledConnection(pool, conn)
    else:
        # Fallback si pas de pool
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        # Set IVFFlat probes for better semantic search quality
        with conn.cursor() as cur:
            cur.execute("SET ivfflat.probes = 10")
        return conn


def _is_postgres():
    """Check if using PostgreSQL."""
    return DATABASE_URL is not None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors (fallback for SQLite)."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ==================== PGVector Semantic Search ====================

def semantic_search_pgvector(query: str, limit: int = 10) -> list[dict]:
    """
    Search documents using PGVector for efficient similarity search.

    Uses HNSW index with cosine distance for fast vector similarity.

    Args:
        query: Natural language search query
        limit: Maximum number of results

    Returns:
        List of documents with similarity scores (0-1)
    """
    conn = get_db()

    # Embed the query via OpenRouter API
    query_embedding = get_embeddings([query])[0]
    query_vector_str = str(query_embedding.tolist())

    cursor = conn.cursor()

    # Use PGVector cosine distance
    # cosine distance = 1 - cosine similarity
    cursor.execute("""
        SELECT doc_id, paragraph_summary, one_sentence_summary,
               category, date_range_earliest, date_range_latest,
               1 - (embedding_vector <=> %s::vector) as similarity
        FROM all_embeddings_mv
        WHERE embedding_vector IS NOT NULL
        ORDER BY embedding_vector <=> %s::vector
        LIMIT %s
    """, [query_vector_str, query_vector_str, limit])

    results = []
    for row in cursor:
        results.append({
            "doc_id": row["doc_id"],
            "similarity": float(row["similarity"]),
            "summary": row["paragraph_summary"] or row["one_sentence_summary"],
            "category": row["category"],
            "date_range": f"{row['date_range_earliest'] or '?'} - {row['date_range_latest'] or '?'}"
        })

    cursor.close()
    conn.close()

    return results


def semantic_search_fallback(query: str, limit: int = 10) -> list[dict]:
    """
    Fallback semantic search using numpy for SQLite.

    Note: This loads all embeddings and calculates similarity in Python.
    Not recommended for large datasets - use PGVector instead.
    """
    conn = get_db()

    # Embed the query via OpenRouter API
    query_embedding = get_embeddings([query])[0]

    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.doc_id, e.embedding, d.paragraph_summary, d.one_sentence_summary,
               d.category, d.date_range_earliest, d.date_range_latest
        FROM all_document_embeddings e
        JOIN all_documents d ON e.doc_id = d.doc_id
    """)

    results = []
    for row in cursor:
        row_dict = dict(row) if hasattr(row, 'keys') else row

        # Decode embedding from blob
        embedding_data = row_dict["embedding"]
        if isinstance(embedding_data, memoryview):
            embedding_data = bytes(embedding_data)
        doc_embedding = np.frombuffer(embedding_data, dtype=np.float32)
        similarity = cosine_similarity(query_embedding, doc_embedding)

        results.append({
            "doc_id": row_dict["doc_id"],
            "similarity": similarity,
            "summary": row_dict["paragraph_summary"] or row_dict["one_sentence_summary"],
            "category": row_dict["category"],
            "date_range": f"{row_dict['date_range_earliest'] or '?'} - {row_dict['date_range_latest'] or '?'}"
        })

    # Sort by similarity and return top results
    results.sort(key=lambda x: x["similarity"], reverse=True)
    cursor.close()
    conn.close()

    return results[:limit]


def semantic_search(query: str, limit: int = 10) -> list[dict]:
    """
    Search documents by semantic similarity.

    Uses PGVector for PostgreSQL (efficient) or numpy for SQLite (fallback).
    """
    if _is_postgres():
        return semantic_search_pgvector(query, limit)
    else:
        return semantic_search_fallback(query, limit)


# ==================== Intelligent Keyword Search ====================

def keyword_search_postgres(keywords: list[str], limit: int = 10) -> list[dict]:
    """
    Keyword search using PostgreSQL full-text search (tsvector).

    Uses ts_rank for ranking and to_tsquery for matching.

    Args:
        keywords: List of keywords to search for
        limit: Maximum number of results

    Returns:
        List of documents with keyword relevance scores
    """
    conn = get_db()
    cursor = conn.cursor()

    # Build tsquery from keywords
    # Using & for AND, | for OR
    tsquery = " & ".join([f"'{kw}'" for kw in keywords])

    # Optimized: sample 500 candidates with all needed columns in one scan
    # Then sort and limit - avoids expensive JOIN
    cursor.execute("""
        SELECT doc_id, paragraph_summary, one_sentence_summary,
               category, date_range_earliest, date_range_latest, rank
        FROM (
            SELECT doc_id, paragraph_summary, one_sentence_summary,
                   category, date_range_earliest, date_range_latest,
                   ts_rank(text_search_vector, to_tsquery('english', %s)) as rank
            FROM all_embeddings_mv
            WHERE text_search_vector @@ to_tsquery('english', %s)
            LIMIT 500
        ) AS candidates
        ORDER BY rank DESC
        LIMIT %s
    """, [tsquery, tsquery, limit])

    results = []
    for row in cursor:
        results.append({
            "doc_id": row["doc_id"],
            "rank": float(row["rank"]),
            "summary": row["paragraph_summary"] or row["one_sentence_summary"],
            "category": row["category"],
            "date_range": f"{row['date_range_earliest'] or '?'} - {row['date_range_latest'] or '?'}"
        })

    cursor.close()
    conn.close()

    return results


def keyword_search_fallback(keywords: list[str], limit: int = 10) -> list[dict]:
    """
    Fallback keyword search using ILIKE for SQLite.

    Note: This is slower and less accurate than PostgreSQL full-text search.
    """
    conn = get_db()
    cursor = conn.cursor()

    conditions = " AND ".join(["full_text LIKE ?" for _ in keywords])
    params = [f"%{kw}%" for kw in keywords] + [limit]

    query = f"""
        SELECT doc_id, paragraph_summary, one_sentence_summary, category,
               date_range_earliest, date_range_latest
        FROM all_documents
        WHERE {conditions}
        LIMIT ?
    """

    cursor.execute(query, params)

    results = []
    for row in cursor:
        row_dict = dict(row) if hasattr(row, 'keys') else row
        results.append({
            "doc_id": row_dict["doc_id"],
            "rank": 1.0,  # All results have equal rank in fallback
            "summary": row_dict["paragraph_summary"] or row_dict["one_sentence_summary"],
            "category": row_dict["category"],
            "date_range": f"{row_dict['date_range_earliest'] or '?'} - {row_dict['date_range_latest'] or '?'}"
        })

    cursor.close()
    conn.close()

    return results


def keyword_search(keywords: list[str], limit: int = 10) -> list[dict]:
    """
    Search documents by keywords using full-text search (PostgreSQL) or ILIKE (SQLite).
    """
    if _is_postgres():
        return keyword_search_postgres(keywords, limit)
    else:
        return keyword_search_fallback(keywords, limit)


# ==================== MRR Scoring ====================

def calculate_mrr(relevance_scores: list[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for ranking evaluation.

    MRR = 1 / position_of_first_relevant_result

    Args:
        relevance_scores: List where 1 = relevant, 0 = not relevant
                           Position in list indicates rank

    Returns:
        MRR score (0-1), 0 if no relevant results
    """
    for rank, is_relevant in enumerate(relevance_scores, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def reciprocal_rank_score(rank: int) -> float:
    """
    Convert a rank to reciprocal rank score (1/rank).

    Higher rank (smaller number) = higher score.
    """
    return 1.0 / max(rank, 1)


# ==================== Hybrid Search with MRR ====================

def hybrid_search_pgvector(
    query: str,
    keywords: Optional[list[str]] = None,
    limit: int = 10,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4
) -> list[dict]:
    """
    Hybrid search combining semantic similarity and keyword relevance with MRR.

    Scores are computed as:
        final_score = semantic_weight * semantic_score +
                     keyword_weight * keyword_score +
                     mrr_boost

    The MRR boost is applied based on the combined ranking position.

    Args:
        query: Natural language query
        keywords: Optional keywords for filtering
        limit: Maximum results
        semantic_weight: Weight for semantic similarity (0-1)
        keyword_weight: Weight for keyword relevance (0-1)

    Returns:
        Ranked list of documents with combined scores
    """
    # Get semantic results (more to allow re-ranking)
    semantic_results = semantic_search_pgvector(query, limit * 2)

    if not keywords:
        return semantic_results[:limit]

    # Get keyword results
    keyword_results = keyword_search_postgres(keywords, limit * 2)

    # Combine scores using MRR-inspired ranking
    # Create a map of doc_id to combined score
    combined_scores = {}

    # Add semantic scores with MRR-based decay
    for rank, result in enumerate(semantic_results):
        doc_id = result["doc_id"]
        semantic_score = result["similarity"]
        # Higher rank = higher score via reciprocal rank
        rank_boost = reciprocal_rank_score(rank + 1)
        combined_scores[doc_id] = {
            "semantic_score": semantic_score,
            "semantic_rank_boost": rank_boost,
            "keyword_score": 0.0,
            "keyword_rank_boost": 0.0,
            "summary": result["summary"],
            "category": result["category"],
            "date_range": result["date_range"]
        }

    # Add keyword scores
    keyword_map = {r["doc_id"]: r for r in keyword_results}
    for rank, result in enumerate(keyword_results):
        doc_id = result["doc_id"]
        if doc_id in combined_scores:
            combined_scores[doc_id]["keyword_score"] = result["rank"]
            combined_scores[doc_id]["keyword_rank_boost"] = reciprocal_rank_score(rank + 1)
        else:
            combined_scores[doc_id] = {
                "semantic_score": 0.0,
                "semantic_rank_boost": 0.0,
                "keyword_score": result["rank"],
                "keyword_rank_boost": reciprocal_rank_score(rank + 1),
                "summary": result["summary"],
                "category": result["category"],
                "date_range": result["date_range"]
            }

    # Calculate final scores
    for doc_id, scores in combined_scores.items():
        # Normalize keyword score (ts_rank can be > 1)
        normalized_keyword = min(scores["keyword_score"], 1.0)

        # Combined score with MRR boost
        final_score = (
            semantic_weight * scores["semantic_score"] +
            keyword_weight * normalized_keyword
        ) * (1 + 0.2 * (scores["semantic_rank_boost"] + scores["keyword_rank_boost"]) / 2)

        combined_scores[doc_id]["final_score"] = final_score

    # Sort by final score
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1]["final_score"],
        reverse=True
    )

    # Format results
    results = []
    for doc_id, scores in sorted_results[:limit]:
        results.append({
            "doc_id": doc_id,
            "score": scores["final_score"],
            "semantic_similarity": scores["semantic_score"],
            "keyword_relevance": scores["keyword_score"],
            "summary": scores["summary"],
            "category": scores["category"],
            "date_range": scores["date_range"]
        })

    return results


def hybrid_search(
    query: str,
    keywords: Optional[list[str]] = None,
    limit: int = 10,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4
) -> list[dict]:
    """
    Hybrid search combining semantic and keyword search.

    Uses PGVector for PostgreSQL or falls back to simple combination for SQLite.
    """
    if _is_postgres():
        return hybrid_search_pgvector(query, keywords, limit, semantic_weight, keyword_weight)

    # Fallback: get semantic results and filter by keywords
    semantic_results = semantic_search(query, limit * 3)

    if not keywords:
        return semantic_results[:limit]

    # Filter by keywords
    filtered_results = []
    for result in semantic_results:
        doc_text = get_document_text(result["doc_id"])
        if doc_text:
            doc_text_lower = doc_text.lower()
            if all(kw.lower() in doc_text_lower for kw in keywords):
                filtered_results.append(result)

    return filtered_results[:limit]


# ==================== Document Retrieval ====================

def get_document_text(doc_id: str) -> str | None:
    """Get full text of a document."""
    conn = get_db()
    cursor = conn.cursor()

    if _is_postgres():
        cursor.execute("SELECT full_text FROM all_embeddings_mv WHERE doc_id = %s", [doc_id])
    else:
        cursor.execute("SELECT full_text FROM all_documents WHERE doc_id = ?", [doc_id])

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row:
        row_dict = dict(row) if hasattr(row, 'keys') else row
        return row_dict["full_text"]
    return None


def get_document_with_metadata(doc_id: str) -> dict | None:
    """Get full document with metadata."""
    conn = get_db()
    cursor = conn.cursor()

    if _is_postgres():
        cursor.execute("""
            SELECT doc_id, full_text, one_sentence_summary, paragraph_summary,
                   category, date_range_earliest, date_range_latest, file_path
            FROM all_documents WHERE doc_id = %s
        """, [doc_id])
    else:
        cursor.execute("""
            SELECT doc_id, full_text, one_sentence_summary, paragraph_summary,
                   category, date_range_earliest, date_range_latest, file_path
            FROM all_documents WHERE doc_id = ?
        """, [doc_id])

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row:
        row_dict = dict(row) if hasattr(row, 'keys') else row
        return {
            "doc_id": row_dict["doc_id"],
            "full_text": row_dict["full_text"],
            "summary": row_dict["paragraph_summary"] or row_dict["one_sentence_summary"],
            "one_sentence_summary": row_dict["one_sentence_summary"],
            "paragraph_summary": row_dict["paragraph_summary"],
            "category": row_dict["category"],
            "date_range": f"{row_dict['date_range_earliest'] or '?'} - {row_dict['date_range_latest'] or '?'}",
            "file_path": row_dict["file_path"]
        }
    return None


# ==================== Graph Queries (PostgreSQL) ====================

def _resolve_persons(cursor, persons: list[str]) -> list[str]:
    """
    Resolve input names to canonical names as stored in rdf_triples.actor_canonical / target_canonical.
    Uses entity_aliases for lookup, with ILIKE fallback.
    Returns deduplicated canonical names.
    """
    canonical = set()
    unresolved = []

    for p in persons:
        # Fast: exact PK lookup on entity_aliases
        cursor.execute(
            "SELECT canonical_name FROM entity_aliases WHERE original_name = %s",
            (p,),
        )
        row = cursor.fetchone()
        if row:
            canonical.add(row["canonical_name"])
            continue

        # Fast: maybe the input IS already a canonical name
        cursor.execute(
            "SELECT 1 FROM rdf_triples WHERE actor_canonical = %s LIMIT 1",
            (p,),
        )
        if cursor.fetchone():
            canonical.add(p)
            continue

        unresolved.append(p)

    # ILIKE fallback for unresolved names
    if unresolved:
        like_clauses = []
        params = []
        for p in unresolved:
            like_clauses.append("actor_canonical ILIKE %s")
            params.append(f"%{p}%")
        cursor.execute(
            f"SELECT DISTINCT actor_canonical FROM rdf_triples WHERE {' OR '.join(like_clauses)} LIMIT 10",
            params,
        )
        for row in cursor:
            canonical.add(row["actor_canonical"])

    return list(canonical) if canonical else persons


def get_subgraph_for_persons(persons: list[str], depth: int = 1, limit: int = 50) -> dict:
    """
    Build a focused subgraph showing how the queried persons relate.

    Multi-person strategy (single optimised SQL):
      1. Direct links between queried persons
      2. Best shared connection (1 intermediary connecting any pair)
      3. Top distinct connections per person as complement

    Single-person: returns their top connections directly.

    Args:
        persons: List of person names to center the graph around
        depth: unused (kept for API compat) — intermediary logic replaces it
        limit: Max relationships to return
    """
    conn = get_db()
    cursor = conn.cursor()

    canonical_names = _resolve_persons(cursor, persons)
    if not canonical_names:
        cursor.close()
        conn.close()
        return {"nodes": [], "edges": [], "queried_persons": persons, "depth": depth}

    queried_set = {c.lower() for c in canonical_names}
    nodes_map = {}
    edges = []
    seen = set()

    def _add(row):
        key = (row["source"], row["target"], row["doc_id"])
        if key in seen:
            return
        seen.add(key)
        for p in (row["source"], row["target"]):
            if p not in nodes_map:
                nodes_map[p] = {"id": p, "label": p, "connections": 0}
            nodes_map[p]["connections"] += 1
        edges.append({
            "source": row["source"], "target": row["target"],
            "action": row["action"], "doc_id": row["doc_id"],
            "timestamp": row.get("timestamp"), "location": row.get("location"),
            "topic": row.get("topic"),
        })

    ct = tuple(canonical_names)

    # === 1. Direct links between queried persons ===
    if len(canonical_names) >= 2:
        cursor.execute("""
            SELECT actor_canonical AS source, target_canonical AS target,
                   action, doc_id, timestamp, location,
                   COALESCE(explicit_topic, implicit_topic) AS topic
            FROM rdf_triples
            WHERE actor_canonical IN %s AND target_canonical IN %s
              AND actor_canonical != target_canonical
            LIMIT %s
        """, (ct, ct, limit))
        for row in cursor:
            _add(row)

        # === 2. Best shared connection per pair (max 1 intermediary) ===
        # Single query: for each pair (a, b), find the top-1 shared neighbour
        # Uses a lateral join for efficiency — PostgreSQL optimises this well
        from itertools import combinations
        for a, b in combinations(canonical_names, 2):
            cursor.execute("""
                SELECT sub.shared, sub.a_action, sub.a_doc_id, sub.b_action, sub.b_doc_id
                FROM (
                    SELECT
                        t1.target_canonical AS shared,
                        t1.action AS a_action, t1.doc_id AS a_doc_id,
                        t2.action AS b_action, t2.doc_id AS b_doc_id
                    FROM rdf_triples t1
                    JOIN rdf_triples t2 ON t1.target_canonical = t2.target_canonical
                    WHERE t1.actor_canonical = %s AND t2.actor_canonical = %s
                      AND t1.target_canonical != %s AND t1.target_canonical != %s
                      AND t1.actor_canonical != t1.target_canonical
                    LIMIT 1
                ) sub
            """, (a, b, a, b))
            row = cursor.fetchone()
            if row:
                _add({"source": a, "target": row["shared"],
                      "action": row["a_action"], "doc_id": row["a_doc_id"]})
                _add({"source": b, "target": row["shared"],
                      "action": row["b_action"], "doc_id": row["b_doc_id"]})

    # === 3. Top 5 connections per person ranked by exchange count ===
    top_n = 5
    # Filter out non-person entries (generic labels, emails, orgs)
    noise_filter = """
        AND {col} !~ '@'
        AND {col} NOT IN (
            'email','communication','unknown','recipient','transaction',
            'DOJ document','Confidential Document','document'
        )
    """
    for person in canonical_names:
        # Find top 5 most connected persons (by number of exchanges)
        actor_filter = noise_filter.format(col="target_canonical")
        target_filter = noise_filter.format(col="actor_canonical")
        cursor.execute(f"""
            SELECT connected, SUM(cnt) AS total FROM (
                SELECT target_canonical AS connected, COUNT(*) AS cnt
                FROM rdf_triples
                WHERE actor_canonical = %s AND actor_canonical != target_canonical
                {actor_filter}
                GROUP BY target_canonical
                UNION ALL
                SELECT actor_canonical AS connected, COUNT(*) AS cnt
                FROM rdf_triples
                WHERE target_canonical = %s AND actor_canonical != target_canonical
                {target_filter}
                GROUP BY actor_canonical
            ) sub
            GROUP BY connected
            ORDER BY total DESC
            LIMIT %s
        """, (person, person, top_n))
        top_connections = [(row["connected"], row["total"]) for row in cursor]

        if not top_connections:
            continue

        # Pre-populate exchange_count on nodes for top connections
        for connected_name, exchange_count in top_connections:
            if connected_name not in nodes_map:
                nodes_map[connected_name] = {"id": connected_name, "label": connected_name, "connections": 0}
            # Keep max exchange_count if shared between multiple queried persons
            prev = nodes_map[connected_name].get("exchange_count", 0)
            nodes_map[connected_name]["exchange_count"] = max(prev, int(exchange_count))

        # Fetch a sample of edges per top connection (max 10 each)
        per_conn = 10
        for connected_name, _ in top_connections:
            cursor.execute("""
                (SELECT actor_canonical AS source, target_canonical AS target,
                        action, doc_id, timestamp, location,
                        COALESCE(explicit_topic, implicit_topic) AS topic
                 FROM rdf_triples
                 WHERE actor_canonical = %s AND target_canonical = %s
                 LIMIT %s)
                UNION ALL
                (SELECT actor_canonical AS source, target_canonical AS target,
                        action, doc_id, timestamp, location,
                        COALESCE(explicit_topic, implicit_topic) AS topic
                 FROM rdf_triples
                 WHERE actor_canonical = %s AND target_canonical = %s
                 LIMIT %s)
            """, (person, connected_name, per_conn, connected_name, person, per_conn))
            for row in cursor:
                _add(row)

    for name in nodes_map:
        if name.lower() in queried_set:
            nodes_map[name]["is_queried"] = True

    nodes = sorted(nodes_map.values(), key=lambda x: x["connections"], reverse=True)
    cursor.close()
    conn.close()

    return {
        "nodes": nodes, "edges": edges,
        "queried_persons": persons, "depth": depth,
    }


def _get_neighbours(cursor, names: set[str], per_person: int = 100) -> dict[str, list[tuple[str, str, str]]]:
    """
    For a set of canonical names, return their direct neighbours.
    Returns {person: [(neighbour, action, doc_id), ...]}.
    Queries each person separately with a LIMIT to avoid huge result sets.
    """
    if not names:
        return {}

    neighbours: dict[str, list[tuple[str, str, str]]] = {}
    for person in names:
        cursor.execute("""
            (SELECT actor_canonical AS source, target_canonical AS target,
                    action, doc_id
             FROM rdf_triples
             WHERE actor_canonical = %s AND actor_canonical != target_canonical
             LIMIT %s)
            UNION
            (SELECT actor_canonical AS source, target_canonical AS target,
                    action, doc_id
             FROM rdf_triples
             WHERE target_canonical = %s AND actor_canonical != target_canonical
             LIMIT %s)
        """, (person, per_person, person, per_person))

        for row in cursor:
            s, t = row["source"], row["target"]
            neighbours.setdefault(s, []).append((t, row["action"], row["doc_id"]))
            neighbours.setdefault(t, []).append((s, row["action"], row["doc_id"]))

    return neighbours


def find_shortest_path(person1: str, person2: str, max_depth: int = 5) -> dict:
    """
    Find the shortest connection path between two persons.
    Bidirectional BFS driven from Python — each hop only queries the frontier
    neighbours via indexed actor_canonical / target_canonical lookups.

    Args:
        person1: Name of first person
        person2: Name of second person
        max_depth: Maximum number of hops to search

    Returns:
        Path information with nodes and edges
    """
    conn = get_db()
    cursor = conn.cursor()

    resolved = _resolve_persons(cursor, [person1, person2])
    start = resolved[0] if len(resolved) >= 1 else person1
    end = resolved[1] if len(resolved) >= 2 else (resolved[0] if len(resolved) == 1 else person2)

    if start.lower() == end.lower():
        cursor.close()
        conn.close()
        return {
            "found": True, "person1": person1, "person2": person2,
            "path_length": 0,
            "nodes": [{"id": start, "label": start}], "edges": [],
        }

    # Bidirectional BFS — expand the smaller frontier each step
    parent_fwd = {start: None}
    parent_bwd = {end: None}
    frontier_fwd = {start}
    frontier_bwd = {end}
    meeting_node = None

    for _ in range(max_depth):
        if len(frontier_fwd) <= len(frontier_bwd):
            nbrs = _get_neighbours(cursor, frontier_fwd)
            next_frontier = set()
            for node in frontier_fwd:
                for (neighbour, action, doc_id) in nbrs.get(node, []):
                    if neighbour in parent_fwd:
                        continue
                    parent_fwd[neighbour] = (node, action, doc_id)
                    next_frontier.add(neighbour)
                    if neighbour in parent_bwd:
                        meeting_node = neighbour
                        break
                if meeting_node:
                    break
            frontier_fwd = next_frontier
        else:
            nbrs = _get_neighbours(cursor, frontier_bwd)
            next_frontier = set()
            for node in frontier_bwd:
                for (neighbour, action, doc_id) in nbrs.get(node, []):
                    if neighbour in parent_bwd:
                        continue
                    parent_bwd[neighbour] = (node, action, doc_id)
                    next_frontier.add(neighbour)
                    if neighbour in parent_fwd:
                        meeting_node = neighbour
                        break
                if meeting_node:
                    break
            frontier_bwd = next_frontier

        if meeting_node:
            break
        if not frontier_fwd and not frontier_bwd:
            break

    cursor.close()
    conn.close()

    if not meeting_node:
        return {
            "found": False, "person1": person1, "person2": person2,
            "message": f"No path found between {start} and {end} within {max_depth} hops",
        }

    # Reconstruct path: start → meeting_node → end
    path_fwd, edges_fwd = [], []
    node = meeting_node
    while parent_fwd[node] is not None:
        prev, action, doc_id = parent_fwd[node]
        path_fwd.append(node)
        edges_fwd.append({"source": prev, "target": node, "action": action, "doc_id": doc_id})
        node = prev
    path_fwd.append(start)
    path_fwd.reverse()
    edges_fwd.reverse()

    path_bwd, edges_bwd = [], []
    node = meeting_node
    while parent_bwd[node] is not None:
        prev, action, doc_id = parent_bwd[node]
        path_bwd.append(prev)
        edges_bwd.append({"source": node, "target": prev, "action": action, "doc_id": doc_id})
        node = prev

    full_path = path_fwd + path_bwd
    full_edges = edges_fwd + edges_bwd

    return {
        "found": True, "person1": person1, "person2": person2,
        "path_length": len(full_edges),
        "nodes": [{"id": n, "label": n} for n in full_path],
        "edges": full_edges,
    }


def search_persons(query: str, limit: int = 20) -> list[dict]:
    """
    Search for persons by name using actor_canonical / target_canonical columns.
    Single query, no JOINs.

    Args:
        query: Search query (partial name match)
        limit: Maximum results

    Returns:
        List of matching persons with connection counts
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT person, SUM(cnt) AS connections FROM (
            SELECT actor_canonical AS person, COUNT(*) AS cnt
            FROM rdf_triples
            WHERE actor_canonical ILIKE %s
            GROUP BY actor_canonical
            UNION ALL
            SELECT target_canonical AS person, COUNT(*) AS cnt
            FROM rdf_triples
            WHERE target_canonical ILIKE %s
            GROUP BY target_canonical
        ) sub
        GROUP BY person
        ORDER BY connections DESC
        LIMIT %s
    """, (f"%{query}%", f"%{query}%", limit))

    results = [{"name": row["person"], "connections": row["connections"]} for row in cursor]
    cursor.close()
    conn.close()
    return results


import time as _time

_graph_overview_cache: dict[int, tuple[float, dict]] = {}
_GRAPH_OVERVIEW_TTL = 7 * 24 * 3600  # 1 week


def get_graph_overview(limit: int = 30) -> dict:
    """
    Top N most-connected persons and their inter-connections.
    Direct queries on actor_canonical / target_canonical — no JOINs.
    Cached in memory for 1 week (data is static).
    """
    now = _time.monotonic()
    cached = _graph_overview_cache.get(limit)
    if cached and (now - cached[0]) < _GRAPH_OVERVIEW_TTL:
        return cached[1]

    conn = get_db()
    cursor = conn.cursor()

    # Step 1: Top N persons by mention count
    cursor.execute("""
        SELECT person, COUNT(*) AS degree FROM (
            SELECT actor_canonical AS person FROM rdf_triples
            UNION ALL
            SELECT target_canonical AS person FROM rdf_triples
        ) sub
        GROUP BY person
        ORDER BY degree DESC
        LIMIT %s
    """, (limit,))

    top_persons = []
    nodes_map = {}
    for row in cursor:
        name, degree = row["person"], row["degree"]
        top_persons.append(name)
        nodes_map[name] = {"id": name, "label": name, "degree": degree}

    if not top_persons:
        cursor.close()
        conn.close()
        return {"nodes": [], "edges": [], "meta": {"total_displayed": 0}}

    # Step 2: Edges between top persons (aggregated, undirected)
    top_tuple = tuple(top_persons)
    cursor.execute("""
        SELECT
            LEAST(actor_canonical, target_canonical) AS source,
            GREATEST(actor_canonical, target_canonical) AS target,
            COUNT(*) AS weight,
            array_agg(DISTINCT action) FILTER (WHERE action IS NOT NULL) AS actions,
            array_agg(DISTINCT doc_id) FILTER (WHERE doc_id IS NOT NULL) AS doc_ids
        FROM rdf_triples
        WHERE actor_canonical IN %s AND target_canonical IN %s
          AND actor_canonical != target_canonical
        GROUP BY LEAST(actor_canonical, target_canonical), GREATEST(actor_canonical, target_canonical)
        ORDER BY weight DESC
    """, (top_tuple, top_tuple))

    edges = []
    for row in cursor:
        edges.append({
            "source": row["source"],
            "target": row["target"],
            "weight": row["weight"],
            "actions": row["actions"] or [],
            "doc_ids": row["doc_ids"] or [],
        })

    # Step 3: Total edges in the whole graph
    cursor.execute("SELECT COUNT(*) AS total FROM rdf_triples")
    total_edges = cursor.fetchone()["total"]

    cursor.close()
    conn.close()

    nodes = sorted(nodes_map.values(), key=lambda x: x["degree"], reverse=True)

    result = {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "total_displayed_nodes": len(nodes),
            "total_displayed_edges": len(edges),
            "total_edges_in_graph": total_edges,
        },
    }

    _graph_overview_cache[limit] = (_time.monotonic(), result)
    return result


# ==================== PostgreSQL Relationship Queries ====================

def get_relationships_for_actor(actor: str, limit: int = 50) -> list[dict]:
    """Get relationships involving an actor."""
    # Trigram index requires at least 3 characters for efficiency
    if len(actor.strip()) < 3:
        return []

    conn = get_db()
    cursor = conn.cursor()

    if _is_postgres():
        # Optimized query: fetch limited candidates first, then deduplicate
        # This avoids sorting millions of rows when search term is common
        cursor.execute("""
            WITH actor_matches AS (
                SELECT doc_id, timestamp, actor, action, target, location, explicit_topic, implicit_topic
                FROM rdf_triples
                WHERE actor ILIKE %s
                LIMIT %s
            ),
            target_matches AS (
                SELECT doc_id, timestamp, actor, action, target, location, explicit_topic, implicit_topic
                FROM rdf_triples
                WHERE target ILIKE %s
                LIMIT %s
            ),
            combined AS (
                SELECT * FROM actor_matches
                UNION
                SELECT * FROM target_matches
            )
            SELECT DISTINCT ON (doc_id)
                doc_id, timestamp, actor, action, target, location, explicit_topic, implicit_topic
            FROM combined
            ORDER BY doc_id, timestamp
            LIMIT %s
        """, [f"%{actor}%", limit * 2, f"%{actor}%", limit * 2, limit])
    else:
        cursor.execute("""
            SELECT DISTINCT doc_id, timestamp, actor, action, target, location, explicit_topic, implicit_topic
            FROM rdf_triples
            WHERE actor LIKE ? OR target LIKE ?
            ORDER BY timestamp
            LIMIT ?
        """, [f"%{actor}%", f"%{actor}%", limit])

    results = []
    for row in cursor:
        row_dict = dict(row) if hasattr(row, 'keys') else row
        results.append({
            "doc_id": row_dict["doc_id"],
            "timestamp": row_dict["timestamp"],
            "actor": row_dict["actor"],
            "action": row_dict["action"],
            "target": row_dict["target"],
            "location": row_dict["location"],
            "topic": row_dict["explicit_topic"] or row_dict["implicit_topic"]
        })

    cursor.close()
    conn.close()
    return results


def build_connection_graph(persons: list[str], limit: int = 100) -> dict:
    """
    Build a graph of connections between persons.

    Returns a graph structure with nodes (persons) and edges (relationships).
    """
    conn = get_db()
    cursor = conn.cursor()

    # Build query for multiple persons
    all_relationships = []

    for person in persons:
        if _is_postgres():
            cursor.execute("""
                SELECT doc_id, timestamp, actor, action, target, location,
                       explicit_topic, implicit_topic
                FROM rdf_triples
                WHERE actor ILIKE %s OR target ILIKE %s
                LIMIT %s
            """, [f"%{person}%", f"%{person}%", limit])
        else:
            cursor.execute("""
                SELECT doc_id, timestamp, actor, action, target, location,
                       explicit_topic, implicit_topic
                FROM rdf_triples
                WHERE actor LIKE ? OR target LIKE ?
                LIMIT ?
            """, [f"%{person}%", f"%{person}%", limit])

        for row in cursor:
            row_dict = dict(row) if hasattr(row, 'keys') else row
            all_relationships.append(row_dict)

    cursor.close()
    conn.close()

    # Build nodes and edges
    nodes_map = {}  # person_name -> node data
    edges_map = {}  # (source, target) -> edge data

    for rel in all_relationships:
        actor = rel["actor"]
        target = rel["target"]
        action = rel["action"]
        doc_id = rel["doc_id"]

        # Add/update actor node
        if actor not in nodes_map:
            nodes_map[actor] = {
                "id": actor,
                "label": actor,
                "connections": 0,
                "doc_ids": set()
            }
        nodes_map[actor]["connections"] += 1
        nodes_map[actor]["doc_ids"].add(doc_id)

        # Add/update target node
        if target not in nodes_map:
            nodes_map[target] = {
                "id": target,
                "label": target,
                "connections": 0,
                "doc_ids": set()
            }
        nodes_map[target]["connections"] += 1
        nodes_map[target]["doc_ids"].add(doc_id)

        # Add/update edge (use sorted tuple to avoid duplicates A->B and B->A)
        edge_key = tuple(sorted([actor, target]))
        if edge_key not in edges_map:
            edges_map[edge_key] = {
                "source": edge_key[0],
                "target": edge_key[1],
                "actions": [],
                "doc_ids": set(),
                "count": 0
            }
        edges_map[edge_key]["actions"].append(action)
        edges_map[edge_key]["doc_ids"].add(doc_id)
        edges_map[edge_key]["count"] += 1

    # Convert to lists and summarize
    nodes = []
    for node in nodes_map.values():
        nodes.append({
            "id": node["id"],
            "label": node["label"],
            "connections": node["connections"],
            "doc_ids": list(node["doc_ids"])[:10]  # Limit doc_ids
        })

    edges = []
    for edge in edges_map.values():
        # Summarize actions (most common)
        from collections import Counter
        action_counts = Counter(edge["actions"])
        top_actions = [a for a, _ in action_counts.most_common(3)]

        edges.append({
            "source": edge["source"],
            "target": edge["target"],
            "label": ", ".join(top_actions),
            "actions": top_actions,
            "doc_ids": list(edge["doc_ids"])[:10],
            "count": edge["count"]
        })

    # Sort nodes by connections (most connected first)
    nodes.sort(key=lambda x: x["connections"], reverse=True)

    # Sort edges by count
    edges.sort(key=lambda x: x["count"], reverse=True)

    return {
        "nodes": nodes,
        "edges": edges,
        "total_relationships": len(all_relationships)
    }


# ==================== MCP Tools ====================

@mcp.tool()
def search_documents(query: str, limit: int = 10) -> str:
    """
    Search documents using semantic similarity (PGVector).

    Args:
        query: Natural language search query (e.g., "meetings with politicians")
        limit: Maximum number of results to return (default 10)

    Returns:
        List of relevant documents with similarity scores.
    """
    results = semantic_search(query, limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def search_by_keywords(keywords: str, limit: int = 10) -> str:
    """
    Search documents by keywords using full-text search.

    Args:
        keywords: Comma-separated keywords (e.g., "island, flight, 2005")
        limit: Maximum number of results to return (default 10)

    Returns:
        List of documents containing all keywords, ranked by relevance.
    """
    keyword_list = [k.strip() for k in keywords.split(",")]
    results = keyword_search(keyword_list, limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """
    Get the full text of a specific document.

    Args:
        doc_id: Document ID (e.g., "gov.uscourts.nysd.447706.195.0")

    Returns:
        Full document text.
    """
    text = get_document_text(doc_id)
    if text:
        return text
    return f"Document not found: {doc_id}"


@mcp.tool()
def search_actor(actor_name: str, limit: int = 50) -> str:
    """
    Get all relationships involving a specific person/actor.

    Args:
        actor_name: Name of the person (e.g., "Bill Clinton", "Ghislaine Maxwell")
        limit: Maximum number of relationships to return

    Returns:
        List of relationships (actor -> action -> target) involving this person.
    """
    results = get_relationships_for_actor(actor_name, limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def get_stats() -> str:
    """
    Get database statistics (cached for 1 week).

    Returns:
        Total documents, relationships, actors, and search capabilities.
    """
    global _stats_cache

    # Check cache
    if _stats_cache["data"] and (time.time() - _stats_cache["timestamp"]) < STATS_CACHE_TTL:
        return _stats_cache["data"]

    conn = get_db()

    stats = {
        "total_documents": conn.execute("SELECT COUNT(*) FROM all_embeddings_mv").fetchone()[0],
        "total_relationships": conn.execute("SELECT COUNT(*) FROM rdf_triples").fetchone()[0],
        "total_actors": conn.execute("SELECT COUNT(DISTINCT actor) FROM rdf_triples").fetchone()[0],
        "total_embeddings": conn.execute("SELECT COUNT(*) FROM all_embeddings_mv").fetchone()[0],
        "search_backend": "PGVector (PostgreSQL) - Materialized View" if _is_postgres() else "NumPy (SQLite fallback)",
        "categories": []
    }

    cursor = conn.execute("""
        SELECT category, COUNT(*) as count
        FROM all_embeddings_mv
        GROUP BY category
        ORDER BY count DESC
    """)
    stats["categories"] = [{"name": row[0], "count": row[1]} for row in cursor]

    conn.close()

    # Update cache
    result = json.dumps(stats, indent=2)
    _stats_cache = {"data": result, "timestamp": time.time()}

    return result


@mcp.tool()
def hybrid_search_tool(query: str, keywords: str = "", limit: int = 10) -> str:
    """
    Combined semantic + keyword search with MRR scoring.

    Uses PGVector for semantic similarity and full-text search for keywords.
    Results are re-ranked using Mean Reciprocal Rank (MRR) principles.

    Args:
        query: Natural language query for semantic search
        keywords: Optional comma-separated keywords to filter results
        limit: Maximum results to return

    Returns:
        Documents ranked by combined semantic and keyword relevance.
    """
    keyword_list = [k.strip() for k in keywords.split(",")] if keywords else None
    results = hybrid_search(query, keyword_list, limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def answer_question(question: str, num_docs: int = 5) -> str:
    """
    Retrieve relevant context for answering a question.

    Args:
        question: The question to answer
        num_docs: Number of documents to retrieve for context (default 5)

    Returns:
        Relevant document excerpts.
    """
    results = semantic_search(question, num_docs)

    context_parts = []
    for i, result in enumerate(results, 1):
        doc_text = get_document_text(result["doc_id"])
        preview = doc_text[:2000] if doc_text else "No text available"

        context_parts.append(f"""
--- Document {i} (similarity: {result['similarity']:.3f}) ---
Doc ID: {result['doc_id']}
Category: {result['category']}
Date Range: {result['date_range']}
Summary: {result['summary']}

Excerpt:
{preview}
""")

    return "\n".join(context_parts)


if __name__ == "__main__":
    mcp.run()
