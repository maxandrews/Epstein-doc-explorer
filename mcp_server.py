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
