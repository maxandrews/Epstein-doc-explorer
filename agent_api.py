#!/usr/bin/env python3
"""
LangGraph Agent API for Epstein Document Explorer.

Provides an agentic RAG system that can search, analyze, and answer
questions about the Epstein document corpus.
"""

import asyncio
import json
import logging
import operator
import os
import re
from typing import Annotated, Sequence, TypedDict

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from dotenv import load_dotenv
load_dotenv(override=True)

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_api")

# ============== S3 Client Configuration ==============

BUCKET_NAME = os.environ.get("BUCKET_NAME")
BUCKET_ACCESS_KEY_ID = os.environ.get("BUCKET_ACCESS_KEY_ID")
BUCKET_SECRET_ACCESS_KEY = os.environ.get("BUCKET_SECRET_ACCESS_KEY")
BUCKET_REGION = os.environ.get("BUCKET_REGION", "us-west-1")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "https://t3.storageapi.dev")

s3_client = None
if BUCKET_NAME and BUCKET_ACCESS_KEY_ID and BUCKET_SECRET_ACCESS_KEY:
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=BUCKET_ACCESS_KEY_ID,
            aws_secret_access_key=BUCKET_SECRET_ACCESS_KEY,
            region_name=BUCKET_REGION,
            config=Config(signature_version="s3v4"),
        )
        logger.info("S3 client configured for bucket: %s", BUCKET_NAME)
    except Exception as e:
        logger.warning("Failed to configure S3 client: %s", e)


def extract_bucket_filename(doc_id: str) -> str | None:
    """
    Extract the bucket filename from a doc_id.
    Handles various formats like TEXT-002-HOUSE_OVERSIGHT_033508 to extract HOUSE_OVERSIGHT_033508.

    Args:
        doc_id: Document ID in various formats

    Returns:
        The cleaned filename for bucket lookup (e.g., HOUSE_OVERSIGHT_033508)
    """
    if not doc_id:
        return None

    # Pattern to match HOUSE_OVERSIGHT_XXXXXX or similar patterns (WORD_WORD_digits)
    match = re.search(r'([A-Z]+_[A-Z]+_\d+)', doc_id)
    if match:
        return match.group(1)

    # Fallback: return original doc_id if no pattern found
    return doc_id


def generate_presigned_url(doc_id: str, expiration: int = 3600) -> str | None:
    """
    Generate a presigned URL for an image in the S3 bucket.
    Files are stored as {doc_id}.jpg in epstein/001/ to epstein/012/ folders.

    Args:
        doc_id: Document ID (e.g., HOUSE_OVERSIGHT_010479 or TEXT-002-HOUSE_OVERSIGHT_033508)
        expiration: URL expiration time in seconds (default 1 hour)

    Returns:
        Presigned URL string or None if generation fails
    """
    if not s3_client or not BUCKET_NAME or not doc_id:
        return None

    # Extract the bucket filename from doc_id (handles prefixes like TEXT-002-)
    bucket_filename = extract_bucket_filename(doc_id)
    if not bucket_filename:
        return None

    filename = f"{bucket_filename}.jpg"

    # Try each folder from 001 to 012
    for i in range(1, 13):
        folder = f"{i:03d}"
        key = f"epstein/{folder}/{filename}"

        try:
            # Check if file exists
            s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
            # File exists, generate presigned URL
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError:
            # File not in this folder, try next
            continue

    logger.warning("File not found in any folder for doc_id: %s (extracted: %s)", doc_id, bucket_filename)
    return None


# Réutiliser les fonctions du MCP server
from mcp_server import (
    semantic_search,
    keyword_search,
    hybrid_search,
    get_document_text,
    get_document_with_metadata,
    get_relationships_for_actor,
    get_db,
    DATABASE_URL,
)

# Graph functions (Neo4j-based)
from neo4j_graph import (
    get_subgraph_for_persons,
    find_shortest_path,
    search_persons,
    get_graph_overview,
    expand_node,
    get_graph_stats,
)

# ============== FastAPI App ==============

app = FastAPI(
    title="Epstein Docs Agent API",
    description="Agentic RAG system for exploring the Epstein document corpus",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warmup_graph_cache():
    """Warm up graph overview cache on startup."""
    logger.info("Warming up graph overview cache...")
    try:
        get_graph_overview(limit=30)
        logger.info("Graph overview cache warmed up successfully")
    except Exception as e:
        logger.warning("Graph cache warmup failed: %s", e)


# API Key pour sécuriser les endpoints
API_KEY = os.environ.get("API_KEY")

async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Vérifie la clé API si elle est configurée."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key

# ============== LangChain Tools ==============


@tool
async def search_documents(query: str, limit: int = 10) -> str:
    """
    Search documents using semantic similarity with PGVector.
    Use this to find documents related to a topic, person, or event.
    Fast and efficient vector similarity search.

    Args:
        query: Natural language search query (e.g., "meetings with politicians", "flights to the island")
        limit: Maximum number of results to return (default 10)
    """
    results = await asyncio.to_thread(semantic_search, query, limit)
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
async def search_actor(actor_name: str, limit: int = 50) -> str:
    """
    Get all relationships involving a specific person/actor.
    Use this to find what actions a person took or what happened to them.

    Args:
        actor_name: Name of the person (e.g., "Bill Clinton", "Ghislaine Maxwell", "Prince Andrew")
        limit: Maximum number of relationships to return
    """
    results = await asyncio.to_thread(get_relationships_for_actor, actor_name, limit)
    return json.dumps(results, indent=2, ensure_ascii=False)


def _get_document_sync(doc_id: str) -> str:
    """Sync helper for get_document."""
    text = get_document_text(doc_id)
    if text:
        if len(text) > 8000:
            return text[:8000] + "\n\n[... document tronqué, utilisez une recherche plus spécifique ...]"
        return text
    return f"Document not found: {doc_id}"


@tool
async def get_document(doc_id: str) -> str:
    """
    Get the full text of a specific document.
    Use this when you need to read the complete content of a document.

    Args:
        doc_id: Document ID (e.g., "HOUSE_OVERSIGHT_010568")
    """
    return await asyncio.to_thread(_get_document_sync, doc_id)


@tool
async def search_by_keywords(keywords: str, limit: int = 10) -> str:
    """
    Search documents by keywords using PostgreSQL full-text search.
    Uses tsvector and ts_rank for intelligent ranking. Fast and accurate.

    Args:
        keywords: Comma-separated keywords (e.g., "island, flight, 2005")
        limit: Maximum number of results to return
    """
    keyword_list = [k.strip() for k in keywords.split(",")]
    results = await asyncio.to_thread(keyword_search, keyword_list, limit)
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
async def search_hybrid(query: str, keywords: str = "", limit: int = 10) -> str:
    """
    Hybrid search combining semantic similarity and keyword relevance with MRR scoring.
    Best results when you have both a natural language query and specific keywords.

    Uses PGVector for semantic search, full-text search for keywords,
    and Mean Reciprocal Rank (MRR) for intelligent re-ranking.

    Args:
        query: Natural language query (e.g., "meetings with politicians")
        keywords: Optional comma-separated keywords (e.g., "Bill Clinton, 2002")
        limit: Maximum number of results to return
    """
    keyword_list = [k.strip() for k in keywords.split(",")] if keywords.strip() else None
    results = await asyncio.to_thread(hybrid_search, query, keyword_list, limit)
    return json.dumps(results, indent=2, ensure_ascii=False)


def _get_database_stats_sync() -> str:
    """Sync helper for get_database_stats."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as cnt FROM documents")
    total_documents = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(*) as cnt FROM rdf_triples")
    total_relationships = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(DISTINCT actor) as cnt FROM rdf_triples")
    unique_actors = cursor.fetchone()["cnt"]

    cursor.execute("SELECT COUNT(DISTINCT target) as cnt FROM rdf_triples")
    unique_targets = cursor.fetchone()["cnt"]

    stats = {
        "total_documents": total_documents,
        "total_relationships": total_relationships,
        "unique_actors": unique_actors,
        "unique_targets": unique_targets,
        "categories": [],
    }

    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC
        LIMIT 10
    """)
    stats["categories"] = [{"name": row["category"], "count": row["count"]} for row in cursor]

    cursor.close()
    conn.close()
    return json.dumps(stats, indent=2, ensure_ascii=False)


@tool
async def get_database_stats() -> str:
    """
    Get statistics about the document database.
    Use this to understand the scope and content of the corpus.
    """
    return await asyncio.to_thread(_get_database_stats_sync)


@tool
def get_connection_graph(persons: str, depth: int = 1) -> str:
    """
    Get a graph of connections between people using Neo4j.
    Use this when the user asks about relationships, connections, or links between people.
    Returns a graph structure with edge weights (for visualizing connection strength).

    IMPORTANT: Use this tool for ANY question about:
    - connections between people ("quelle connexion entre X et Y")
    - relationships ("relation entre...")
    - links ("lien entre...")
    - how people are connected
    - network of a person

    Edge weight indicates how many times two people appear together (thicker = more frequent).

    Args:
        persons: Comma-separated names of people (e.g., "Bill Clinton, Jeffrey Epstein")
        depth: How many connection hops to include (1=direct only, 2=friends of friends)
    """
    person_list = [p.strip() for p in persons.split(",")]
    result = get_subgraph_for_persons(person_list, depth=depth, limit=50)
    # Mark this as graph data for frontend
    result["_type"] = "graph"
    return json.dumps(result, indent=2, ensure_ascii=False)


@tool
def find_connection_path(person1: str, person2: str) -> str:
    """
    Find the shortest path connecting two people in the relationship graph.
    Use this to discover how two people are connected through others.

    Args:
        person1: Name of first person
        person2: Name of second person
    """
    result = find_shortest_path(person1, person2, max_depth=5)
    result["_type"] = "path"
    return json.dumps(result, indent=2, ensure_ascii=False)


# ============== LangGraph Agent ==============

class AgentState(TypedDict):
    """État de l'agent entre les étapes."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Liste des outils disponibles
tools = [
    search_documents,
    search_actor,
    get_document,
    search_by_keywords,
    search_hybrid,
    get_database_stats,
    get_connection_graph,
    find_connection_path,
]

# Modèle via OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")

model = ChatOpenAI(
    model=OPENROUTER_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=4096,
    default_headers={
        "HTTP-Referer": "https://epstein-doc-explorer.local",
        "X-Title": "Epstein Doc Explorer",
    },
).bind_tools(tools)

# Modèle léger pour générer les titres de conversation (sans outils)
# Note: max_tokens doit être assez élevé pour les modèles avec reasoning tokens (ex: MiniMax)
title_model = ChatOpenAI(
    model=OPENROUTER_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=200,
    default_headers={
        "HTTP-Referer": "https://epstein-doc-explorer.local",
        "X-Title": "Epstein Doc Explorer",
    },
)

async def generate_conversation_title(messages: Sequence[BaseMessage]) -> str:
    """
    Génère un résumé court de la conversation via le LLM.
    Fallback sur la première requête tronquée en cas d'erreur.
    """
    logger.debug("generate_conversation_title: %d messages reçus", len(messages))

    # Extraire les échanges user/assistant (exclure les ToolMessage)
    conversation_parts = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            conversation_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            conversation_parts.append(f"Assistant: {msg.content}")

    logger.debug("generate_conversation_title: %d conversation_parts extraits", len(conversation_parts))

    if not conversation_parts:
        logger.warning("generate_conversation_title: aucune conversation_parts, fallback 'Conversation'")
        return "Conversation"

    # Limiter la taille pour le prompt (derniers échanges)
    conversation_text = "\n".join(conversation_parts[-6:])
    if len(conversation_text) > 1500:
        conversation_text = conversation_text[:1500] + "..."

    prompt = f"""Generate a very short title (5-10 words maximum) that summarizes this conversation.
Reply with ONLY the title, nothing else. Use the same language as the user.

Conversation:
{conversation_text}
"""
    try:
        response = await title_model.ainvoke([HumanMessage(content=prompt)])
        logger.info("generate_conversation_title: response=%r", response)
        logger.info("generate_conversation_title: additional_kwargs=%s, response_metadata=%s",
                    getattr(response, 'additional_kwargs', None), getattr(response, 'response_metadata', None))

        # Handle different content formats (string or list of content blocks)
        content = response.content
        if isinstance(content, list):
            # Some models return list of content blocks
            text_parts = [block.get("text", "") if isinstance(block, dict) else str(block) for block in content]
            content = " ".join(text_parts)

        title = (content or "").strip()
        if not title:
            logger.warning("generate_conversation_title: LLM a retourné une réponse vide, fallback 'Conversation'")
            return "Conversation"
        logger.info("generate_conversation_title: titre généré = '%s'", title[:80])
        return title[:80]
    except Exception as e:
        logger.warning("Échec génération titre conversation, fallback: %s", e, exc_info=True)
        # Fallback: première requête utilisateur
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                q = msg.content.strip()
                return q[:50].rsplit(" ", 1)[0] + "..." if len(q) > 50 else q
        return "Conversation"


# System prompt for the agent
SYSTEM_PROMPT = """You are an expert assistant specialized in analyzing the Epstein documents.
You have access to a database containing thousands of court documents, emails, depositions,
and other materials related to the Jeffrey Epstein case.

Your capabilities:
- Semantic search with PGVector (fast vector search)
- Keyword search with PostgreSQL full-text search
- Hybrid search combining semantic + keywords with MRR scoring
- Exploring relationships between people (who did what to whom)
- Reading full document text
- **Graph visualization of connections between people (Neo4j graph database)**

Instructions:
1. Use search tools to find relevant information
2. For thematic questions, use search_documents (semantic search)
3. For questions with specific terms, use search_by_keywords
4. For best results when you have both context and keywords, use search_hybrid
5. For questions about specific people, use search_actor first
6. Always cite your sources with the doc_id
7. Be factual and objective - report what the documents say
8. If you cannot find information, say so clearly

**CRITICAL - Graph/Connection Questions:**
When the user asks about connections, relationships, or links between people, you MUST use the graph tools:
- Use `get_connection_graph` for questions like:
  - "connexion entre X et Y" / "connection between X and Y"
  - "relation entre..." / "relationship between..."
  - "lien entre..." / "link between..."
  - "comment X est lié à Y" / "how is X connected to Y"
  - "réseau de X" / "network of X"
  - "qui connaît X" / "who knows X"
- Use `find_connection_path` to find the shortest path between two people

The graph data will be displayed visually by the frontend. Always use these tools for connection questions!

IMPORTANT: Always respond in the same language as the user's question. If the question is in French, respond in French. If in Spanish, respond in Spanish. Always match the user's language."""


def agent_node(state: AgentState) -> dict:
    """Nœud principal de l'agent qui décide des actions."""
    messages = state["messages"]

    # Ajouter le system prompt au début si pas déjà présent
    if not any(hasattr(m, "type") and m.type == "system" for m in messages):
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Détermine si l'agent doit continuer ou terminer."""
    last_message = state["messages"][-1]

    # Si le dernier message a des tool_calls, continuer vers les outils
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Sinon, terminer
    return END


# Construire le graphe
workflow = StateGraph(AgentState)

# Ajouter les nœuds
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Définir le point d'entrée
workflow.set_entry_point("agent")

# Ajouter les arêtes conditionnelles
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

# Les outils renvoient toujours vers l'agent
workflow.add_edge("tools", "agent")

# Compiler avec mémoire pour les sessions
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)


# ============== API Models ==============

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class SourceInfo(BaseModel):
    doc_id: str
    summary: str | None = None
    category: str | None = None
    date_range: str | None = None
    score: float | None = None
    full_text: str | None = None
    image_url: str | None = None


class GraphNode(BaseModel):
    id: str
    label: str
    connections: int = 0
    is_queried: bool = False
    doc_ids: list[str] = []


class GraphEdge(BaseModel):
    source: str
    target: str
    action: str | None = None
    label: str | None = None
    doc_id: str | None = None
    doc_ids: list[str] = []
    actions: list[str] = []
    count: int = 1
    weight: int = 1  # For edge thickness visualization


class GraphData(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    queried_persons: list[str] = []
    path_length: int | None = None
    found: bool = True


class OverviewNode(BaseModel):
    id: str
    label: str
    degree: int = 0


class OverviewEdge(BaseModel):
    source: str
    target: str
    weight: int = 1
    actions: list[str] = []
    doc_ids: list[str] = []


class OverviewMeta(BaseModel):
    total_displayed_nodes: int = 0
    total_displayed_edges: int = 0
    total_edges_in_graph: int = 0


class GraphOverviewResponse(BaseModel):
    nodes: list[OverviewNode]
    edges: list[OverviewEdge]
    meta: OverviewMeta


class ExpandNode(BaseModel):
    id: str
    label: str
    connections: int = 0
    is_center: bool = False


class ExpandEdge(BaseModel):
    source: str
    target: str
    weight: int = 1
    actions: list[str] = []
    doc_ids: list[str] = []


class ExpandResponse(BaseModel):
    nodes: list[ExpandNode]
    edges: list[ExpandEdge]
    center_person: str
    found: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    tool_calls: list[dict]
    conversation_title: str | None = None
    graph: GraphData | None = None


class StreamEvent(BaseModel):
    type: str  # "token", "tool_start", "tool_end", "done", "error"
    content: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None


class SearchRequest(BaseModel):
    query: str
    keywords: str | None = None
    limit: int = 10


class HybridSearchRequest(BaseModel):
    query: str
    keywords: str | None = None
    limit: int = 10
    semantic_weight: float = 0.6
    keyword_weight: float = 0.4


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "Epstein Docs Agent API"}


@app.get("/api/documents/{doc_id}")
async def get_document_endpoint(doc_id: str, _: str = Depends(verify_api_key)):
    """
    Get a document by its ID with full text and metadata.
    """
    document = get_document_with_metadata(doc_id)
    if document:
        document["image_url"] = generate_presigned_url(doc_id)
        return document
    raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")


_stats_cache = {"data": None, "timestamp": 0}
STATS_CACHE_TTL = 604800  # 1 semaine


def _get_stats_sync():
    """Synchronous stats fetch - runs in thread pool with caching."""
    import time

    # Return cached if still valid
    if _stats_cache["data"] and (time.time() - _stats_cache["timestamp"]) < STATS_CACHE_TTL:
        logger.info("Returning cached stats")
        return _stats_cache["data"]

    logger.info("Fetching fresh stats from DB...")
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM rdf_triples")
    row = cursor.fetchone()
    total_rels = row["count"] if isinstance(row, dict) else row[0]

    cursor.execute("SELECT COUNT(DISTINCT actor) as count FROM rdf_triples")
    row = cursor.fetchone()
    unique_actors = row["count"] if isinstance(row, dict) else row[0]

    cursor.execute("SELECT COUNT(*) as count FROM all_embeddings_mv")
    row = cursor.fetchone()
    total_documents = row["count"] if isinstance(row, dict) else row[0]

    cursor.close()
    conn.close()

    stats = {
        "total_documents": total_documents,
        "total_relationships": total_rels,
        "unique_actors": unique_actors,
        "search_backend": "PGVector (PostgreSQL)" if DATABASE_URL else "NumPy (SQLite fallback)",
    }

    # Update cache
    _stats_cache["data"] = stats
    _stats_cache["timestamp"] = time.time()
    logger.info("Stats cached")

    return stats


@app.get("/api/stats")
async def get_stats():
    """Get database statistics."""
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(None, _get_stats_sync)
    return stats


@app.get("/api/graph/overview", response_model=GraphOverviewResponse)
async def graph_overview(limit: int = 30):
    """
    Get a simplified overview of the relationship graph from Neo4j.
    Returns the top N most-connected persons and their inter-connections.
    Edge weights indicate connection frequency (for edge thickness visualization).
    Designed for initial graph rendering — expand nodes on click for details.
    """
    result = get_graph_overview(limit=min(limit, 100))

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return GraphOverviewResponse(
        nodes=[OverviewNode(**n) for n in result["nodes"]],
        edges=[OverviewEdge(**e) for e in result["edges"]],
        meta=OverviewMeta(**result["meta"]),
    )


@app.get("/api/graph/expand/{person}", response_model=ExpandResponse)
async def graph_expand(person: str, limit: int = 20):
    """
    Expand a node to show its top connections.
    Use this when clicking on a person node in the graph visualization.

    Returns the person's most frequent connections with edge weights.
    Edge weight represents how many times these two people appear together
    (use for edge thickness: higher weight = thicker line).

    Args:
        person: Name of the person to expand
        limit: Maximum number of connections to return (default 20)
    """
    result = expand_node(person, limit=min(limit, 50))

    if not result.get("found", True):
        raise HTTPException(status_code=404, detail=f"Person not found: {person}")

    return ExpandResponse(
        nodes=[ExpandNode(**n) for n in result["nodes"]],
        edges=[ExpandEdge(**e) for e in result["edges"]],
        center_person=result["center_person"],
        found=result["found"],
    )


@app.get("/api/graph/search")
async def graph_search(q: str, limit: int = 20):
    """
    Search for persons by name in the Neo4j graph.

    Args:
        q: Search query (partial name match)
        limit: Maximum number of results (default 20)
    """
    results = search_persons(q, limit=min(limit, 50))
    return {"results": results, "query": q}


@app.get("/api/graph/stats")
async def graph_stats_endpoint():
    """Get statistics about the Neo4j graph."""
    return get_graph_stats()


@app.post("/api/search/semantic")
async def semantic_search_endpoint(request: SearchRequest, _: str = Depends(verify_api_key)):
    """
    Direct semantic search endpoint using PGVector.
    Fast vector similarity search without agent overhead.
    """
    results = semantic_search(request.query, request.limit)
    for r in results:
        r["image_url"] = generate_presigned_url(r.get("doc_id"))
    return {"results": results, "method": "semantic_search", "backend": "pgvector"}


@app.post("/api/search/keywords")
async def keyword_search_endpoint(request: SearchRequest, _: str = Depends(verify_api_key)):
    """
    Direct keyword search endpoint using PostgreSQL full-text search.
    """
    keyword_list = [k.strip() for k in request.keywords.split(",")] if request.keywords else []
    results = keyword_search(keyword_list, request.limit)
    for r in results:
        r["image_url"] = generate_presigned_url(r.get("doc_id"))
    return {"results": results, "method": "keyword_search", "backend": "tsvector"}


@app.post("/api/search/hybrid")
async def hybrid_search_endpoint(request: HybridSearchRequest, _: str = Depends(verify_api_key)):
    """
    Direct hybrid search endpoint combining semantic and keyword search with MRR scoring.

    Combines PGVector semantic search with PostgreSQL full-text search,
    re-ranked using Mean Reciprocal Rank principles.
    """
    keyword_list = [k.strip() for k in request.keywords.split(",")] if request.keywords else None
    results = hybrid_search(
        request.query,
        keyword_list,
        request.limit,
        request.semantic_weight,
        request.keyword_weight
    )
    for r in results:
        r["image_url"] = generate_presigned_url(r.get("doc_id"))
    return {
        "results": results,
        "method": "hybrid_search",
        "weights": {
            "semantic": request.semantic_weight,
            "keyword": request.keyword_weight
        }
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, _: str = Depends(verify_api_key)):
    """
    Query the document corpus with an agentic approach.
    The agent will search, analyze, and synthesize information to answer the question.
    """
    logger.info("POST /api/query session_id=%s question=%s...", request.session_id, (request.question or "")[:50])
    config = {"configurable": {"thread_id": request.session_id or "default"}}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.question)]},
            config=config,
        )

        # Extraire les informations de la réponse
        messages = result["messages"]
        final_answer = ""
        sources_dict = {}  # Use dict to dedupe by doc_id
        tool_calls_info = []
        graph_data = None  # Graph data from connection tools

        for msg in messages:
            # Collecter les tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_info.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                    })

            # Extraire les sources et graph data des résultats d'outils
            if isinstance(msg, ToolMessage):
                try:
                    content = json.loads(msg.content)

                    # Check for graph data (from get_connection_graph or find_connection_path)
                    if isinstance(content, dict) and content.get("_type") in ("graph", "path"):
                        # Extract graph data
                        nodes = [
                            GraphNode(
                                id=n.get("id", ""),
                                label=n.get("label", n.get("id", "")),
                                connections=n.get("connections", 0),
                                is_queried=n.get("is_queried", False),
                                doc_ids=n.get("doc_ids", [])
                            )
                            for n in content.get("nodes", [])
                        ]
                        edges = [
                            GraphEdge(
                                source=e.get("source", ""),
                                target=e.get("target", ""),
                                action=e.get("action"),
                                label=e.get("label"),
                                doc_id=e.get("doc_id"),
                                doc_ids=e.get("doc_ids", []),
                                actions=e.get("actions", []),
                                count=e.get("count", 1),
                                weight=e.get("weight", 1),
                            )
                            for e in content.get("edges", [])
                        ]
                        graph_data = GraphData(
                            nodes=nodes,
                            edges=edges,
                            queried_persons=content.get("queried_persons", []),
                            path_length=content.get("path_length"),
                            found=content.get("found", True)
                        )

                    # Extract document sources
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "doc_id" in item:
                                doc_id = item["doc_id"]
                                if doc_id not in sources_dict:
                                    # Get full document data
                                    doc_data = get_document_with_metadata(doc_id)

                                    # Get score from search result
                                    score = item.get("similarity") or item.get("score") or item.get("rank")

                                    sources_dict[doc_id] = SourceInfo(
                                        doc_id=doc_id,
                                        summary=doc_data.get("one_sentence_summary") if doc_data else item.get("summary"),
                                        category=item.get("category"),
                                        date_range=item.get("date_range"),
                                        score=score,
                                        full_text=doc_data.get("full_text") if doc_data else None,
                                        image_url=generate_presigned_url(doc_id),
                                    )
                except (json.JSONDecodeError, TypeError):
                    pass

            # La dernière réponse AI est la réponse finale
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_answer = msg.content

        # Générer le titre à partir du résumé de la conversation
        conversation_title = await generate_conversation_title(messages)

        return QueryResponse(
            answer=final_answer,
            sources=list(sources_dict.values()),
            tool_calls=tool_calls_info,
            conversation_title=conversation_title,
            graph=graph_data,
        )

    except Exception as e:
        logger.exception("Erreur /api/query: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def iter_with_disconnect(async_iter, http_request, timeout=120):
    """Wrapper that checks for client disconnect between iterations."""
    iterator = async_iter.__aiter__()
    while True:
        # Check disconnect before waiting for next item
        if await http_request.is_disconnected():
            logger.info("Client disconnected, stopping iteration")
            return
        try:
            # Wait for next item with timeout
            item = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
            yield item
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            logger.warning("Stream iteration timeout after %ds", timeout)
            return


@app.post("/api/query/stream")
async def query_documents_stream(
    request: QueryRequest,
    http_request: Request,
    _: str = Depends(verify_api_key)
):
    """
    Stream the agent's response for real-time UI updates.
    Returns Server-Sent Events (SSE).
    """
    logger.info("POST /api/query/stream session_id=%s question=%s...", request.session_id, (request.question or "")[:50])
    config = {"configurable": {"thread_id": request.session_id or "default"}}

    async def generate():
        try:
            stream = agent.astream_events(
                {"messages": [HumanMessage(content=request.question)]},
                config=config,
                version="v2",
            )
            async for event in iter_with_disconnect(stream, http_request):
                event_type = event.get("event", "")

                # Streaming des tokens de réponse
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content}, ensure_ascii=False)}\n\n"

                # Début d'un appel d'outil
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool_name': tool_name, 'tool_args': tool_input}, ensure_ascii=False)}\n\n"

                # Fin d'un appel d'outil
                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    tool_output = event.get("data", {}).get("output", "")

                    # Extract content from ToolMessage if needed
                    if hasattr(tool_output, "content"):
                        tool_output = tool_output.content

                    # Extraire les sources et graph data
                    sources = []
                    graph_data = None
                    seen_doc_ids = set()
                    try:
                        # Handle both string JSON and already parsed objects
                        if isinstance(tool_output, str) and tool_output:
                            parsed = json.loads(tool_output)
                        else:
                            parsed = tool_output

                        # Check for graph data (from get_connection_graph or find_connection_path)
                        if isinstance(parsed, dict) and parsed.get("_type") in ("graph", "path"):
                            graph_data = {
                                "nodes": parsed.get("nodes", []),
                                "edges": parsed.get("edges", []),
                                "queried_persons": parsed.get("queried_persons", []),
                                "path_length": parsed.get("path_length"),
                                "found": parsed.get("found", True),
                                "depth": parsed.get("depth")
                            }
                            # Extract sources from graph edges doc_ids (cap to 20 to keep streaming fast)
                            max_graph_sources = 20
                            for edge in parsed.get("edges", []):
                                if len(sources) >= max_graph_sources:
                                    break
                                doc_id = edge.get("doc_id")
                                if not doc_id or doc_id in seen_doc_ids:
                                    continue
                                seen_doc_ids.add(doc_id)
                                doc_data = get_document_with_metadata(doc_id)
                                sources.append({
                                    "doc_id": doc_id,
                                    "summary": doc_data.get("one_sentence_summary") if doc_data else None,
                                    "category": doc_data.get("category") if doc_data else None,
                                    "date_range": doc_data.get("date_range") if doc_data else None,
                                    "full_text": doc_data.get("full_text") if doc_data else None,
                                    "image_url": generate_presigned_url(doc_id),
                                })
                        elif isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "doc_id" in item:
                                    doc_id = item["doc_id"]
                                    if doc_id in seen_doc_ids:
                                        continue
                                    seen_doc_ids.add(doc_id)

                                    # Récupérer le document complet
                                    doc_data = get_document_with_metadata(doc_id)

                                    source_info = {
                                        "doc_id": doc_id,
                                        "summary": doc_data.get("one_sentence_summary") if doc_data else item.get("summary"),
                                        "category": item.get("category") or (doc_data.get("category") if doc_data else None),
                                        "date_range": item.get("date_range") or (doc_data.get("date_range") if doc_data else None),
                                        "full_text": doc_data.get("full_text") if doc_data else None,
                                        "image_url": generate_presigned_url(doc_id),
                                    }
                                    # Ajouter le score si disponible
                                    if "similarity" in item:
                                        source_info["score"] = item["similarity"]
                                    elif "score" in item:
                                        source_info["score"] = item["score"]
                                    elif "rank" in item:
                                        source_info["score"] = item["rank"]
                                    sources.append(source_info)
                    except (json.JSONDecodeError, TypeError):
                        pass

                    # Send graph data as separate event type (include sources too)
                    if graph_data:
                        logger.info("Sending graph event with %d nodes, %d edges",
                                   len(graph_data.get("nodes", [])), len(graph_data.get("edges", [])))
                        yield f"data: {json.dumps({'type': 'graph', 'tool_name': tool_name, 'graph': graph_data, 'sources': sources}, ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'tool_end', 'tool_name': tool_name, 'sources': sources}, ensure_ascii=False)}\n\n"

            # All tool events processed, now generate title and send done
            logger.info("All stream events processed, generating title...")

            # Générer le titre à partir du résumé de la conversation
            try:
                state = agent.get_state(config)
                messages = (state.values or {}).get("messages", []) if state else []
                logger.info("Stream title gen: state=%s, messages_count=%d", bool(state), len(messages))
                conversation_title = await generate_conversation_title(messages)
            except Exception as e:
                logger.warning("Erreur génération titre stream, fallback: %s", e, exc_info=True)
                question = request.question.strip()
                conversation_title = question[:50].rsplit(" ", 1)[0] + "..." if len(question) > 50 else question

            logger.info("Sending done event, closing stream")
            yield f"data: {json.dumps({'type': 'done', 'conversation_title': conversation_title}, ensure_ascii=False)}\n\n"

        except asyncio.CancelledError:
            logger.info("Client disconnected, stopping stream for session %s", request.session_id)
            return
        except Exception as e:
            logger.exception("Erreur /api/query/stream: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    config = {"configurable": {"thread_id": session_id}}

    try:
        state = agent.get_state(config)
        if state and state.values:
            messages = state.values.get("messages", [])
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content:
                    history.append({"role": "assistant", "content": msg.content})
            return {"session_id": session_id, "messages": history}
        return {"session_id": session_id, "messages": []}
    except Exception:
        return {"session_id": session_id, "messages": []}


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's history."""
    # Note: MemorySaver ne supporte pas la suppression directe
    # En production, utiliser un checkpointer persistant (Redis, PostgreSQL)
    return {"status": "ok", "message": f"Session {session_id} marked for clearing"}


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AGENT_PORT", 3002))
    logger.info("Démarrage API sur le port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
