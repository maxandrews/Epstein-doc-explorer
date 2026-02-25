# CLAUDE.md — Project Knowledge for Epstein Agent API

## Stack
- **API**: FastAPI (`agent_api.py`) + MCP Server (`mcp_server.py`)
- **Agent**: LangGraph with tool-calling loop, model via OpenRouter
- **DB**: PostgreSQL (RDS) — `DATABASE_URL` in `.env`
- **Vector search**: PGVector (384d, `all-minilm-l6-v2`)
- **Storage**: S3-compatible (Tigris) for document images
- **Python**: 3.12, venv at `.venv/`

## Running
```bash
.venv/bin/python agent_api.py   # starts on port 3002
```

## Database Schema — Key Tables

### `rdf_triples` (~3M+ rows, the core relationship table)
```
id, doc_id, timestamp, actor, action, target, location,
actor_likely_type, triple_tags (JSONB), explicit_topic, implicit_topic,
sequence_order, top_cluster_ids (JSONB), created_at,
actor_canonical, target_canonical    -- PRE-COMPUTED canonical names
```

**CRITICAL**: `actor_canonical` and `target_canonical` already contain resolved canonical names (or the raw name as fallback if no alias exists). These columns ARE the resolved COALESCE(entity_aliases.canonical_name, actor/target). **Never JOIN entity_aliases at query time for graph queries** — use `actor_canonical`/`target_canonical` directly.

**Indexes** (all exist):
- `idx_rdf_triples_actor_canonical` (btree)
- `idx_rdf_triples_target_canonical` (btree)
- `idx_rdf_triples_actor_canonical_trgm` (GIN trigram — for ILIKE)
- `idx_rdf_triples_actor` / `idx_rdf_triples_target` (btree on raw names)
- `idx_rdf_triples_doc_id` (btree)

**Volume warning**: Jeffrey Epstein alone has ~2.2M rows as actor_canonical and ~780k as target_canonical. Any `WHERE actor_canonical = 'Jeffrey Epstein'` without LIMIT will return millions of rows. Always LIMIT.

### `entity_aliases` (small lookup table)
```
original_name TEXT PRIMARY KEY,
canonical_name TEXT NOT NULL,
reasoning, hop_distance_from_principal, created_at, created_by
```
Use this to resolve user input ("Bill Clinton") → canonical name. PK lookup on `original_name` is instant. Index on `canonical_name`.

### `canonical_entities`
```
canonical_name TEXT PRIMARY KEY,
hop_distance_from_principal INTEGER NOT NULL
```
Registry of canonical names. Not needed at query time — `actor_canonical`/`target_canonical` in rdf_triples already have the resolved values.

### `documents`
```
doc_id (UNIQUE), file_path, full_text, one_sentence_summary, paragraph_summary,
category, date_range_earliest, date_range_latest, content_tags (JSONB),
text_search_vector (tsvector)
```

### `all_embeddings_mv` (materialized view)
Union of document_embeddings + doj_document_embeddings with PGVector index (IVFFlat cosine).

## Writing Graph Queries — Rules

1. **Always use `actor_canonical` / `target_canonical`**, never raw `actor`/`target` with JOIN entity_aliases
2. **Always LIMIT** — high-volume entities will return millions of rows
3. **Per-person queries**: When querying multiple persons, query each separately with their own LIMIT to avoid one person starving the others
4. **Multi-person graph logic** (priority order):
   - Direct links between queried persons (`actor_canonical IN (...) AND target_canonical IN (...)`)
   - Shared connections: JOIN rdf_triples t1 ON t2 with `t1.target_canonical = t2.target_canonical` — cap at 1 intermediary
   - Complement: top distinct connections per person
5. **Self-loop filter**: Always add `AND actor_canonical != target_canonical`
6. **Name resolution**: Use `_resolve_persons()` helper which tries exact PK lookup on entity_aliases, then exact match on actor_canonical, then ILIKE fallback
7. **Dedup edges**: Use a `seen` set on `(source, target, doc_id)` to avoid duplicates from UNION queries

## Query Patterns

### Find connections between 2 people
```sql
-- Direct links
SELECT actor_canonical AS source, target_canonical AS target, action, doc_id
FROM rdf_triples
WHERE actor_canonical IN ('Person A', 'Person B')
  AND target_canonical IN ('Person A', 'Person B')
  AND actor_canonical != target_canonical
LIMIT 50;

-- Shared connection (1 intermediary)
SELECT t1.target_canonical AS shared,
       t1.action AS a_action, t1.doc_id AS a_doc_id,
       t2.action AS b_action, t2.doc_id AS b_doc_id
FROM rdf_triples t1
JOIN rdf_triples t2 ON t1.target_canonical = t2.target_canonical
WHERE t1.actor_canonical = 'Person A' AND t2.actor_canonical = 'Person B'
  AND t1.target_canonical NOT IN ('Person A', 'Person B')
  AND t1.actor_canonical != t1.target_canonical
LIMIT 1;
```

### Top connections for a person (capped)
```sql
(SELECT actor_canonical AS source, target_canonical AS target, action, doc_id
 FROM rdf_triples WHERE actor_canonical = %s AND actor_canonical != target_canonical LIMIT 25)
UNION ALL
(SELECT actor_canonical AS source, target_canonical AS target, action, doc_id
 FROM rdf_triples WHERE target_canonical = %s AND actor_canonical != target_canonical LIMIT 25)
```

### Search persons by name
```sql
-- Uses GIN trigram index for fast ILIKE
SELECT actor_canonical AS person, COUNT(*) AS cnt
FROM rdf_triples WHERE actor_canonical ILIKE '%query%'
GROUP BY actor_canonical ORDER BY cnt DESC LIMIT 20;
```

### Overview (top N most connected)
```sql
SELECT person, COUNT(*) AS degree FROM (
    SELECT actor_canonical AS person FROM rdf_triples
    UNION ALL
    SELECT target_canonical AS person FROM rdf_triples
) sub GROUP BY person ORDER BY degree DESC LIMIT 30;
```

## API Architecture

### Tool flow (agent)
User question → LangGraph agent → selects tool → tool executes SQL → returns JSON → agent synthesizes response

### Graph tools in `agent_api.py`
- `get_connection_graph(persons, depth)` → calls `get_subgraph_for_persons()`
- `find_connection_path(person1, person2)` → calls `find_shortest_path()` (bidirectional BFS)

### Streaming
`/api/query/stream` returns SSE with event types: `token`, `tool_start`, `tool_end`, `graph`, `done`, `error`

Graph data is sent as a separate `graph` event type that the frontend renders visually.

## File Map
| File | Purpose |
|------|---------|
| `agent_api.py` | FastAPI app, LangGraph agent, tools, endpoints |
| `mcp_server.py` | Search functions, graph queries, DB connection |
| `migrate_to_postgres.py` | SQLite → PostgreSQL migration |
| `migrate_to_pgvector.py` | Add vector columns + indexes |
| `create_materialized_view.py` | Create all_embeddings_mv |
| `migrate_to_neo4j.py` | Legacy — no longer used (graph is PostgreSQL now) |
