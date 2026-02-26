#!/usr/bin/env python3
"""
Neo4j Graph Module for Epstein Document Explorer.

Provides graph operations using Neo4j instead of PostgreSQL.
Includes connection weights for edge thickness visualization.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neo4j_graph")

# Neo4j connection
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

_driver = None


def get_neo4j_driver():
    """Get or create Neo4j driver (singleton)."""
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        if not NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD environment variable is required")
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        logger.info("Neo4j driver initialized: %s", NEO4J_URI)
    return _driver


def close_driver():
    """Close Neo4j driver."""
    global _driver
    if _driver:
        _driver.close()
        _driver = None


# ============== Graph Overview ==============

_graph_overview_cache: dict[int, tuple[float, dict]] = {}
_GRAPH_OVERVIEW_TTL = 7 * 24 * 3600  # 1 week


def get_graph_overview(limit: int = 30) -> dict:
    """
    Get top N most-connected persons and their inter-connections from Neo4j.
    Returns nodes with degree and edges with weight (connection count).
    Cached in memory for 1 week.
    """
    import time
    now = time.monotonic()
    cached = _graph_overview_cache.get(limit)
    if cached and (now - cached[0]) < _GRAPH_OVERVIEW_TTL:
        return cached[1]

    driver = get_neo4j_driver()

    with driver.session() as session:
        # Step 1: Get top N persons by connection count
        result = session.run("""
            MATCH (p:Person)-[r:RELATION]-()
            WITH p.name AS name, count(r) AS degree
            ORDER BY degree DESC
            LIMIT $limit
            RETURN name, degree
        """, limit=limit)

        nodes = []
        top_persons = []
        for record in result:
            name = record["name"]
            degree = record["degree"]
            nodes.append({"id": name, "label": name, "degree": degree})
            top_persons.append(name)

        if not top_persons:
            return {"nodes": [], "edges": [], "meta": {"total_displayed_nodes": 0, "total_displayed_edges": 0, "total_edges_in_graph": 0}}

        # Step 2: Get edges between top persons with aggregated weights
        result = session.run("""
            MATCH (a:Person)-[r:RELATION]->(b:Person)
            WHERE a.name IN $persons AND b.name IN $persons AND a.name < b.name
            WITH a.name AS source, b.name AS target,
                 count(r) AS weight,
                 collect(DISTINCT r.action)[0..5] AS actions,
                 collect(DISTINCT r.doc_id)[0..10] AS doc_ids
            RETURN source, target, weight, actions, doc_ids
            ORDER BY weight DESC
        """, persons=top_persons)

        edges = []
        for record in result:
            edges.append({
                "source": record["source"],
                "target": record["target"],
                "weight": record["weight"],
                "actions": record["actions"] or [],
                "doc_ids": record["doc_ids"] or [],
            })

        # Get total edge count
        result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS total")
        total_edges = result.single()["total"]

    response = {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "total_displayed_nodes": len(nodes),
            "total_displayed_edges": len(edges),
            "total_edges_in_graph": total_edges,
        }
    }

    _graph_overview_cache[limit] = (now, response)
    return response


# ============== Search Persons ==============

def search_persons(query: str, limit: int = 20) -> list[dict]:
    """
    Search for persons by name using Neo4j full-text search.

    Args:
        query: Search query (partial name match)
        limit: Maximum results

    Returns:
        List of matching persons with connection counts
    """
    driver = get_neo4j_driver()

    with driver.session() as session:
        # Use CONTAINS for partial matching (case-insensitive)
        result = session.run("""
            MATCH (p:Person)
            WHERE toLower(p.name) CONTAINS toLower($query)
            WITH p
            MATCH (p)-[r:RELATION]-()
            RETURN p.name AS name, count(r) AS connections
            ORDER BY connections DESC
            LIMIT $limit
        """, query=query, limit=limit)

        return [{"name": record["name"], "connections": record["connections"]} for record in result]


# ============== Subgraph for Persons ==============

def get_subgraph_for_persons(persons: list[str], depth: int = 1, limit: int = 50) -> dict:
    """
    Build a focused subgraph showing how queried persons relate.

    Strategy:
      1. Direct links between queried persons
      2. Shared connections (intermediaries)
      3. Top connections per person

    Returns nodes and edges with weight (for edge thickness).
    """
    driver = get_neo4j_driver()

    with driver.session() as session:
        # Resolve names (fuzzy match)
        canonical_names = _resolve_persons_neo4j(session, persons)
        if not canonical_names:
            return {"nodes": [], "edges": [], "queried_persons": persons, "depth": depth}

        nodes_map = {}
        edges = []
        seen_edges = set()

        def add_edge(source, target, weight, actions, doc_ids):
            # Normalize edge direction
            if source > target:
                source, target = target, source
            key = (source, target)
            if key in seen_edges:
                return
            seen_edges.add(key)

            for p in (source, target):
                if p not in nodes_map:
                    nodes_map[p] = {"id": p, "label": p, "connections": 0, "is_queried": p in canonical_names}
                nodes_map[p]["connections"] += weight

            edges.append({
                "source": source,
                "target": target,
                "weight": weight,
                "actions": actions[:5] if actions else [],
                "doc_ids": doc_ids[:10] if doc_ids else [],
            })

        # === 1. Direct links between queried persons ===
        if len(canonical_names) >= 2:
            result = session.run("""
                MATCH (a:Person)-[r:RELATION]->(b:Person)
                WHERE a.name IN $persons AND b.name IN $persons AND a.name <> b.name
                WITH CASE WHEN a.name < b.name THEN a.name ELSE b.name END AS source,
                     CASE WHEN a.name < b.name THEN b.name ELSE a.name END AS target,
                     count(r) AS weight,
                     collect(DISTINCT r.action) AS actions,
                     collect(DISTINCT r.doc_id) AS doc_ids
                RETURN source, target, weight, actions, doc_ids
                ORDER BY weight DESC
                LIMIT $limit
            """, persons=canonical_names, limit=limit)

            for record in result:
                add_edge(record["source"], record["target"], record["weight"],
                        record["actions"], record["doc_ids"])

        # === 2. Shared connections (1 intermediary) ===
        if len(canonical_names) >= 2 and len(edges) < limit:
            result = session.run("""
                MATCH (a:Person)-[r1:RELATION]-(shared:Person)-[r2:RELATION]-(b:Person)
                WHERE a.name IN $persons AND b.name IN $persons
                  AND a.name <> b.name
                  AND NOT shared.name IN $persons
                WITH shared.name AS shared_name,
                     a.name AS person_a, b.name AS person_b,
                     count(DISTINCT r1) + count(DISTINCT r2) AS total_weight,
                     collect(DISTINCT r1.action) + collect(DISTINCT r2.action) AS all_actions,
                     collect(DISTINCT r1.doc_id) + collect(DISTINCT r2.doc_id) AS all_doc_ids
                ORDER BY total_weight DESC
                LIMIT 5
                RETURN shared_name, person_a, person_b, total_weight, all_actions, all_doc_ids
            """, persons=canonical_names)

            for record in result:
                shared = record["shared_name"]
                # Add edges to shared node
                add_edge(record["person_a"], shared, record["total_weight"] // 2,
                        record["all_actions"][:3], record["all_doc_ids"][:5])
                add_edge(record["person_b"], shared, record["total_weight"] // 2,
                        record["all_actions"][3:6], record["all_doc_ids"][5:10])

        # === 3. Top connections per person ===
        remaining = limit - len(edges)
        if remaining > 0:
            per_person_limit = max(5, remaining // len(canonical_names))
            for person in canonical_names:
                result = session.run("""
                    MATCH (p:Person {name: $person})-[r:RELATION]-(other:Person)
                    WHERE other.name <> $person
                    WITH other.name AS other_name,
                         count(r) AS weight,
                         collect(DISTINCT r.action) AS actions,
                         collect(DISTINCT r.doc_id) AS doc_ids
                    ORDER BY weight DESC
                    LIMIT $limit
                    RETURN other_name, weight, actions, doc_ids
                """, person=person, limit=per_person_limit)

                for record in result:
                    add_edge(person, record["other_name"], record["weight"],
                            record["actions"], record["doc_ids"])

        # Mark queried persons
        for name in canonical_names:
            if name in nodes_map:
                nodes_map[name]["is_queried"] = True

        return {
            "nodes": list(nodes_map.values()),
            "edges": edges,
            "queried_persons": canonical_names,
            "depth": depth,
        }


# ============== Expand Node ==============

def expand_node(person: str, limit: int = 50) -> dict:
    """
    Expand a single node to show its top connections.
    Used when clicking on a node in the graph.

    Returns the person's top connections with edge weights.
    The weight represents how many times these two people appear together.
    """
    driver = get_neo4j_driver()

    with driver.session() as session:
        # Resolve the person name
        resolved = _resolve_persons_neo4j(session, [person])
        if not resolved:
            return {"nodes": [], "edges": [], "center_person": person, "found": False}

        center = resolved[0]
        nodes_map = {center: {"id": center, "label": center, "connections": 0, "is_center": True}}
        edges = []

        # Get top connections with aggregated weights
        # Sample first to avoid memory issues with high-degree nodes
        result = session.run("""
            MATCH (p:Person {name: $person})-[r:RELATION]-(other:Person)
            WHERE other.name <> $person
            WITH other, r
            LIMIT 10000
            WITH other.name AS other_name,
                 count(r) AS weight,
                 collect(DISTINCT r.action)[0..5] AS actions,
                 collect(DISTINCT r.doc_id)[0..10] AS doc_ids
            ORDER BY weight DESC
            LIMIT $limit
            RETURN other_name, weight, actions, doc_ids
        """, person=center, limit=limit)

        for record in result:
            other = record["other_name"]
            weight = record["weight"]

            nodes_map[other] = {
                "id": other,
                "label": other,
                "connections": weight,
                "is_center": False
            }
            nodes_map[center]["connections"] += weight

            # Normalize edge direction
            source, target = (center, other) if center < other else (other, center)
            edges.append({
                "source": source,
                "target": target,
                "weight": weight,
                "actions": record["actions"] or [],
                "doc_ids": record["doc_ids"] or [],
            })

        return {
            "nodes": list(nodes_map.values()),
            "edges": edges,
            "center_person": center,
            "found": True,
        }


# ============== Find Shortest Path ==============

def find_shortest_path(person1: str, person2: str, max_depth: int = 5) -> dict:
    """
    Find the shortest path between two persons using Neo4j's built-in algorithm.
    """
    driver = get_neo4j_driver()

    with driver.session() as session:
        # Resolve names
        resolved = _resolve_persons_neo4j(session, [person1, person2])
        if len(resolved) < 2:
            return {
                "found": False,
                "person1": person1,
                "person2": person2,
                "message": "One or both persons not found"
            }

        start, end = resolved[0], resolved[1]

        if start == end:
            return {
                "found": True,
                "person1": person1,
                "person2": person2,
                "path_length": 0,
                "nodes": [{"id": start, "label": start}],
                "edges": [],
            }

        # Use Neo4j shortestPath
        result = session.run("""
            MATCH (a:Person {name: $start}), (b:Person {name: $end}),
                  path = shortestPath((a)-[:RELATION*1..""" + str(max_depth) + """]-(b))
            WITH path, relationships(path) AS rels, nodes(path) AS ns
            UNWIND range(0, size(rels)-1) AS i
            WITH ns, rels, i
            RETURN [n IN ns | n.name] AS node_names,
                   [r IN rels | {action: r.action, doc_id: r.doc_id}] AS rel_data
            LIMIT 1
        """, start=start, end=end)

        record = result.single()
        if not record:
            return {
                "found": False,
                "person1": person1,
                "person2": person2,
                "path_length": None,
                "message": f"No path found within {max_depth} hops"
            }

        node_names = record["node_names"]
        rel_data = record["rel_data"]

        nodes = [{"id": name, "label": name} for name in node_names]
        edges = []
        for i, rel in enumerate(rel_data):
            source, target = node_names[i], node_names[i + 1]
            if source > target:
                source, target = target, source
            edges.append({
                "source": source,
                "target": target,
                "action": rel["action"],
                "doc_id": rel["doc_id"],
                "weight": 1,
            })

        return {
            "found": True,
            "person1": person1,
            "person2": person2,
            "path_length": len(edges),
            "nodes": nodes,
            "edges": edges,
        }


# ============== Helper Functions ==============

def _resolve_persons_neo4j(session, names: list[str]) -> list[str]:
    """
    Resolve person names to exact matches in Neo4j.
    First tries exact match, then case-insensitive, then fuzzy.
    """
    resolved = []
    for name in names:
        # Try exact match first
        result = session.run("""
            MATCH (p:Person {name: $name})
            RETURN p.name AS name
            LIMIT 1
        """, name=name)
        record = result.single()
        if record:
            resolved.append(record["name"])
            continue

        # Try case-insensitive match
        result = session.run("""
            MATCH (p:Person)
            WHERE toLower(p.name) = toLower($name)
            RETURN p.name AS name
            LIMIT 1
        """, name=name)
        record = result.single()
        if record:
            resolved.append(record["name"])
            continue

        # Try contains match
        result = session.run("""
            MATCH (p:Person)
            WHERE toLower(p.name) CONTAINS toLower($name)
            RETURN p.name AS name
            ORDER BY size(p.name)
            LIMIT 1
        """, name=name)
        record = result.single()
        if record:
            resolved.append(record["name"])

    return resolved


# ============== Graph Stats ==============

def get_graph_stats() -> dict:
    """Get statistics about the Neo4j graph."""
    driver = get_neo4j_driver()

    with driver.session() as session:
        # Count nodes
        result = session.run("MATCH (p:Person) RETURN count(p) AS count")
        person_count = result.single()["count"]

        # Count relationships
        result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS count")
        relation_count = result.single()["count"]

        # Top connected persons
        result = session.run("""
            MATCH (p:Person)-[r:RELATION]-()
            RETURN p.name AS name, count(r) AS connections
            ORDER BY connections DESC
            LIMIT 10
        """)
        top_persons = [{"name": r["name"], "connections": r["connections"]} for r in result]

    return {
        "total_persons": person_count,
        "total_relationships": relation_count,
        "top_connected_persons": top_persons,
        "backend": "neo4j",
    }
