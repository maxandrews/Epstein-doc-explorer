#!/usr/bin/env python3
"""
Migrate relationship data from PostgreSQL (rdf_triples) to Neo4j.

Creates a graph with:
- Person nodes (actors and targets)
- RELATION edges with action, doc_id, timestamp, etc.

Environment variables:
- DATABASE_URL: PostgreSQL connection string
- NEO4J_URI: Neo4j bolt URI (default: bolt://localhost:7687)
- NEO4J_USER: Neo4j username (default: neo4j)
- NEO4J_PASSWORD: Neo4j password
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_to_neo4j")

# PostgreSQL connection
DATABASE_URL = os.environ.get("DATABASE_URL")

# Neo4j connection
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")


def get_postgres_conn():
    """Get PostgreSQL connection."""
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


def get_neo4j_driver():
    """Get Neo4j driver."""
    from neo4j import GraphDatabase

    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD environment variable is required")

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def create_constraints(driver):
    """Create constraints and indexes in Neo4j."""
    with driver.session() as session:
        # Create constraint for unique Person names
        session.run("""
            CREATE CONSTRAINT person_name IF NOT EXISTS
            FOR (p:Person) REQUIRE p.name IS UNIQUE
        """)

        # Create index for faster lookups
        session.run("""
            CREATE INDEX person_name_index IF NOT EXISTS
            FOR (p:Person) ON (p.name)
        """)

        # Create full-text index for searching
        try:
            session.run("""
                CREATE FULLTEXT INDEX person_search IF NOT EXISTS
                FOR (p:Person) ON EACH [p.name]
            """)
        except Exception as e:
            logger.warning("Full-text index might already exist: %s", e)

    logger.info("Constraints and indexes created")


def migrate_relationships(driver, batch_size: int = 1000):
    """Migrate relationships from PostgreSQL to Neo4j."""
    pg_conn = get_postgres_conn()
    cursor = pg_conn.cursor()

    # Count total relationships (excluding self-loops and null canonicals)
    cursor.execute("""
        SELECT COUNT(*) as cnt FROM rdf_triples
        WHERE actor_canonical IS NOT NULL AND target_canonical IS NOT NULL
          AND actor_canonical != target_canonical
    """)
    total = cursor.fetchone()["cnt"]
    logger.info("Total relationships to migrate: %d", total)

    # Fetch all relationships using canonical names
    cursor.execute("""
        SELECT doc_id, timestamp, actor_canonical, action, target_canonical, location,
               explicit_topic, implicit_topic
        FROM rdf_triples
        WHERE actor_canonical IS NOT NULL AND target_canonical IS NOT NULL
          AND actor_canonical != target_canonical
        ORDER BY doc_id
    """)

    batch = []
    processed = 0

    for row in cursor:
        batch.append({
            "actor": row["actor_canonical"],
            "target": row["target_canonical"],
            "action": row["action"],
            "doc_id": row["doc_id"],
            "timestamp": row["timestamp"],
            "location": row["location"],
            "topic": row["explicit_topic"] or row["implicit_topic"]
        })

        if len(batch) >= batch_size:
            insert_batch(driver, batch)
            processed += len(batch)
            logger.info("Processed %d / %d relationships (%.1f%%)",
                       processed, total, 100 * processed / total)
            batch = []

    # Insert remaining batch
    if batch:
        insert_batch(driver, batch)
        processed += len(batch)
        logger.info("Processed %d / %d relationships (100%%)", processed, total)

    cursor.close()
    pg_conn.close()

    logger.info("Migration complete!")


def insert_batch(driver, batch: list[dict]):
    """Insert a batch of relationships into Neo4j."""
    with driver.session() as session:
        session.run("""
            UNWIND $batch AS rel
            MERGE (actor:Person {name: rel.actor})
            MERGE (target:Person {name: rel.target})
            CREATE (actor)-[:RELATION {
                action: rel.action,
                doc_id: rel.doc_id,
                timestamp: rel.timestamp,
                location: rel.location,
                topic: rel.topic
            }]->(target)
        """, batch=batch)


def get_stats(driver) -> dict:
    """Get statistics about the Neo4j graph."""
    with driver.session() as session:
        # Count nodes
        result = session.run("MATCH (p:Person) RETURN count(p) as count")
        person_count = result.single()["count"]

        # Count relationships
        result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count")
        relation_count = result.single()["count"]

        # Top connected persons
        result = session.run("""
            MATCH (p:Person)-[r:RELATION]-()
            RETURN p.name as name, count(r) as connections
            ORDER BY connections DESC
            LIMIT 20
        """)
        top_persons = [{"name": r["name"], "connections": r["connections"]}
                      for r in result]

    return {
        "total_persons": person_count,
        "total_relationships": relation_count,
        "top_connected_persons": top_persons
    }


def main():
    """Run the migration."""
    if not DATABASE_URL:
        logger.error("DATABASE_URL environment variable is required")
        return

    if not NEO4J_PASSWORD:
        logger.error("NEO4J_PASSWORD environment variable is required")
        return

    logger.info("Connecting to Neo4j at %s", NEO4J_URI)
    driver = get_neo4j_driver()

    try:
        # Test connection
        driver.verify_connectivity()
        logger.info("Neo4j connection successful")

        # Create constraints
        create_constraints(driver)

        # Migrate data
        migrate_relationships(driver)

        # Show stats
        stats = get_stats(driver)
        logger.info("Migration stats:")
        logger.info("  - Total persons: %d", stats["total_persons"])
        logger.info("  - Total relationships: %d", stats["total_relationships"])
        logger.info("  - Top 5 connected persons:")
        for p in stats["top_connected_persons"][:5]:
            logger.info("    - %s: %d connections", p["name"], p["connections"])

    finally:
        driver.close()


if __name__ == "__main__":
    main()
