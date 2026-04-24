"""
Neo4j Adapter for Causal Knowledge Graph

Provides connectivity to a Neo4j database for persistent storage and
advanced querying of the drug discovery knowledge graph.
"""

import logging
import os

try:
    from neo4j import Driver, GraphDatabase
except ImportError:
    GraphDatabase = None

logger = logging.getLogger(__name__)


class Neo4jAdapter:
    """Adapter for Neo4j database interactions."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        """
        Initialize Neo4j adapter.

        Args:
            uri: Neo4j URI (bolt or neo4j)
            user: Username
            password: Password
        """
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.driver: Driver | None = None

        if GraphDatabase is None:
            logger.warning("neo4j-driver not installed. Neo4jAdapter will be unavailable.")
        else:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                logger.info(f"Connected to Neo4j at {self.uri}")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")

    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def execute_query(self, query: str, parameters: dict[str, any] | None = None) -> list[dict[str, any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            logger.error("Neo4j driver not initialized.")
            return []

        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Neo4j query: {e}")
            return []

    def create_node(self, node_type: str, node_id: str, properties: dict[str, any] | None = None):
        """
        Create a node in Neo4j.

        Args:
            node_type: Label for the node (e.g., Molecule, Protein)
            node_id: Unique identifier for the node
            properties: Additional properties for the node
        """
        query = f"MERGE (n:{node_type} {{id: $node_id}}) SET n += $properties RETURN n"
        params = {"node_id": node_id, "properties": properties or {}}
        return self.execute_query(query, params)

    def create_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, any] | None = None,
    ):
        """
        Create a relationship between two nodes.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            rel_type: Type of relationship (e.g., TREATS, BINDS)
            properties: Additional properties for the relationship
        """
        query = (
            f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            f"SET r += $properties "
            f"RETURN r"
        )
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "properties": properties or {},
        }
        return self.execute_query(query, params)

    def get_neighbors(self, node_id: str, rel_type: str | None = None) -> list[dict[str, any]]:
        """
        Get neighbors of a node.

        Args:
            node_id: ID of the node
            rel_type: Optional relationship type filter

        Returns:
            List of neighboring nodes and relationship data
        """
        rel_str = f":{rel_type}" if rel_type else ""
        query = (
            f"MATCH (n {{id: $node_id}})-[r{rel_str}]-(m) "
            f"RETURN m.id as neighbor_id, labels(m) as labels, r as relationship"
        )
        return self.execute_query(query, {"node_id": node_id})

    def run_causal_inference(self, intervention_node: str, outcome_node: str):
        """
        Stub for causal inference using graph structure.
        In a real implementation, this might use libraries like DoWhy with graph data.
        """
        logger.info(f"Running causal inference: {intervention_node} -> {outcome_node}")
        # Implementation would involve extracting subgraphs and running causal models
        return {"causal_effect": "estimated", "confidence": 0.85}
