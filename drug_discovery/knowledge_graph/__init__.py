"""
Knowledge Graph for Drug Discovery

Stores and queries relationships between molecules, proteins, diseases with:
- Structured graph storage (nodes and edges)
- Vector database integration for semantic search
- Hybrid retrieval combining graph structure and embeddings
- Multi-hop reasoning and path finding
- Neo4j persistence and GNN link prediction
"""

from .graph import DrugKnowledgeGraph, KnowledgeGraphBuilder
from .ingestion import KGIngestor
from .knowledge_graph import (
    EdgeType,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    NodeType,
    VectorDatabase,
)
from .link_prediction import LinkPredictionService, LinkPredictorGNN
from .neo4j_adapter import Neo4jAdapter

__all__ = [
    "DrugKnowledgeGraph",
    "KnowledgeGraphBuilder",
    "KnowledgeGraph",
    "VectorDatabase",
    "KGNode",
    "KGEdge",
    "NodeType",
    "EdgeType",
    "Neo4jAdapter",
    "LinkPredictorGNN",
    "LinkPredictionService",
    "KGIngestor",
]
