"""
Knowledge Graph for Drug Discovery
Stores and queries relationships between molecules, proteins, diseases
"""

from .graph import DrugKnowledgeGraph, KnowledgeGraphBuilder

__all__ = ['DrugKnowledgeGraph', 'KnowledgeGraphBuilder']
