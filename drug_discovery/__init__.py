"""
Drug Discovery AI Platform
A state-of-the-art autonomous AI system for drug discovery
"""

__version__ = "1.0.0"
__author__ = "AI Drug Discovery Team"

from .pipeline import DrugDiscoveryPipeline
from .models import MolecularGNN, MolecularTransformer
from .data import DataCollector, MolecularDataset

__all__ = [
    "DrugDiscoveryPipeline",
    "MolecularGNN",
    "MolecularTransformer",
    "DataCollector",
    "MolecularDataset",
]
