"""
Drug Discovery AI Platform
A state-of-the-art autonomous AI system for drug discovery
"""

__version__ = "1.0.0"
__author__ = "AI Drug Discovery Team"

from .ai_support import LlamaSupportAssistant
from .data import DataCollector, MolecularDataset
from .models import DrugModeler, MolecularGNN, MolecularTransformer
from .pipeline import DrugDiscoveryPipeline

__all__ = [
    "DrugDiscoveryPipeline",
    "MolecularGNN",
    "MolecularTransformer",
    "DrugModeler",
    "DataCollector",
    "MolecularDataset",
    "LlamaSupportAssistant",
]
