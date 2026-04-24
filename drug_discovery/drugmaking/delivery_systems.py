"""
Delivery System Definitions

Defines Lipid Nanoparticles (LNPs) and Polymeric Delivery Systems
using NetworkX for graph-based representation.
"""

import logging
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class DeliverySystem:
    """Base class for drug delivery systems."""

    name: str
    components: dict[str, float]  # Name -> molar ratio
    properties: dict[str, any] = field(default_factory=dict)
    structure: nx.Graph = field(default_factory=nx.Graph)


class LNP(DeliverySystem):
    """Lipid Nanoparticle delivery system."""

    def __init__(self, name: str, ionizable_lipid: float, helper_lipid: float, cholesterol: float, peg_lipid: float):
        components = {
            "ionizable_lipid": ionizable_lipid,
            "helper_lipid": helper_lipid,
            "cholesterol": cholesterol,
            "peg_lipid": peg_lipid,
        }
        super().__init__(name, components)
        self._build_graph()

    def _build_graph(self):
        """Construct a graph representation of the LNP structure."""
        self.structure.add_node("core", type="cargo")
        self.structure.add_node("shell", type="lipid_bilayer")
        self.structure.add_edge("core", "shell", interaction="encapsulation")

        for comp, ratio in self.components.items():
            self.structure.add_node(comp, type="lipid", ratio=ratio)
            self.structure.add_edge("shell", comp, interaction="composition")


class PolymericSystem(DeliverySystem):
    """Polymeric delivery system (e.g., PLGA, PEG)."""

    def __init__(self, name: str, polymers: dict[str, float], crosslinker_ratio: float):
        super().__init__(name, polymers)
        self.properties["crosslinker_ratio"] = crosslinker_ratio
        self._build_graph()

    def _build_graph(self):
        """Construct a graph representation of the polymer network."""
        for poly, ratio in self.components.items():
            self.structure.add_node(poly, type="polymer", ratio=ratio)

        # Add crosslinks
        polymers = list(self.components.keys())
        for i in range(len(polymers)):
            for j in range(i + 1, len(polymers)):
                self.structure.add_edge(
                    polymers[i], polymers[j], type="crosslink", density=self.properties["crosslinker_ratio"]
                )


def analyze_stability(system: DeliverySystem) -> float:
    """Analyze stability based on graph connectivity and component ratios."""
    if isinstance(system, LNP):
        # Heuristic: Balance between ionizable lipid and cholesterol
        ratio = system.components.get("ionizable_lipid", 0) / (system.components.get("cholesterol", 1) + 1e-5)
        return max(0, 1 - abs(ratio - 1.2))
    return 0.5
