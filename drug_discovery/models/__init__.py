"""ZANE Models — Unified model registry with guarded imports."""

import logging
logger = logging.getLogger(__name__)
MODEL_REGISTRY = {}

try:
    from drug_discovery.models.equivariant_gnn import (
        EquivariantGNN, EquivariantGNNConfig, GaussianRBF, CosineCutoff,
        EGNNLayer, SchNetLayer, build_radius_graph)
    MODEL_REGISTRY["egnn"] = {"class": EquivariantGNN, "config": EquivariantGNNConfig, "variant": "egnn"}
    MODEL_REGISTRY["schnet"] = {"class": EquivariantGNN, "config": EquivariantGNNConfig, "variant": "schnet"}
except ImportError as e:
    logger.debug(f"Equivariant GNN not available: {e}")

try:
    from drug_discovery.models.diffusion_generator import (
        MolecularDiffusionModel, DiffusionMoleculeGenerator, DiffusionConfig)
    MODEL_REGISTRY["diffusion"] = {"class": MolecularDiffusionModel, "config": DiffusionConfig, "variant": None}
except ImportError as e:
    logger.debug(f"Diffusion generator not available: {e}")

try:
    from drug_discovery.models.gflownet import GFlowNetPolicy, GFlowNetTrainer, GFlowNetConfig
    MODEL_REGISTRY["gflownet"] = {"class": GFlowNetPolicy, "config": GFlowNetConfig, "variant": None}
except ImportError as e:
    logger.debug(f"GFlowNet not available: {e}")

try:
    from drug_discovery.models.gnn import GNNModel
    MODEL_REGISTRY["gnn"] = {"class": GNNModel, "config": None, "variant": None}
except ImportError:
    pass
try:
    from drug_discovery.models.transformer import TransformerModel
    MODEL_REGISTRY["transformer"] = {"class": TransformerModel, "config": None, "variant": None}
except ImportError:
    pass
try:
    from drug_discovery.models.ensemble import EnsembleModel
    MODEL_REGISTRY["ensemble"] = {"class": EnsembleModel, "config": None, "variant": None}
except ImportError:
    pass


def get_model(name, **kwargs):
    """Instantiate a model by name (egnn, schnet, diffusion, gflownet, gnn, transformer, ensemble)."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys())) or "(none — install torch)"
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    entry = MODEL_REGISTRY[name]
    if entry["config"] is not None:
        config_cls = entry["config"]
        kw = kwargs.copy()
        if entry.get("variant"):
            kw.setdefault("variant", entry["variant"])
        valid_keys = set(config_cls.__dataclass_fields__) if hasattr(config_cls, "__dataclass_fields__") else set()
        filtered = {k: v for k, v in kw.items() if k in valid_keys} if valid_keys else kw
        return entry["class"](config_cls(**filtered))
    return entry["class"](**kwargs)


def list_models():
    return sorted(MODEL_REGISTRY.keys())

__all__ = ["get_model", "list_models", "MODEL_REGISTRY"]
