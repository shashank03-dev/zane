"""TorchDrug-based molecular property scorer.

Wraps the ``DeepGraphLearning/torchdrug`` external submodule to predict
drug-relevant molecular properties (toxicity, solubility, bioactivity, …)
using graph neural networks.  All imports are deferred and the scorer
degrades gracefully when TorchDrug is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)

_DEFAULT_TASKS = ("tox21",)


@dataclass
class PropertyScoreResult:
    """Molecular property scores from TorchDrug."""

    smiles: str
    scores: dict[str, float | None] = field(default_factory=dict)
    success: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "scores": self.scores,
            "success": self.success,
            "error": self.error,
        }


class TorchDrugScorer:
    """Score molecular properties with TorchDrug GNN models.

    Wraps ``DeepGraphLearning/torchdrug``.  When the library is not
    available every call returns a failed :class:`PropertyScoreResult` so
    that upstream pipeline code can handle missing scores gracefully.

    Example::

        scorer = TorchDrugScorer(tasks=("tox21", "bace"))
        result = scorer.score("CCO")
        print(result.scores)
    """

    def __init__(self, tasks: tuple[str, ...] = _DEFAULT_TASKS):
        """
        Args:
            tasks: Tuple of TorchDrug task names to evaluate (e.g. ``"tox21"``,
                ``"bace"``, ``"hiv"``).
        """
        self.tasks = tasks

    def is_available(self) -> bool:
        """Return ``True`` when the ``torchdrug`` package is importable."""
        return get_integration_status("torchdrug").available

    def score(self, smiles: str) -> PropertyScoreResult:
        """Predict properties for a single molecule.

        Args:
            smiles: SMILES string of the molecule to score.

        Returns:
            :class:`PropertyScoreResult` with per-task score values.  When
            TorchDrug is unavailable ``success`` will be ``False``.
        """
        if not smiles:
            return PropertyScoreResult(smiles=smiles, error="Empty SMILES")

        ensure_local_checkout_on_path("torchdrug")

        try:
            from torchdrug import data as td_data

            mol = td_data.Molecule.from_smiles(smiles)
            scores: dict[str, float | None] = {}
            for task_name in self.tasks:
                try:
                    score_val = self._run_task(mol, task_name)
                    scores[task_name] = score_val
                except Exception as task_exc:
                    logger.debug("TorchDrug task %s failed for %s: %s", task_name, smiles, task_exc)
                    scores[task_name] = None

            return PropertyScoreResult(smiles=smiles, scores=scores, success=True)
        except Exception as exc:
            logger.warning("TorchDrug scoring failed for %s: %s", smiles, exc)
            return PropertyScoreResult(smiles=smiles, error=str(exc))

    @staticmethod
    def _run_task(molecule: Any, task_name: str) -> float | None:
        """Run a single TorchDrug prediction task and return a scalar score."""
        try:
            from torchdrug import models
            from torchdrug import tasks as td_tasks
            from torchdrug.data import feature as td_feature

            task_cls = getattr(td_tasks, task_name, None)
            if task_cls is None:
                return None

            model = models.GIN(
                input_dim=td_feature.atom_default.feature_dim,
                hidden_dims=[256, 256],
                short_cut=True,
                batch_norm=True,
                concat_hidden=True,
            )
            task = task_cls(model)
            result = task(molecule)
            if result is None:
                return None
            if hasattr(result, "item"):
                return float(result.item())
            return float(result)
        except Exception:
            return None

    def batch_score(self, smiles_list: list[str]) -> list[PropertyScoreResult]:
        """Score multiple molecules.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of :class:`PropertyScoreResult` objects in input order.
        """
        return [self.score(s) for s in smiles_list]
