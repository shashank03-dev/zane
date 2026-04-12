"""DiffDock adapter for diffusion-based protein–ligand docking.

This module wraps the ``gcorso/DiffDock`` external submodule and exposes a
simple, failure-safe API that integrates with the rest of ZANE.  All heavy
imports are deferred so that the rest of the codebase is unaffected when
DiffDock is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from drug_discovery.integrations import ensure_local_checkout_on_path, get_integration_status

logger = logging.getLogger(__name__)


@dataclass
class DockingPose:
    """A single docked pose returned by DiffDock."""

    rank: int
    confidence: float | None = None
    position: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "confidence": self.confidence,
            "position": self.position,
            "metadata": self.metadata,
        }


@dataclass
class DiffDockResult:
    """Outcome of a DiffDock docking run."""

    ligand_smiles: str
    protein_pdb_path: str
    poses: list[DockingPose] = field(default_factory=list)
    success: bool = False
    error: str | None = None

    def best_pose(self) -> DockingPose | None:
        """Return the highest-confidence pose, or ``None`` if there are none."""
        if not self.poses:
            return None
        return max(self.poses, key=lambda p: p.confidence if p.confidence is not None else float("-inf"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "ligand_smiles": self.ligand_smiles,
            "protein_pdb_path": self.protein_pdb_path,
            "poses": [p.as_dict() for p in self.poses],
            "success": self.success,
            "error": self.error,
        }


class DiffDockAdapter:
    """Adapter for DiffDock diffusion-based molecular docking.

    Wraps ``gcorso/DiffDock``.  When the dependency is unavailable the adapter
    logs a warning and returns a failed :class:`DiffDockResult` rather than
    raising an exception, so that pipeline code can handle it gracefully.

    Example::

        adapter = DiffDockAdapter(num_poses=5)
        result = adapter.dock("CCO", "/data/protein.pdb")
        if result.success:
            best = result.best_pose()
    """

    def __init__(self, num_poses: int = 10):
        """
        Args:
            num_poses: Number of docking poses to generate.
        """
        self.num_poses = num_poses

    def is_available(self) -> bool:
        """Return ``True`` when the DiffDock package is importable."""
        return get_integration_status("diffdock").available

    def dock(self, ligand_smiles: str, protein_pdb_path: str) -> DiffDockResult:
        """Dock *ligand_smiles* against the protein described in *protein_pdb_path*.

        Args:
            ligand_smiles: Ligand SMILES string.
            protein_pdb_path: Path to the target protein PDB file.

        Returns:
            :class:`DiffDockResult` with a ranked list of poses.  ``success``
            will be ``False`` when DiffDock is unavailable or the run fails.
        """
        if not ligand_smiles or not protein_pdb_path:
            return DiffDockResult(
                ligand_smiles=ligand_smiles,
                protein_pdb_path=protein_pdb_path,
                error="ligand_smiles and protein_pdb_path must both be non-empty",
            )

        ensure_local_checkout_on_path("diffdock")

        try:
            import diffdock

            raw_poses = diffdock.dock(ligand_smiles, protein_pdb_path, num_poses=self.num_poses)
            poses: list[DockingPose] = []
            for rank, item in enumerate(raw_poses, start=1):
                if isinstance(item, dict):
                    pose = DockingPose(
                        rank=rank,
                        confidence=item.get("confidence"),
                        position=item.get("position"),
                        metadata={k: v for k, v in item.items() if k not in ("confidence", "position")},
                    )
                else:
                    pose = DockingPose(rank=rank, confidence=float(item) if item is not None else None)
                poses.append(pose)

            return DiffDockResult(
                ligand_smiles=ligand_smiles,
                protein_pdb_path=protein_pdb_path,
                poses=poses,
                success=True,
            )
        except Exception as exc:
            logger.warning("DiffDock docking failed: %s", exc)
            return DiffDockResult(
                ligand_smiles=ligand_smiles,
                protein_pdb_path=protein_pdb_path,
                error=str(exc),
            )

    def batch_dock(self, ligand_smiles_list: list[str], protein_pdb_path: str) -> list[DiffDockResult]:
        """Dock multiple ligands against the same protein.

        Args:
            ligand_smiles_list: List of ligand SMILES strings.
            protein_pdb_path: Path to the target protein PDB file.

        Returns:
            List of :class:`DiffDockResult` objects in input order.
        """
        return [self.dock(smiles, protein_pdb_path) for smiles in ligand_smiles_list]
