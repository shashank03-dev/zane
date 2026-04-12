"""Tests for the new external-tool adapters added in PR #9.

All tests are designed to pass whether or not the optional dependencies
(MolecularTransformer, DiffDock, TorchDrug, OpenFold, OpenMM, Pistachio) are
installed.  When a dependency is absent the adapters return structured failure
results rather than raising, which is what these tests verify.
"""

from __future__ import annotations

import pytest

from drug_discovery.evaluation.torchdrug_scorer import PropertyScoreResult, TorchDrugScorer
from drug_discovery.physics.diffdock_adapter import DiffDockAdapter, DiffDockResult
from drug_discovery.physics.openmm_adapter import MDSimulationResult, OpenMMAdapter
from drug_discovery.physics.protein_structure import OpenFoldAdapter, StructurePrediction
from drug_discovery.synthesis.pistachio_datasets import PistachioDatasetResult, PistachioDatasets
from drug_discovery.synthesis.reaction_prediction import MolecularTransformerAdapter, ReactionPrediction

# ---------------------------------------------------------------------------
# MolecularTransformerAdapter
# ---------------------------------------------------------------------------


class TestMolecularTransformerAdapter:
    def test_instantiation(self):
        adapter = MolecularTransformerAdapter(beam_size=3)
        assert adapter.beam_size == 3

    def test_predict_returns_reaction_prediction(self):
        adapter = MolecularTransformerAdapter()
        result = adapter.predict("CC(=O)O.CCN")
        assert isinstance(result, ReactionPrediction)
        assert result.reactants == "CC(=O)O.CCN"

    def test_predict_empty_smiles_returns_error(self):
        adapter = MolecularTransformerAdapter()
        result = adapter.predict("")
        assert result.success is False
        assert result.error is not None

    def test_predict_unavailable_returns_failed_result(self):
        adapter = MolecularTransformerAdapter()
        if adapter.is_available():
            pytest.skip("MolecularTransformer is installed; skip unavailability test")
        result = adapter.predict("CCO")
        assert result.success is False
        assert result.error is not None
        assert result.predictions == []

    def test_batch_predict_returns_list(self):
        adapter = MolecularTransformerAdapter()
        results = adapter.batch_predict(["CC(=O)O", "CCN"])
        assert len(results) == 2
        assert all(isinstance(r, ReactionPrediction) for r in results)

    def test_reaction_prediction_as_dict(self):
        rp = ReactionPrediction(reactants="CCO", predictions=["CC=O"], scores=[0.9], success=True)
        d = rp.as_dict()
        assert d["reactants"] == "CCO"
        assert d["predictions"] == ["CC=O"]
        assert d["success"] is True


# ---------------------------------------------------------------------------
# DiffDockAdapter
# ---------------------------------------------------------------------------


class TestDiffDockAdapter:
    def test_instantiation(self):
        adapter = DiffDockAdapter(num_poses=5)
        assert adapter.num_poses == 5

    def test_dock_returns_diffdock_result(self):
        adapter = DiffDockAdapter()
        result = adapter.dock("CCO", "/fake/protein.pdb")
        assert isinstance(result, DiffDockResult)
        assert result.ligand_smiles == "CCO"
        assert result.protein_pdb_path == "/fake/protein.pdb"

    def test_dock_empty_inputs_returns_error(self):
        adapter = DiffDockAdapter()
        result = adapter.dock("", "/fake/protein.pdb")
        assert result.success is False
        assert result.error is not None

    def test_dock_unavailable_returns_failed_result(self):
        adapter = DiffDockAdapter()
        if adapter.is_available():
            pytest.skip("DiffDock is installed; skip unavailability test")
        result = adapter.dock("CCO", "/fake/protein.pdb")
        assert result.success is False
        assert result.error is not None

    def test_best_pose_none_when_no_poses(self):
        result = DiffDockResult(ligand_smiles="CCO", protein_pdb_path="/p.pdb")
        assert result.best_pose() is None

    def test_batch_dock_returns_list(self):
        adapter = DiffDockAdapter()
        results = adapter.batch_dock(["CCO", "CC(=O)O"], "/fake/protein.pdb")
        assert len(results) == 2

    def test_diffdock_result_as_dict(self):
        result = DiffDockResult(ligand_smiles="CCO", protein_pdb_path="/p.pdb", success=False, error="unavailable")
        d = result.as_dict()
        assert d["ligand_smiles"] == "CCO"
        assert d["success"] is False


# ---------------------------------------------------------------------------
# TorchDrugScorer
# ---------------------------------------------------------------------------


class TestTorchDrugScorer:
    def test_instantiation(self):
        scorer = TorchDrugScorer(tasks=("tox21",))
        assert scorer.tasks == ("tox21",)

    def test_score_returns_property_score_result(self):
        scorer = TorchDrugScorer()
        result = scorer.score("CCO")
        assert isinstance(result, PropertyScoreResult)
        assert result.smiles == "CCO"

    def test_score_empty_smiles_returns_error(self):
        scorer = TorchDrugScorer()
        result = scorer.score("")
        assert result.success is False
        assert result.error is not None

    def test_score_unavailable_returns_failed_result(self):
        scorer = TorchDrugScorer()
        if scorer.is_available():
            pytest.skip("TorchDrug is installed; skip unavailability test")
        result = scorer.score("CCO")
        assert result.success is False
        assert result.error is not None

    def test_batch_score_returns_list(self):
        scorer = TorchDrugScorer()
        results = scorer.batch_score(["CCO", "CC(=O)O"])
        assert len(results) == 2

    def test_property_score_result_as_dict(self):
        r = PropertyScoreResult(smiles="CCO", scores={"tox21": 0.1}, success=True)
        d = r.as_dict()
        assert d["smiles"] == "CCO"
        assert d["scores"]["tox21"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# OpenFoldAdapter
# ---------------------------------------------------------------------------


class TestOpenFoldAdapter:
    def test_instantiation(self):
        adapter = OpenFoldAdapter(use_templates=False)
        assert adapter.use_templates is False

    def test_predict_structure_returns_result(self):
        adapter = OpenFoldAdapter()
        result = adapter.predict_structure("ACDEFGHIKLMNPQRSTVWY")
        assert isinstance(result, StructurePrediction)
        assert result.sequence == "ACDEFGHIKLMNPQRSTVWY"

    def test_predict_empty_sequence_returns_error(self):
        adapter = OpenFoldAdapter()
        result = adapter.predict_structure("")
        assert result.success is False
        assert result.error is not None

    def test_predict_unavailable_returns_failed_result(self):
        adapter = OpenFoldAdapter()
        if adapter.is_available():
            pytest.skip("OpenFold is installed; skip unavailability test")
        result = adapter.predict_structure("ACDE")
        assert result.success is False
        assert result.error is not None

    def test_batch_predict_returns_list(self):
        adapter = OpenFoldAdapter()
        results = adapter.batch_predict(["ACDE", "FGHI"])
        assert len(results) == 2

    def test_structure_prediction_as_dict(self):
        sp = StructurePrediction(sequence="ACDE", success=False, error="unavailable")
        d = sp.as_dict()
        assert d["sequence"] == "ACDE"
        assert d["success"] is False


# ---------------------------------------------------------------------------
# OpenMMAdapter
# ---------------------------------------------------------------------------


class TestOpenMMAdapter:
    def test_instantiation(self):
        adapter = OpenMMAdapter(temperature=310.0, num_steps=5000)
        assert adapter.temperature == 310.0
        assert adapter.num_steps == 5000

    def test_simulate_ligand_returns_result(self):
        adapter = OpenMMAdapter(num_steps=100)
        result = adapter.simulate_ligand("CCO")
        assert isinstance(result, MDSimulationResult)
        assert result.smiles == "CCO"

    def test_simulate_ligand_empty_smiles_returns_error(self):
        adapter = OpenMMAdapter()
        result = adapter.simulate_ligand("")
        assert result.success is False
        assert result.error is not None

    def test_fallback_enabled_when_openmm_absent(self):
        adapter = OpenMMAdapter(num_steps=100, use_fallback=True)
        if adapter.is_available():
            pytest.skip("OpenMM is installed; skip fallback test")
        result = adapter.simulate_ligand("CCO")
        # With fallback=True the internal estimator should produce a result
        assert isinstance(result, MDSimulationResult)
        assert result.smiles == "CCO"

    def test_simulate_complex_empty_inputs_returns_error(self):
        adapter = OpenMMAdapter()
        result = adapter.simulate_complex("", "/fake/protein.pdb")
        assert result.success is False
        assert result.error is not None

    def test_md_simulation_result_as_dict(self):
        r = MDSimulationResult(smiles="CCO", success=False, error="unavailable")
        d = r.as_dict()
        assert d["smiles"] == "CCO"
        assert d["success"] is False


# ---------------------------------------------------------------------------
# PistachioDatasets
# ---------------------------------------------------------------------------


class TestPistachioDatasets:
    def test_instantiation(self):
        loader = PistachioDatasets(limit=500, filter_drug_like=True)
        assert loader.limit == 500
        assert loader.filter_drug_like is True

    def test_load_returns_result(self):
        loader = PistachioDatasets()
        result = loader.load("/fake/reactions.json")
        assert isinstance(result, PistachioDatasetResult)
        assert result.dataset_path == "/fake/reactions.json"

    def test_load_empty_path_returns_error(self):
        loader = PistachioDatasets()
        result = loader.load("")
        assert result.success is False
        assert result.error is not None

    def test_load_unavailable_returns_failed_result(self):
        loader = PistachioDatasets()
        if loader.is_available():
            pytest.skip("Pistachio is installed; skip unavailability test")
        result = loader.load("/fake/reactions.json")
        assert result.success is False
        assert result.error is not None

    def test_load_from_directory_nonexistent(self, tmp_path):
        loader = PistachioDatasets()
        results = loader.load_from_directory(str(tmp_path / "nonexistent"))
        assert len(results) == 1
        assert results[0].success is False

    def test_pistachio_result_as_dict(self):
        r = PistachioDatasetResult(dataset_path="/p", success=False, error="unavailable")
        d = r.as_dict()
        assert d["dataset_path"] == "/p"
        assert d["success"] is False


# ---------------------------------------------------------------------------
# Integration registry
# ---------------------------------------------------------------------------


class TestNewIntegrationRegistrations:
    """Verify that all six new tools are registered in INTEGRATIONS."""

    @pytest.mark.parametrize(
        "key",
        ["molecular_transformer", "diffdock", "torchdrug", "openfold", "openmm", "pistachio"],
    )
    def test_integration_key_registered(self, key):
        from drug_discovery.integrations import INTEGRATIONS, get_integration_status

        assert key in INTEGRATIONS, f"Integration '{key}' not registered"
        status = get_integration_status(key)
        assert status.key == key
        assert status.url

    @pytest.mark.parametrize(
        "key,expected_path",
        [
            ("molecular_transformer", "external/MolecularTransformer"),
            ("diffdock", "external/DiffDock"),
            ("torchdrug", "external/torchdrug"),
            ("openfold", "external/openfold"),
            ("openmm", "external/openmm"),
            ("pistachio", "external/pistachio"),
        ],
    )
    def test_submodule_path(self, key, expected_path):
        from drug_discovery.integrations import INTEGRATIONS

        assert INTEGRATIONS[key].submodule_path == expected_path


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


class TestPipelineNewMethods:
    """Smoke-test that the pipeline exposes the new step methods."""

    def setup_method(self):
        from drug_discovery import DrugDiscoveryPipeline

        self.pipeline = DrugDiscoveryPipeline(model_type="gnn", device="cpu")

    def test_predict_reaction_method_exists(self):
        assert hasattr(self.pipeline, "predict_reaction")

    def test_predict_reaction_returns_dict(self):
        result = self.pipeline.predict_reaction("CC(=O)O.CCN")
        assert isinstance(result, dict)
        assert "success" in result

    def test_score_properties_torchdrug_returns_dict(self):
        result = self.pipeline.score_properties_torchdrug("CCO")
        assert isinstance(result, dict)
        assert "success" in result

    def test_dock_diffdock_returns_dict(self):
        result = self.pipeline.dock_diffdock("CCO", "/fake/protein.pdb")
        assert isinstance(result, dict)
        assert "success" in result

    def test_predict_protein_structure_returns_dict(self):
        result = self.pipeline.predict_protein_structure("ACDE")
        assert isinstance(result, dict)
        assert "success" in result

    def test_simulate_md_returns_dict(self):
        result = self.pipeline.simulate_md("CCO", num_steps=100)
        assert isinstance(result, dict)
        assert "success" in result

    def test_load_pistachio_reactions_returns_dict(self):
        result = self.pipeline.load_pistachio_reactions("/fake/reactions.json")
        assert isinstance(result, dict)
        assert "success" in result
