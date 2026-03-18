"""
Tests for ADMET Predictor
"""

import pytest
from drug_discovery.evaluation import ADMETPredictor


class TestADMETPredictor:
    """Test ADMET prediction functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.predictor = ADMETPredictor()
        self.aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
        self.invalid_smiles = "INVALID"

    def test_lipinski_properties(self):
        """Test Lipinski properties calculation"""
        props = self.predictor.calculate_lipinski_properties(self.aspirin)

        assert props is not None
        assert 'molecular_weight' in props
        assert 'logp' in props
        assert 'h_bond_donors' in props
        assert 'h_bond_acceptors' in props

        # Aspirin should have reasonable values
        assert 100 < props['molecular_weight'] < 300
        assert props['h_bond_donors'] >= 0
        assert props['h_bond_acceptors'] >= 0

    def test_lipinski_rule(self):
        """Test Lipinski's Rule of Five"""
        result = self.predictor.check_lipinski_rule(self.aspirin)

        assert result is not None
        assert 'passes' in result
        assert 'violations' in result
        assert 'properties' in result
        assert isinstance(result['passes'], bool)

    def test_qed_calculation(self):
        """Test QED calculation"""
        qed = self.predictor.calculate_qed(self.aspirin)

        assert qed is not None
        assert 0 <= qed <= 1

    def test_toxicity_flags(self):
        """Test toxicity flag prediction"""
        flags = self.predictor.predict_toxicity_flags(self.aspirin)

        assert flags is not None
        assert 'contains_reactive_groups' in flags
        assert 'potential_pains' in flags
        assert isinstance(flags['contains_reactive_groups'], bool)
        assert isinstance(flags['potential_pains'], bool)

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES"""
        props = self.predictor.calculate_lipinski_properties(self.invalid_smiles)
        assert props is None

        qed = self.predictor.calculate_qed(self.invalid_smiles)
        assert qed is None
