"""
Test configuration
"""

import pytest


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing"""
    return {
        'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'invalid': 'INVALID_SMILES'
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing"""
    import pandas as pd
    return pd.DataFrame({
        'smiles': [
            'CC(=O)OC1=CC=CC=C1C(=O)O',
            'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        ],
        'property': [1.0, 2.0, 3.0]
    })
