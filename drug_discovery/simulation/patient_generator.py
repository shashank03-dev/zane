"""
Synthetic Patient Generator

Uses SDV (Synthetic Data Vault) GANs to generate realistic synthetic patient
cohorts for clinical trial simulations.
"""

import logging

import pandas as pd

try:
    from sdv.evaluation import evaluate
    from sdv.tabular import GAN
except ImportError:
    GAN = None

logger = logging.getLogger(__name__)


class PatientGenerator:
    """Generates synthetic patient data for in silico trials."""

    def __init__(self):
        self.model = None
        if GAN is None:
            logger.warning("SDV not installed. Patient generation will be unavailable.")

    def train_on_real_data(self, real_data: pd.DataFrame):
        """Train the GAN on a real patient dataset."""
        if GAN is None:
            return

        logger.info(f"Training patient generator on {len(real_data)} real patient records.")
        self.model = GAN()
        self.model.fit(real_data)

    def generate_cohort(self, num_patients: int = 100) -> pd.DataFrame:
        """Generate a synthetic patient cohort."""
        if self.model is None:
            # Fallback: Generate random data if model not trained
            logger.warning("Patient generator model not trained. Generating random data.")
            return pd.DataFrame(
                {
                    "age": pd.Series([20 + (x % 60) for x in range(num_patients)]),
                    "weight": pd.Series([50 + (x % 70) for x in range(num_patients)]),
                    "gender": pd.Series(["M" if x % 2 == 0 else "F" for x in range(num_patients)]),
                    "baseline_biomarker": pd.Series([10.5 + (x % 5) for x in range(num_patients)]),
                }
            )

        return self.model.sample(num_patients)

    def validate_cohort(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
        """Validate synthetic data against real data."""
        if GAN is None:
            return 0.0
        return evaluate(synthetic_data, real_data)
