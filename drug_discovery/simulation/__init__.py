"""ZANE Simulation — Physics-based simulation and ML-accelerated FEP."""

from .bayesian_pkpd import BayesianPKPD
from .clinical_trial import ClinicalTrialSimulator
from .coarse_grained_md import CGSimulator
from .patient_generator import PatientGenerator

__all__ = ["CGSimulator", "PatientGenerator", "BayesianPKPD", "ClinicalTrialSimulator"]

try:
    from drug_discovery.simulation.free_energy import (
        FEPConfig as FEPConfig,
    )
    from drug_discovery.simulation.free_energy import (
        FEPPipeline as FEPPipeline,
    )
    from drug_discovery.simulation.free_energy import (
        FEPSurrogateNetwork as FEPSurrogateNetwork,
    )
    from drug_discovery.simulation.free_energy import (
        generate_lambda_schedule as generate_lambda_schedule,
    )

    __all__.extend(["FEPPipeline", "FEPConfig", "FEPSurrogateNetwork", "generate_lambda_schedule"])
except ImportError:
    pass
