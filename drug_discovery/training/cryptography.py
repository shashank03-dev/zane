"""
Cryptographic Utilities for Federated Learning

Provides Homomorphic Encryption (HE) using TenSEAL and privacy controls
using PySyft.
"""

import logging

import numpy as np

try:
    import tenseal as ts
except ImportError:
    ts = None

logger = logging.getLogger(__name__)


class EncryptionProvider:
    """Handles encryption and decryption of model weights/gradients."""

    def __init__(self):
        self.context = None
        if ts:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            self.context.generate_relin_keys()
            self.context.global_scale = 2**40
            logger.info("TenSEAL context initialized for CKKS encryption.")

    def encrypt_vector(self, vector: list[float]) -> any:
        """Encrypt a vector of weights."""
        if not self.context:
            return None
        return ts.ckks_vector(self.context, vector)

    def decrypt_vector(self, encrypted_vector: any, secret_key: any) -> list[float]:
        """Decrypt an encrypted vector."""
        # In a real scenario, the secret key is managed securely
        return encrypted_vector.decrypt(secret_key)

    def aggregate_encrypted(self, encrypted_vectors: list[any]) -> any:
        """Sum multiple encrypted vectors (homomorphic addition)."""
        if not encrypted_vectors:
            return None

        result = encrypted_vectors[0]
        for vec in encrypted_vectors[1:]:
            result += vec
        return result


class PrivacyControl:
    """Stub for PySyft-based privacy controls (Differential Privacy)."""

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def apply_dp(self, weights: np.ndarray) -> np.ndarray:
        """Apply Laplacian noise for Differential Privacy."""
        noise = np.random.laplace(0, 1.0 / self.epsilon, weights.shape)
        return weights + noise
