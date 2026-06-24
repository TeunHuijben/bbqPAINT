"""
Simple noise model for fluorescence trace simulation.
"""

import numpy as np


def add_gaussian_noise(signal: np.ndarray, sigma: float) -> np.ndarray:
    """
    Add Gaussian noise to signal.

    Args:
        signal: Input signal
        sigma: Standard deviation of Gaussian noise

    Returns:
        Signal with Gaussian noise added
    """
    noise = np.random.normal(0, sigma, signal.shape)
    return signal + noise
