"""
Noise models for fluorescence trace simulation.

This module contains classes for adding Gaussian noise to simulated traces.
"""

import numpy as np


class GaussianNoise:
    """Gaussian background noise"""

    def __init__(self, sigma: float):
        """
        Initialize Gaussian noise model.

        Args:
            sigma: Standard deviation of Gaussian noise
        """
        self.sigma = sigma

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to signal.

        Args:
            signal: Input signal

        Returns:
            Signal with Gaussian noise added
        """
        noise = np.random.normal(0, self.sigma, signal.shape)
        return signal + noise
