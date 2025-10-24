"""
Noise models for fluorescence trace simulation.

This module contains classes for adding various types of noise
to simulated fluorescence traces.
"""

from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    """Base class for noise models"""

    @abstractmethod
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add noise to a signal.

        Args:
            signal: Input signal without noise

        Returns:
            Signal with noise added
        """


class GaussianNoise(NoiseModel):
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


class OutlierModel(NoiseModel):
    """Outlier contamination model"""

    def __init__(self, outlier_rate: float, outlier_value: float):
        """
        Initialize outlier model.

        Args:
            outlier_rate: Probability of outlier per time point
            outlier_value: Value of outlier contamination
        """
        self.outlier_rate = outlier_rate
        self.outlier_value = outlier_value

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add outlier contamination to signal.

        Args:
            signal: Input signal

        Returns:
            Signal with outliers added
        """
        # Create outlier mask
        outlier_mask = np.random.random(signal.shape) < self.outlier_rate

        # Add outliers
        signal_with_outliers = signal.copy()
        signal_with_outliers[outlier_mask] += self.outlier_value

        return signal_with_outliers


class CompositeNoise(NoiseModel):
    """Composite noise model combining multiple noise sources"""

    def __init__(self, noise_models: list[NoiseModel]):
        """
        Initialize composite noise model.

        Args:
            noise_models: List of noise models to apply
        """
        self.noise_models = noise_models

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add composite noise to signal.

        Args:
            signal: Input signal

        Returns:
            Signal with all noise sources added
        """
        noisy_signal = signal.copy()
        for noise_model in self.noise_models:
            noisy_signal = noise_model.add_noise(noisy_signal)
        return noisy_signal


class PoissonNoise(NoiseModel):
    """Poisson noise model for photon counting"""

    def __init__(self, gain: float = 1.0):
        """
        Initialize Poisson noise model.

        Args:
            gain: Gain factor for signal
        """
        self.gain = gain

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Add Poisson noise to signal.

        Args:
            signal: Input signal

        Returns:
            Signal with Poisson noise added
        """
        # Ensure signal is positive for Poisson
        signal_positive = np.maximum(signal, 0)

        # Apply Poisson noise
        noisy_signal = np.random.poisson(signal_positive * self.gain) / self.gain

        return noisy_signal
