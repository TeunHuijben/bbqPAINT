"""
Theoretical bispectrum models for mCOAST.

This module contains classes implementing theoretical bispectrum models
for single-molecule blinking dynamics.
"""

from typing import Union

import numpy as np


class BispectrumModel:
    """Theoretical bispectrum models"""

    @staticmethod
    def discrete_ksum_c3(
        k1_mat: Union[np.ndarray, float],
        k2_mat: Union[np.ndarray, float],
        k_sum: float,
        c3: float,
        dt: float,
        n: int,
    ) -> np.ndarray:
        """
        Discrete bispectrum for kSum and C3 model.

        This implements the theoretical bispectrum formula from the MATLAB code:
        BS_discrete_ksum_C3

        Args:
            k1_mat: k1 frequency matrix or scalar
            k2_mat: k2 frequency matrix or scalar
            k_sum: Total transition rate (k_on + k_off)
            c3: Third cumulant
            dt: Sampling time
            n: Number of elements in time trace

        Returns:
            Theoretical discrete bispectrum
        """
        c = np.exp(-k_sum * dt)
        k3_mat = k1_mat + k2_mat

        # Calculate the three terms from the MATLAB formula
        temp1 = (
            (np.cos(2 * np.pi * k1_mat / n) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / n))
            + (np.cos(2 * np.pi * k2_mat / n) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / n))
            + (np.cos(2 * np.pi * k3_mat / n) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / n))
        )

        temp2 = (
            np.cos(2 * np.pi * (k1_mat - k2_mat) / n)
            - c * (np.cos(2 * np.pi * k1_mat / n) + np.cos(2 * np.pi * k2_mat / n))
            + c**2
        ) / (
            (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / n))
            * (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / n))
        )

        temp3 = (
            np.cos(2 * np.pi * (k1_mat + k3_mat) / n)
            - c * (np.cos(2 * np.pi * k1_mat / n) + np.cos(2 * np.pi * k3_mat / n))
            + c**2
        ) / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / n)) + (
            np.cos(2 * np.pi * (k2_mat + k3_mat) / n)
            - c * (np.cos(2 * np.pi * k2_mat / n) + np.cos(2 * np.pi * k3_mat / n))
            + c**2
        ) / (
            1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / n)
        )

        # Final bispectrum formula
        bs_discrete = (
            dt**2
            * c3
            * (
                1
                + 2 * c * temp1
                + 2
                * c**2
                * (temp2 + temp3 / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / n)))
            )
        )

        return bs_discrete

    @staticmethod
    def calculate_variance(
        k1_mat: np.ndarray,
        k2_mat: np.ndarray,
        k_chops: int,
        k_sum: float,
        s2: float,
        pk_bg: float,
        dt: float,
        n: int,
    ) -> np.ndarray:
        """
        Calculate bispectrum variance for weighting.

        This implements the variance calculation from the MATLAB code:
        BS_variance_kSum_C3

        Args:
            k1_mat: k1 frequency matrix
            k2_mat: k2 frequency matrix
            k_chops: Number of chops
            k_sum: Total transition rate
            s2: Variance parameter
            pk_bg: Background noise level
            dt: Sampling time
            n: Number of elements in time trace

        Returns:
            Variance matrix for weighting
        """
        k3_mat = k1_mat + k2_mat
        c = np.exp(-k_sum * dt)

        # Calculate power spectrum values
        pk1 = (
            s2 * (1 - c**2) * dt / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / n))
            + pk_bg
        )
        pk2 = (
            s2 * (1 - c**2) * dt / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / n))
            + pk_bg
        )
        pk3 = (
            s2 * (1 - c**2) * dt / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / n))
            + pk_bg
        )

        # Bispectrum variance formula
        variance = pk1 * pk2 * pk3 / k_chops

        return variance

    @staticmethod
    def continuous_model(
        freq1: np.ndarray, freq2: np.ndarray, k_sum: float, c3: float, dt: float
    ) -> np.ndarray:
        """
        Continuous bispectrum model.

        Args:
            freq1: First frequency vector
            freq2: Second frequency vector
            k_sum: Total transition rate
            c3: Third cumulant
            dt: Sampling time

        Returns:
            Continuous bispectrum
        """
        # Simplified continuous bispectrum model
        freq3 = freq1 + freq2

        # Continuous bispectrum formula
        bs_continuous = (
            c3
            * dt**2
            / (
                (1 + (2 * np.pi * freq1 / k_sum) ** 2)
                * (1 + (2 * np.pi * freq2 / k_sum) ** 2)
                * (1 + (2 * np.pi * freq3 / k_sum) ** 2)
            )
        )

        return bs_continuous

    @staticmethod
    def multi_emitter_model(
        k1_mat: np.ndarray,
        k2_mat: np.ndarray,
        n_emitters: int,
        k_sum: float,
        c3: float,
        dt: float,
        n: int,
    ) -> np.ndarray:
        """
        Bispectrum model for multiple emitters.

        Args:
            k1_mat: k1 frequency matrix
            k2_mat: k2 frequency matrix
            n_emitters: Number of emitters
            k_sum: Total transition rate
            c3: Third cumulant
            dt: Sampling time
            n: Number of elements in time trace

        Returns:
            Multi-emitter bispectrum
        """
        # Single emitter bispectrum
        bs_single = BispectrumModel.discrete_ksum_c3(k1_mat, k2_mat, k_sum, c3, dt, n)

        # Multi-emitter bispectrum (scaled by number of emitters)
        bs_multi = n_emitters * bs_single

        return bs_multi
