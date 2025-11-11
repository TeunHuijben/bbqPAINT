"""
Bispectrum analysis classes for mCOAST.

This module contains classes for calculating and analyzing bispectra
from fluorescence traces.
"""

from typing import Tuple

import numpy as np


class BispectrumAnalyzer:
    """Bispectrum calculation and analysis"""

    def __init__(self, dt: float):
        """
        Initialize bispectrum analyzer.

        Args:
            dt: Sampling time
        """
        self.dt = dt

    def calculate_bispectrum(
        self, chopped_traces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate bispectrum from chopped traces.

        Args:
            chopped_traces: Matrix of chopped traces [chop_length x n_chops]

        Returns:
            Tuple of (bispectrum, k1_matrix, k2_matrix)
        """
        chop_length, n_chops = chopped_traces.shape

        # Ensure even number of frames
        if chop_length % 2 == 1:
            chopped_traces = chopped_traces[:-1, :]
            chop_length = len(chopped_traces)

        # Calculate FFT for each chop
        fft_chops = self.dt * np.fft.fftshift(
            np.fft.fft(chopped_traces, axis=0), axes=0
        )

        # Take positive frequencies only
        fft_pos = fft_chops[chop_length // 2 :, :]

        # Calculate bispectrum dimensions
        n_b = chop_length // 2

        # Initialize bispectrum matrices
        bispectrum = np.zeros((n_b // 2, n_b))
        k1_mat = np.zeros((n_b // 2, n_b))
        k2_mat = np.zeros((n_b // 2, n_b))

        # Calculate bispectrum in triangular region
        for i in range(n_b // 2):
            for j in range(n_b):
                k1 = j
                k2 = i
                k3 = k1 + k2

                # Check if k3 is within bounds
                if k3 < n_b and k1 >= k2:
                    # Calculate bispectrum component
                    fx = fft_pos[k1, :]
                    fy = fft_pos[k2, :]
                    fz = fft_pos[k3, :]

                    # Bispectrum formula
                    bs_component = np.real(fx * fy * np.conj(fz))
                    bispectrum[i, j] = np.mean(bs_component) / (chop_length * self.dt)

                    # Store frequency indices
                    k1_mat[i, j] = k1
                    k2_mat[i, j] = k2

        return bispectrum, k1_mat, k2_mat

    def apply_mask(
        self, bispectrum: np.ndarray, k1_mat: np.ndarray, k2_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply masking for fitting region.

        Args:
            bispectrum: Bispectrum matrix
            k1_mat: k1 frequency matrix
            k2_mat: k2 frequency matrix

        Returns:
            Tuple of (masked_bispectrum, masked_k1, masked_k2)
        """
        # Apply triangular mask: k2 > 0, k1 >= k2, k1 + k2 < n_b/2
        mask = (
            (k2_mat > 0) & (k1_mat >= k2_mat) & (k1_mat + k2_mat < bispectrum.shape[0])
        )

        bs_masked = bispectrum[mask]
        k1_masked = k1_mat[mask]
        k2_masked = k2_mat[mask]

        return bs_masked, k1_masked, k2_masked

    def calculate_theoretical_bs(
        self,
        k1_mat: np.ndarray,
        k2_mat: np.ndarray,
        k_sum: float,
        c3: float,
        chop_length: int,
    ) -> np.ndarray:
        """
        Calculate theoretical bispectrum.

        Args:
            k1_mat: k1 frequency matrix
            k2_mat: k2 frequency matrix
            k_sum: Total transition rate
            c3: Third cumulant
            chop_length: Length of each chop

        Returns:
            Theoretical bispectrum
        """
        c = np.exp(-k_sum * self.dt)
        k3_mat = k1_mat + k2_mat

        # Theoretical bispectrum formula
        temp1 = (
            (np.cos(2 * np.pi * k1_mat / chop_length) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length))
            + (np.cos(2 * np.pi * k2_mat / chop_length) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length))
            + (np.cos(2 * np.pi * k3_mat / chop_length) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / chop_length))
        )

        temp2 = (
            np.cos(2 * np.pi * (k1_mat - k2_mat) / chop_length)
            - c
            * (
                np.cos(2 * np.pi * k1_mat / chop_length)
                + np.cos(2 * np.pi * k2_mat / chop_length)
            )
            + c**2
        ) / (
            (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length))
            * (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length))
        )

        temp3 = (
            np.cos(2 * np.pi * (k1_mat + k3_mat) / chop_length)
            - c
            * (
                np.cos(2 * np.pi * k1_mat / chop_length)
                + np.cos(2 * np.pi * k3_mat / chop_length)
            )
            + c**2
        ) / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length)) + (
            np.cos(2 * np.pi * (k2_mat + k3_mat) / chop_length)
            - c
            * (
                np.cos(2 * np.pi * k2_mat / chop_length)
                + np.cos(2 * np.pi * k3_mat / chop_length)
            )
            + c**2
        ) / (
            1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length)
        )

        bs_theory = (
            self.dt**2
            * c3
            * (
                1
                + 2 * c * temp1
                + 2
                * c**2
                * (
                    temp2
                    + temp3
                    / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / chop_length))
                )
            )
        )

        return bs_theory

    def calculate_variance(
        self,
        k1_mat: np.ndarray,
        k2_mat: np.ndarray,
        n_chops: int,
        k_sum: float,
        s2: float,
        pk_bg: float,
        chop_length: int,
    ) -> np.ndarray:
        """
        Calculate bispectrum variance for weighting.

        Args:
            k1_mat: k1 frequency matrix
            k2_mat: k2 frequency matrix
            n_chops: Number of chops
            k_sum: Total transition rate
            s2: Variance parameter
            pk_bg: Background noise level
            chop_length: Length of each chop

        Returns:
            Variance matrix for weighting
        """
        k3_mat = k1_mat + k2_mat
        c = np.exp(-k_sum * self.dt)

        # Calculate power spectrum values
        pk1 = (
            s2
            * (1 - c**2)
            * self.dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length))
            + pk_bg
        )
        pk2 = (
            s2
            * (1 - c**2)
            * self.dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length))
            + pk_bg
        )
        pk3 = (
            s2
            * (1 - c**2)
            * self.dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / chop_length))
            + pk_bg
        )

        # Bispectrum variance formula
        variance = pk1 * pk2 * pk3 / n_chops

        return variance
