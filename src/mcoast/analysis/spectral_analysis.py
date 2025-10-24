"""
Spectral analysis classes for mCOAST.

This module contains classes for calculating power spectra and statistical moments
from fluorescence traces.
"""

from typing import Tuple

import numpy as np
from scipy import stats


class SpectralAnalyzer:
    """Main class for spectral analysis"""

    def __init__(self, dt: float, n_chops: int = 5):
        """
        Initialize spectral analyzer.

        Args:
            dt: Sampling time
            n_chops: Number of trace chops for ensemble analysis
        """
        self.dt = dt
        self.n_chops = n_chops

    def calculate_power_spectrum(
        self, trace: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power spectrum from trace.

        Args:
            trace: Intensity time trace

        Returns:
            Tuple of (frequency_vector, power_spectrum)
        """
        n_frames = len(trace)

        # Ensure even number of frames
        if n_frames % 2 == 1:
            trace = trace[:-1]
            n_frames = len(trace)

        # Calculate FFT
        fft_signal = self.dt * np.fft.fft(trace[:-1])
        power_spec_raw = fft_signal * np.conj(fft_signal)

        # Extract positive frequencies (exclude zero frequency)
        power_spec = np.real(power_spec_raw[1 : n_frames // 2]) / (n_frames * self.dt)

        # Frequency vector
        k_vec = np.arange(1, n_frames // 2)
        freq_vec = k_vec / (n_frames * self.dt)

        return freq_vec, power_spec

    def calculate_cumulants(self, trace: np.ndarray, order: int = 3) -> dict:
        """
        Calculate statistical moments and cumulants.

        Args:
            trace: Intensity time trace
            order: Maximum order of moments to calculate

        Returns:
            Dictionary containing moments and cumulants
        """
        results = {}

        # Calculate moments
        for i in range(1, order + 1):
            moment_key = f"moment_{i}"
            results[moment_key] = stats.moment(trace, moment=i)

        # Calculate cumulants
        results["mean"] = np.mean(trace)
        results["variance"] = np.var(trace)
        results["skewness"] = stats.skew(trace)

        if order >= 3:
            results["third_moment"] = stats.moment(trace, moment=3)

        return results

    def chop_trace(self, trace: np.ndarray) -> np.ndarray:
        """
        Chop trace into segments for ensemble analysis.

        Args:
            trace: Intensity time trace

        Returns:
            Matrix of chopped traces [n_chops x chop_length]
        """
        n_frames = len(trace)
        chop_length = n_frames // self.n_chops

        # Ensure chop_length is divisible by 4 (needed for bispectrum)
        chop_length = chop_length - (chop_length % 4)

        # Calculate starting indices for chops
        starts = np.linspace(0, n_frames - chop_length, self.n_chops, dtype=int)

        # Create chop matrix
        chop_matrix = np.zeros((chop_length, self.n_chops))

        for i, start in enumerate(starts):
            sub_trace = trace[start : start + chop_length]
            # Subtract mean from each chop
            chop_matrix[:, i] = sub_trace - np.mean(sub_trace)

        return chop_matrix

    def calculate_theoretical_ps(
        self, k_vec: np.ndarray, k_sum: float, s2: float, pk_bg: float, n_frames: int
    ) -> np.ndarray:
        """
        Calculate theoretical power spectrum.

        Args:
            k_vec: Frequency indices
            k_sum: Total transition rate
            s2: Variance parameter
            pk_bg: Background noise level
            n_frames: Number of frames

        Returns:
            Theoretical power spectrum
        """
        c = np.exp(-k_sum * self.dt)

        # Theoretical power spectrum formula
        ps_theory = (
            s2
            * (1 - c**2)
            * self.dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k_vec / n_frames))
            + pk_bg
        )

        return ps_theory
