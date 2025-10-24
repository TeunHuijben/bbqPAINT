"""
Theoretical power spectrum models for mCOAST.

This module contains classes implementing theoretical power spectrum models
for single-molecule blinking dynamics.
"""

from typing import Union

import numpy as np


class PowerSpectrumModel:
    """Theoretical power spectrum models"""

    @staticmethod
    def aliased_single_moments(
        k_vec: Union[np.ndarray, int],
        n_frames: int,
        k_sum: float,
        s2: float,
        pk_bg: float,
        dt: float,
    ) -> np.ndarray:
        """
        Power spectrum for single-molecule blinking with aliasing.

        This implements the theoretical power spectrum formula from the MATLAB code:
        powerSpectrumAliasedSingle_Moments_var_ksum_Pkbg

        Args:
            k_vec: Frequency indices
            n_frames: Number of frames
            k_sum: Total transition rate (k_on + k_off)
            s2: Variance parameter
            pk_bg: Background noise level
            dt: Sampling time

        Returns:
            Theoretical power spectrum
        """
        c = np.exp(-k_sum * dt)

        # Theoretical power spectrum formula
        ps_theory = (
            s2
            * (1 - c**2)
            * dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k_vec / n_frames))
            + pk_bg
        )

        return ps_theory

    @staticmethod
    def calculate_theoretical_ps(
        params: dict, k_vec: np.ndarray, n_frames: int, dt: float
    ) -> np.ndarray:
        """
        Calculate theoretical power spectrum from parameter dictionary.

        Args:
            params: Dictionary containing k_sum, s2, pk_bg
            k_vec: Frequency indices
            n_frames: Number of frames
            dt: Sampling time

        Returns:
            Theoretical power spectrum
        """
        return PowerSpectrumModel.aliased_single_moments(
            k_vec, n_frames, params["k_sum"], params["s2"], params["pk_bg"], dt
        )

    @staticmethod
    def lorentzian_model(
        freq_vec: np.ndarray, k_sum: float, s2: float, pk_bg: float, dt: float
    ) -> np.ndarray:
        """
        Continuous Lorentzian power spectrum model.

        Args:
            freq_vec: Frequency vector
            k_sum: Total transition rate
            s2: Variance parameter
            pk_bg: Background noise level
            dt: Sampling time

        Returns:
            Continuous power spectrum
        """
        # Lorentzian power spectrum
        ps_continuous = s2 * dt / (1 + (2 * np.pi * freq_vec / k_sum) ** 2) + pk_bg

        return ps_continuous

    @staticmethod
    def multi_emitter_model(
        k_vec: np.ndarray,
        n_frames: int,
        n_emitters: int,
        k_sum: float,
        s2: float,
        pk_bg: float,
        dt: float,
    ) -> np.ndarray:
        """
        Power spectrum model for multiple emitters.

        Args:
            k_vec: Frequency indices
            n_frames: Number of frames
            n_emitters: Number of emitters
            k_sum: Total transition rate
            s2: Variance parameter
            pk_bg: Background noise level
            dt: Sampling time

        Returns:
            Multi-emitter power spectrum
        """
        # Single emitter power spectrum
        ps_single = PowerSpectrumModel.aliased_single_moments(
            k_vec, n_frames, k_sum, s2, pk_bg, dt
        )

        # Multi-emitter power spectrum (scaled by number of emitters)
        ps_multi = n_emitters * ps_single

        return ps_multi
