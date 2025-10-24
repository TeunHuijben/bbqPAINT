"""
Statistical moment and cumulant calculations for mCOAST.

This module contains classes for calculating statistical moments and cumulants
from fluorescence traces.
"""

from typing import Dict

import numpy as np
from scipy import stats


class CumulantCalculator:
    """Statistical moment calculations"""

    @staticmethod
    def calculate_moments(trace: np.ndarray, order: int = 3) -> Dict[str, float]:
        """
        Calculate statistical moments.

        Args:
            trace: Intensity time trace
            order: Maximum order of moments to calculate

        Returns:
            Dictionary containing moments
        """
        moments = {}

        for i in range(1, order + 1):
            moment_key = f"moment_{i}"
            moments[moment_key] = stats.moment(trace, moment=i)

        return moments

    @staticmethod
    def calculate_cumulants(trace: np.ndarray, order: int = 3) -> Dict[str, float]:
        """
        Calculate cumulants from moments.

        Args:
            trace: Intensity time trace
            order: Maximum order of cumulants to calculate

        Returns:
            Dictionary containing cumulants
        """
        cumulants = {}

        # First cumulant (mean)
        cumulants["c1"] = np.mean(trace)

        # Second cumulant (variance)
        cumulants["c2"] = np.var(trace)

        # Third cumulant (skewness-related)
        if order >= 3:
            cumulants["c3"] = stats.moment(trace, moment=3)

        # Fourth cumulant (kurtosis-related)
        if order >= 4:
            cumulants["c4"] = stats.moment(trace, moment=4) - 3 * cumulants["c2"] ** 2

        return cumulants

    @staticmethod
    def calculate_central_moments(
        trace: np.ndarray, order: int = 3
    ) -> Dict[str, float]:
        """
        Calculate central moments (moments about the mean).

        Args:
            trace: Intensity time trace
            order: Maximum order of central moments to calculate

        Returns:
            Dictionary containing central moments
        """
        central_moments = {}
        mean_val = np.mean(trace)

        for i in range(1, order + 1):
            moment_key = f"central_moment_{i}"
            central_moments[moment_key] = np.mean((trace - mean_val) ** i)

        return central_moments

    @staticmethod
    def calculate_standardized_moments(
        trace: np.ndarray, order: int = 4
    ) -> Dict[str, float]:
        """
        Calculate standardized moments (normalized by standard deviation).

        Args:
            trace: Intensity time trace
            order: Maximum order of standardized moments to calculate

        Returns:
            Dictionary containing standardized moments
        """
        standardized_moments = {}
        mean_val = np.mean(trace)
        std_val = np.std(trace)

        for i in range(1, order + 1):
            moment_key = f"standardized_moment_{i}"
            standardized_moments[moment_key] = np.mean(
                ((trace - mean_val) / std_val) ** i
            )

        return standardized_moments

    @staticmethod
    def calculate_skewness_kurtosis(trace: np.ndarray) -> Dict[str, float]:
        """
        Calculate skewness and kurtosis.

        Args:
            trace: Intensity time trace

        Returns:
            Dictionary containing skewness and kurtosis
        """
        return {"skewness": stats.skew(trace), "kurtosis": stats.kurtosis(trace)}

    @staticmethod
    def calculate_theoretical_cumulants(
        k_on: float, k_off: float, n_emitters: int, single_molecule_intensity: float
    ) -> Dict[str, float]:
        """
        Calculate theoretical cumulants for blinking model.

        Args:
            k_on: On transition rate
            k_off: Off transition rate
            n_emitters: Number of emitters
            single_molecule_intensity: Single molecule intensity

        Returns:
            Dictionary containing theoretical cumulants
        """
        k_sum = k_on + k_off
        p_up = k_on / k_sum

        # Theoretical cumulants
        c1_theory = n_emitters * single_molecule_intensity * p_up
        c2_theory = n_emitters * single_molecule_intensity**2 * p_up * (1 - p_up)
        c3_theory = (
            n_emitters
            * single_molecule_intensity**3
            * p_up
            * (1 - p_up)
            * (1 - 2 * p_up)
        )

        return {"c1_theory": c1_theory, "c2_theory": c2_theory, "c3_theory": c3_theory}
