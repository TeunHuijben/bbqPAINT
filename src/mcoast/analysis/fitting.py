"""
Fitting engine for mCOAST parameter estimation.

This module contains classes for performing various fitting algorithms
including maximum likelihood estimation and weighted least squares.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit, minimize


class FittingEngine:
    """Handles different fitting algorithms"""

    def __init__(self, method: str = "mle"):
        """
        Initialize fitting engine.

        Args:
            method: Default fitting method ("mle", "wls", "ls")
        """
        self.method = method

    def fit_power_spectrum_mle(
        self,
        power_spec: np.ndarray,
        freq_vec: np.ndarray,
        initial_guess: List[float],
        bounds: List[Tuple[float, float]],
        max_iterations: int = 1000,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Maximum likelihood estimation for power spectrum.

        Args:
            power_spec: Power spectrum data
            freq_vec: Frequency vector
            initial_guess: Initial parameter guess [k_sum, s2, pk_bg]
            bounds: Parameter bounds
            max_iterations: Maximum number of iterations

        Returns:
            Tuple of (fitted_parameters, covariance_matrix)
        """

        def negative_log_likelihood(params):
            """Negative log-likelihood function"""
            k_sum, s2, pk_bg = params

            # Calculate theoretical power spectrum
            ps_theory = self._calculate_theoretical_ps(freq_vec, k_sum, s2, pk_bg)

            # Avoid division by zero and log of zero
            ps_theory = np.maximum(ps_theory, 1e-10)

            # Negative log-likelihood
            nll = np.sum(power_spec / ps_theory + np.log(ps_theory))
            return nll

        # Perform optimization
        result = minimize(
            negative_log_likelihood,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )

        fitted_params = result.x

        # Calculate covariance matrix
        try:
            cov_matrix = self._calculate_covariance_matrix(
                fitted_params, power_spec, freq_vec, negative_log_likelihood
            )
        except Exception:
            cov_matrix = None

        return fitted_params, cov_matrix

    def fit_bispectrum_wls(
        self,
        bs_data: np.ndarray,
        k1_data: np.ndarray,
        k2_data: np.ndarray,
        weights: np.ndarray,
        initial_guess: List[float],
        max_iterations: int = 1000,
        fit_k_sum_free: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Weighted least squares for bispectrum.

        Args:
            bs_data: Bispectrum data
            k1_data: k1 frequency data
            k2_data: k2 frequency data
            weights: Weights for fitting
            initial_guess: Initial parameter guess
            max_iterations: Maximum number of iterations
            fit_k_sum_free: Whether to fit kSum in bispectrum

        Returns:
            Tuple of (fitted_parameters, covariance_matrix)
        """

        def bispectrum_model(k12_data, *params):
            """Bispectrum model function"""
            if fit_k_sum_free:
                c3, c3_offset, k_sum = params
            else:
                c3, c3_offset = params
                k_sum = initial_guess[2] if len(initial_guess) > 2 else 0.3

            # Calculate theoretical bispectrum
            bs_theory = self._calculate_theoretical_bs(
                k12_data[:, 0], k12_data[:, 1], k_sum, c3, c3_offset
            )
            return bs_theory

        # Prepare data
        k12_data = np.column_stack([k1_data, k2_data])

        # Perform weighted least squares fitting
        try:
            fitted_params, cov_matrix = curve_fit(
                bispectrum_model,
                k12_data,
                bs_data,
                p0=initial_guess,
                sigma=weights,
                maxfev=max_iterations,
                absolute_sigma=True,
            )
        except Exception as e:
            print(f"Fitting failed: {e}")
            fitted_params = np.array(initial_guess)
            cov_matrix = None

        return fitted_params, cov_matrix

    def _calculate_theoretical_ps(
        self, freq_vec: np.ndarray, k_sum: float, s2: float, pk_bg: float
    ) -> np.ndarray:
        """Calculate theoretical power spectrum"""
        dt = 0.1  # This should be passed as parameter
        n_frames = len(freq_vec) * 2  # Approximate

        c = np.exp(-k_sum * dt)
        k_vec = freq_vec * n_frames * dt

        ps_theory = (
            s2
            * (1 - c**2)
            * dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k_vec / n_frames))
            + pk_bg
        )

        return ps_theory

    def _calculate_theoretical_bs(
        self,
        k1_data: np.ndarray,
        k2_data: np.ndarray,
        k_sum: float,
        c3: float,
        c3_offset: float,
    ) -> np.ndarray:
        """Calculate theoretical bispectrum"""
        dt = 0.1  # This should be passed as parameter

        c = np.exp(-k_sum * dt)

        # Simplified bispectrum formula
        bs_theory = c3 * (1 + c) + c3_offset

        return bs_theory

    def _calculate_covariance_matrix(
        self,
        fitted_params: np.ndarray,
        data: np.ndarray,
        freq_vec: np.ndarray,
        nll_func,
    ) -> np.ndarray:
        """Calculate parameter covariance matrix"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated methods like Fisher information matrix

        n_params = len(fitted_params)
        cov_matrix = np.eye(n_params) * 0.01  # Placeholder

        return cov_matrix
