"""
Power spectrum analysis for bbqPAINT.

This module contains the PowerSpectrumAnalyzer class for calculating power spectra,
fitting them with theoretical models, and extracting kinetic parameters.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize


class PowerSpectrumAnalyzer:
    """Power spectrum analysis with integrated fitting"""

    def __init__(self, dt: float, n_chops: int = 5):
        """
        Initialize power spectrum analyzer.

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

    def fit_power_spectrum_mle(
        self,
        power_spec: np.ndarray,
        freq_vec: np.ndarray,
        initial_guess: List[float],
        bounds: List[Tuple[float, float]],
        max_iterations: int = 1000,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Maximum likelihood estimation for power spectrum fitting.

        Args:
            power_spec: Power spectrum data
            freq_vec: Frequency vector
            initial_guess: Initial parameter guess [k_sum, s2, pk_bg]
            bounds: Parameter bounds
            max_iterations: Maximum number of iterations

        Returns:
            Tuple of (fitted_parameters, covariance_matrix)
                fitted_parameters: [k_sum, s2, pk_bg]
                covariance_matrix: 3x3 covariance matrix (or None if calculation fails)
        """

        def negative_log_likelihood(params):
            """Negative log-likelihood function"""
            k_sum, s2, pk_bg = params

            # Calculate theoretical power spectrum using frequency-dependent model
            n_frames = len(power_spec) * 2  # Approximate
            k_vec = freq_vec * n_frames * self.dt

            c = np.exp(-k_sum * self.dt)
            ps_theory = (
                s2
                * (1 - c**2)
                * self.dt
                / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k_vec / n_frames))
                + pk_bg
            )

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
        except (np.linalg.LinAlgError, ValueError) as e:
            print(
                f"Warning: Could not calculate covariance matrix for power spectrum fit: {e}"
            )
            cov_matrix = None

        return fitted_params, cov_matrix

    def _calculate_covariance_matrix(
        self,
        fitted_params: np.ndarray,
        data: np.ndarray,
        freq_vec: np.ndarray,
        nll_func,
    ) -> np.ndarray:
        """
        Calculate covariance matrix using numerical Hessian approximation.

        Args:
            fitted_params: Fitted parameters
            data: Data used for fitting
            freq_vec: Frequency vector
            nll_func: Negative log-likelihood function

        Returns:
            Covariance matrix

        Note:
            This uses finite differences to approximate the Hessian. For critical
            applications, validate uncertainties with bootstrap or other resampling methods.
        """
        n_params = len(fitted_params)
        eps = 1e-5  # Step size for finite differences

        # Initialize Hessian matrix
        hessian = np.zeros((n_params, n_params))

        # Calculate Hessian using central differences
        for i in range(n_params):
            for j in range(i, n_params):
                # Create parameter perturbations
                params_pp = fitted_params.copy()
                params_pm = fitted_params.copy()
                params_mp = fitted_params.copy()
                params_mm = fitted_params.copy()

                params_pp[i] += eps
                params_pp[j] += eps
                params_pm[i] += eps
                params_pm[j] -= eps
                params_mp[i] -= eps
                params_mp[j] += eps
                params_mm[i] -= eps
                params_mm[j] -= eps

                # Calculate second derivative using central differences
                h_ij = (
                    nll_func(params_pp)
                    - nll_func(params_pm)
                    - nll_func(params_mp)
                    + nll_func(params_mm)
                ) / (4 * eps * eps)

                hessian[i, j] = h_ij
                if i != j:
                    hessian[j, i] = h_ij

        # Covariance matrix is inverse of Hessian
        cov_matrix = np.linalg.inv(hessian)

        return cov_matrix
