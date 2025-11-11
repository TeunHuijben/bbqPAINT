"""
Parameter estimation classes for mCOAST.

This module contains the main parameter estimation class that orchestrates
the complete analysis pipeline.
"""

import numpy as np

from mcoast.analysis.bispectrum import BispectrumAnalyzer
from mcoast.analysis.parameters import AnalysisParameters, AnalysisResults
from mcoast.analysis.power_spectrum import PowerSpectrumAnalyzer


class ParameterEstimator:
    """Main parameter estimation class - orchestrates the analysis pipeline"""

    def __init__(self, trace: np.ndarray, params: AnalysisParameters):
        """
        Initialize parameter estimator.

        Args:
            trace: Intensity time trace
            params: AnalysisParameters object
        """
        self.trace = trace
        self.params = params
        self.params.validate()

        # Initialize analyzers (each handles its own fitting)
        self.power_spectrum_analyzer = PowerSpectrumAnalyzer(params.dt, params.n_chops)
        self.bispectrum_analyzer = BispectrumAnalyzer(params.dt)

        # Initialize results container
        self.results = AnalysisResults()
        self.results.trace = trace
        self.results.dt = params.dt
        self.results.n_chops = params.n_chops
        self.results.measurement_time = params.measurement_time
        self.results.n_frames = len(trace)

    def estimate_parameters(self) -> AnalysisResults:
        """
        Complete parameter estimation pipeline.

        Returns:
            AnalysisResults object containing all fitted parameters
        """
        # Calculate power spectrum
        freq_vec, power_spec = self.power_spectrum_analyzer.calculate_power_spectrum(
            self.trace
        )
        self.results.power_spectrum = power_spec

        # Fit power spectrum if requested
        if self.params.fit_power_spectrum:
            ps_params = self.fit_power_spectrum(power_spec, freq_vec)
            self.results.k_sum_fit = ps_params["k_sum"]
            self.results.s2_fit = ps_params["s2"]
            self.results.pk_bg_fit = ps_params["pk_bg"]
            self.results.k_sum_std = ps_params["k_sum_std"]
            self.results.s2_std = ps_params["s2_std"]
            self.results.pk_bg_std = ps_params["pk_bg_std"]

        # Calculate and fit bispectrum if requested
        if self.params.fit_bispectrum:
            bs_params = self.fit_bispectrum()
            self.results.c3_fit = bs_params["c3"]
            self.results.c3_offset_fit = bs_params["c3_offset"]
            self.results.c3_std = bs_params["c3_std"]
            self.results.c3_offset_std = bs_params["c3_offset_std"]

        # Calculate derived molecular parameters
        self.calculate_molecular_parameters()

        return self.results

    def fit_power_spectrum(self, power_spec: np.ndarray, freq_vec: np.ndarray) -> dict:
        """
        Fit power spectrum to extract kSum, s2, Pkbg.

        Args:
            power_spec: Power spectrum data
            freq_vec: Frequency vector

        Returns:
            Dictionary of fitted parameters
        """
        # Prepare initial guess
        initial_guess = [
            self.params.initial_guess.k_sum,
            self.params.initial_guess.s2 or np.var(self.trace),
            self.params.initial_guess.pk_bg,
        ]

        # Prepare bounds
        bounds = [
            self.params.bounds.k_sum,
            self.params.bounds.s2,
            self.params.bounds.pk_bg,
        ]

        # Fit using maximum likelihood estimation (integrated in analyzer)
        fitted_params, cov_matrix = self.power_spectrum_analyzer.fit_power_spectrum_mle(
            power_spec,
            freq_vec,
            initial_guess,
            bounds,
            self.params.max_iterations,
        )

        # Extract parameters and uncertainties
        results = {
            "k_sum": fitted_params[0],
            "s2": fitted_params[1],
            "pk_bg": fitted_params[2],
            "k_sum_std": np.sqrt(cov_matrix[0, 0]) if cov_matrix is not None else None,
            "s2_std": np.sqrt(cov_matrix[1, 1]) if cov_matrix is not None else None,
            "pk_bg_std": np.sqrt(cov_matrix[2, 2]) if cov_matrix is not None else None,
        }

        return results

    def fit_bispectrum(self) -> dict:
        """
        Fit bispectrum to extract C3.

        Returns:
            Dictionary of fitted parameters
        """
        # Chop trace for bispectrum calculation
        chopped_traces = self.power_spectrum_analyzer.chop_trace(self.trace)

        # Calculate bispectrum
        bispectrum, k1_mat, k2_mat = self.bispectrum_analyzer.calculate_bispectrum(
            chopped_traces
        )
        self.results.bispectrum = bispectrum

        # Apply mask for fitting region
        bs_masked, k1_masked, k2_masked = self.bispectrum_analyzer.apply_mask(
            bispectrum, k1_mat, k2_mat
        )

        # Calculate weights
        weights = self.bispectrum_analyzer.calculate_variance(
            k1_masked,
            k2_masked,
            self.params.n_chops,
            self.results.k_sum_fit,
            self.results.s2_fit,
            self.results.pk_bg_fit,
            chopped_traces.shape[0],
        )

        # Prepare initial guess
        # CRITICAL: Use central moment (third cumulant), not raw moment
        # MATLAB moment(x,3) = E[(X-mean)^3], not E[X^3]
        c3_initial = self.params.initial_guess.c3 or np.mean(
            (self.trace - np.mean(self.trace)) ** 3
        )
        initial_guess = [
            c3_initial,
            self.params.initial_guess.c3_offset,
        ]

        if self.params.fit_k_sum_free:
            initial_guess.append(self.results.k_sum_fit)

        # Fit using weighted least squares (integrated in analyzer with PROPER theoretical model)
        fitted_params, cov_matrix = self.bispectrum_analyzer.fit_bispectrum_wls(
            bs_masked,
            k1_masked,
            k2_masked,
            weights,
            chopped_traces.shape[0],  # chop_length
            initial_guess,
            self.params.max_iterations,
            self.params.fit_k_sum_free,
        )

        # Extract parameters and uncertainties
        results = {
            "c3": fitted_params[0],
            "c3_offset": fitted_params[1],
            "c3_std": np.sqrt(cov_matrix[0, 0]) if cov_matrix is not None else None,
            "c3_offset_std": (
                np.sqrt(cov_matrix[1, 1]) if cov_matrix is not None else None
            ),
        }

        if self.params.fit_k_sum_free:
            results["k_sum_bs"] = fitted_params[2]
            results["k_sum_bs_std"] = (
                np.sqrt(cov_matrix[2, 2]) if cov_matrix is not None else None
            )

        return results

    def calculate_molecular_parameters(self) -> None:
        """Calculate derived molecular parameters"""
        if (
            self.results.k_sum_fit is not None
            and self.results.s2_fit is not None
            and self.results.c3_fit is not None
        ):

            # Calculate derived parameters using the formulas from MATLAB code
            i_mean = np.mean(self.trace)

            # Number of emitters
            self.results.n_emitters_fit = (i_mean**2 * self.results.s2_fit) / (
                self.results.s2_fit**2 - self.results.c3_fit * i_mean
            )

            # Single molecule intensity
            self.results.single_molecule_intensity_fit = (
                2 * self.results.s2_fit / i_mean
                - self.results.c3_fit / self.results.s2_fit
            )

            # On and off rates
            k_diff = (self.results.c3_fit * i_mean * self.results.k_sum_fit) / (
                self.results.c3_fit * i_mean - 2 * self.results.s2_fit**2
            )

            self.results.k_on_fit = (self.results.k_sum_fit + k_diff) / 2
            self.results.k_off_fit = (self.results.k_sum_fit - k_diff) / 2
