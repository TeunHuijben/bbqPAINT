"""
Statistical utility functions for mCOAST.

This module contains statistical utility functions including blocking analysis
and bootstrap methods for error estimation.
"""

from typing import List, Optional

import numpy as np


class StatisticalUtils:
    """Statistical utility functions"""

    @staticmethod
    def blocking_analysis(
        data: np.ndarray, block_sizes: Optional[List[int]] = None
    ) -> dict:
        """
        Blocking analysis for error estimation.

        This implements the blocking analysis from the MATLAB code:
        blockingFctExponen

        Args:
            data: Data array for analysis
            block_sizes: List of block sizes to use (if None, uses exponential spacing)

        Returns:
            Dictionary containing blocking analysis results
        """
        n_data = len(data)

        if block_sizes is None:
            # Use exponential spacing like in MATLAB code
            r = 1.2  # Growth factor
            n_max = int(
                np.floor((np.log(1 - (n_data + 1) * (1 - r))) / np.log(r) - 1) + 1
            )
            block_sizes = [int(r**i) for i in range(1, n_max)]

        results = {"block_sizes": [], "means": [], "errors": [], "variances": []}

        n_min = 1
        for block_size in block_sizes:
            if n_min + block_size - 1 > n_data:
                break

            n_max = n_min + block_size - 1

            # Extract block
            block_data = data[n_min - 1 : n_max]

            # Calculate statistics
            block_mean = np.mean(block_data)
            block_var = np.var(block_data)
            block_error = np.sqrt(block_var / block_size)

            results["block_sizes"].append(block_size)
            results["means"].append(block_mean)
            results["errors"].append(block_error)
            results["variances"].append(block_var)

            n_min = n_max + 1

        # Handle remaining data
        if n_min <= n_data:
            remaining_data = data[n_min - 1 :]
            remaining_size = len(remaining_data)

            remaining_mean = np.mean(remaining_data)
            remaining_var = np.var(remaining_data)
            remaining_error = np.sqrt(remaining_var / remaining_size)

            results["block_sizes"].append(remaining_size)
            results["means"].append(remaining_mean)
            results["errors"].append(remaining_error)
            results["variances"].append(remaining_var)

        return results

    @staticmethod
    def bootstrap_analysis(
        data: np.ndarray,
        statistic_func,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> dict:
        """
        Bootstrap analysis for uncertainty estimation.

        Args:
            data: Data array for analysis
            statistic_func: Function to calculate statistic of interest
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals

        Returns:
            Dictionary containing bootstrap results
        """
        n_data = len(data)
        bootstrap_samples = []

        # Generate bootstrap samples
        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(n_data, size=n_data, replace=True)
            bootstrap_sample = data[bootstrap_indices]
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_samples.append(bootstrap_stat)

        bootstrap_samples = np.array(bootstrap_samples)

        # Calculate statistics
        mean_stat = np.mean(bootstrap_samples)
        std_stat = np.std(bootstrap_samples)

        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)

        return {
            "mean": mean_stat,
            "std": std_stat,
            "confidence_interval": (ci_lower, ci_upper),
            "bootstrap_samples": bootstrap_samples,
        }

    @staticmethod
    def jackknife_analysis(data: np.ndarray, statistic_func) -> dict:
        """
        Jackknife analysis for bias and variance estimation.

        Args:
            data: Data array for analysis
            statistic_func: Function to calculate statistic of interest

        Returns:
            Dictionary containing jackknife results
        """
        n_data = len(data)
        jackknife_samples = []

        # Generate jackknife samples
        for i in range(n_data):
            jackknife_data = np.concatenate([data[:i], data[i + 1 :]])
            jackknife_stat = statistic_func(jackknife_data)
            jackknife_samples.append(jackknife_stat)

        jackknife_samples = np.array(jackknife_samples)

        # Calculate jackknife statistics
        jackknife_mean = np.mean(jackknife_samples)
        jackknife_var = np.var(jackknife_samples) * (n_data - 1)
        jackknife_std = np.sqrt(jackknife_var)

        # Calculate bias
        original_stat = statistic_func(data)
        bias = (n_data - 1) * (jackknife_mean - original_stat)

        return {
            "original_statistic": original_stat,
            "jackknife_mean": jackknife_mean,
            "jackknife_std": jackknife_std,
            "bias": bias,
            "bias_corrected": original_stat - bias,
            "jackknife_samples": jackknife_samples,
        }

    @staticmethod
    def calculate_autocorrelation(
        trace: np.ndarray, max_lag: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate autocorrelation function.

        Args:
            trace: Time trace
            max_lag: Maximum lag to calculate (if None, uses trace length)

        Returns:
            Autocorrelation function
        """
        if max_lag is None:
            max_lag = len(trace) - 1

        autocorr = np.correlate(trace, trace, mode="full")
        autocorr = autocorr[autocorr.size // 2 :]

        # Normalize
        autocorr = autocorr / autocorr[0]

        return autocorr[: max_lag + 1]

    @staticmethod
    def calculate_correlation_time(
        trace: np.ndarray, threshold: float = 1 / np.e
    ) -> float:
        """
        Calculate correlation time from autocorrelation function.

        Args:
            trace: Time trace
            threshold: Threshold for correlation time (default: 1/e)

        Returns:
            Correlation time
        """
        autocorr = StatisticalUtils.calculate_autocorrelation(trace)

        # Find where autocorrelation drops below threshold
        correlation_time_idx = np.where(autocorr < threshold)[0]

        if len(correlation_time_idx) > 0:
            return correlation_time_idx[0]
        else:
            return len(autocorr) - 1
