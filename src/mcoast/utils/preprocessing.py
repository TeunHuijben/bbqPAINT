"""
Data preprocessing utilities for mCOAST.

This module contains classes for preprocessing fluorescence traces
including outlier removal, background subtraction, and detrending.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import signal


class TracePreprocessor:
    """Data preprocessing utilities"""

    def __init__(self):
        """Initialize trace preprocessor"""

    def remove_outliers(
        self, trace: np.ndarray, method: str = "iqr", factor: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from trace.

        Args:
            trace: Intensity time trace
            method: Method for outlier detection ("iqr", "zscore", "modified_zscore")
            factor: Factor for outlier detection threshold

        Returns:
            Tuple of (cleaned_trace, outlier_mask)
        """
        if method == "iqr":
            return self._remove_outliers_iqr(trace, factor)
        elif method == "zscore":
            return self._remove_outliers_zscore(trace, factor)
        elif method == "modified_zscore":
            return self._remove_outliers_modified_zscore(trace, factor)
        else:
            raise ValueError(f"Unknown outlier removal method: {method}")

    def _remove_outliers_iqr(
        self, trace: np.ndarray, factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using IQR method"""
        q1 = np.percentile(trace, 25)
        q3 = np.percentile(trace, 75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outlier_mask = (trace < lower_bound) | (trace > upper_bound)
        cleaned_trace = trace.copy()
        cleaned_trace[outlier_mask] = np.nan

        return cleaned_trace, outlier_mask

    def _remove_outliers_zscore(
        self, trace: np.ndarray, factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using Z-score method"""
        z_scores = np.abs((trace - np.mean(trace)) / np.std(trace))
        outlier_mask = z_scores > factor

        cleaned_trace = trace.copy()
        cleaned_trace[outlier_mask] = np.nan

        return cleaned_trace, outlier_mask

    def _remove_outliers_modified_zscore(
        self, trace: np.ndarray, factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using modified Z-score method"""
        median = np.median(trace)
        mad = np.median(np.abs(trace - median))
        modified_z_scores = 0.6745 * (trace - median) / mad

        outlier_mask = np.abs(modified_z_scores) > factor

        cleaned_trace = trace.copy()
        cleaned_trace[outlier_mask] = np.nan

        return cleaned_trace, outlier_mask

    def background_subtraction(
        self,
        trace: np.ndarray,
        background_method: str = "median",
        background_trace: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Subtract background from trace.

        Args:
            trace: Intensity time trace
            background_method: Method for background estimation ("median", "mean", "provided")
            background_trace: Background trace (if method is "provided")

        Returns:
            Background-subtracted trace
        """
        if background_method == "median":
            background = np.median(trace)
        elif background_method == "mean":
            background = np.mean(trace)
        elif background_method == "provided":
            if background_trace is None:
                raise ValueError(
                    "background_trace must be provided when method is 'provided'"
                )
            background = background_trace
        else:
            raise ValueError(f"Unknown background method: {background_method}")

        return trace - background

    def detrend(self, trace: np.ndarray, method: str = "linear") -> np.ndarray:
        """
        Remove trends from trace.

        Args:
            trace: Intensity time trace
            method: Detrending method ("linear", "polynomial", "savgol")

        Returns:
            Detrended trace
        """
        if method == "linear":
            return signal.detrend(trace, type="linear")
        elif method == "polynomial":
            # Fit polynomial and subtract
            x = np.arange(len(trace))
            coeffs = np.polyfit(x, trace, 1)
            trend = np.polyval(coeffs, x)
            return trace - trend
        elif method == "savgol":
            # Savitzky-Golay filter for detrending
            window_length = min(51, len(trace) // 10 * 2 + 1)  # Ensure odd
            if window_length < 3:
                window_length = 3
            return signal.savgol_filter(trace, window_length, 2)
        else:
            raise ValueError(f"Unknown detrending method: {method}")

    def smooth_trace(
        self,
        trace: np.ndarray,
        method: str = "savgol",
        window_length: Optional[int] = None,
    ) -> np.ndarray:
        """
        Smooth trace using various methods.

        Args:
            trace: Intensity time trace
            method: Smoothing method ("savgol", "moving_average", "gaussian")
            window_length: Window length for smoothing

        Returns:
            Smoothed trace
        """
        if window_length is None:
            window_length = min(51, len(trace) // 10 * 2 + 1)
            if window_length < 3:
                window_length = 3

        if method == "savgol":
            return signal.savgol_filter(trace, window_length, 2)
        elif method == "moving_average":
            return np.convolve(
                trace, np.ones(window_length) / window_length, mode="same"
            )
        elif method == "gaussian":
            sigma = window_length / 6  # 3-sigma rule
            return signal.gaussian_filter1d(trace, sigma)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def interpolate_nans(self, trace: np.ndarray, method: str = "linear") -> np.ndarray:
        """
        Interpolate NaN values in trace.

        Args:
            trace: Intensity time trace with NaN values
            method: Interpolation method ("linear", "cubic", "nearest")

        Returns:
            Trace with interpolated values
        """
        nan_mask = np.isnan(trace)
        if not np.any(nan_mask):
            return trace

        valid_indices = ~nan_mask
        valid_values = trace[valid_indices]
        valid_positions = np.where(valid_indices)[0]

        if method == "linear":
            interpolated = np.interp(
                np.arange(len(trace)), valid_positions, valid_values
            )
        elif method == "cubic":
            from scipy.interpolate import interp1d

            f = interp1d(
                valid_positions,
                valid_values,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated = f(np.arange(len(trace)))
        elif method == "nearest":
            from scipy.interpolate import interp1d

            f = interp1d(
                valid_positions,
                valid_values,
                kind="nearest",
                bounds_error=False,
                fill_value="extrapolate",
            )
            interpolated = f(np.arange(len(trace)))
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        return interpolated
