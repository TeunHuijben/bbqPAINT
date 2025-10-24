"""
Visualization utilities for mCOAST.

This module contains classes for plotting fluorescence traces, power spectra,
bispectra, and analysis results.
"""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


class Plotter:
    """Visualization utilities"""

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter.

        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use("default")  # Use default matplotlib style

    def plot_trace(
        self,
        trace: np.ndarray,
        dt: float,
        title: str = "Fluorescence Trace",
        xlabel: str = "Time (s)",
        ylabel: str = "Intensity (a.u.)",
        show_mean: bool = True,
    ) -> plt.Figure:
        """
        Plot intensity trace.

        Args:
            trace: Intensity time trace
            dt: Sampling time
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_mean: Whether to show mean line

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        time_points = np.arange(len(trace)) * dt

        ax.plot(time_points, trace, "b-", linewidth=0.5, alpha=0.7)

        if show_mean:
            mean_intensity = np.mean(trace)
            ax.axhline(
                y=mean_intensity,
                color="r",
                linestyle="--",
                label=f"Mean: {mean_intensity:.2f}",
            )
            ax.legend()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_power_spectrum(
        self,
        freq_vec: np.ndarray,
        power_spec: np.ndarray,
        ps_theory: Optional[np.ndarray] = None,
        title: str = "Power Spectrum",
        xlabel: str = "Frequency (Hz)",
        ylabel: str = "Power (a.u.)",
        log_scale: bool = True,
    ) -> plt.Figure:
        """
        Plot power spectrum.

        Args:
            freq_vec: Frequency vector
            power_spec: Power spectrum data
            ps_theory: Theoretical power spectrum (optional)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Whether to use log scale for y-axis

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(freq_vec, power_spec, "b-", linewidth=1, label="Data", alpha=0.7)

        if ps_theory is not None:
            ax.plot(freq_vec, ps_theory, "r--", linewidth=2, label="Theory")
            ax.legend()

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_bispectrum(
        self,
        bispectrum: np.ndarray,
        k1_mat: Optional[np.ndarray] = None,
        k2_mat: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        title: str = "Bispectrum",
        log_scale: bool = True,
    ) -> plt.Figure:
        """
        Plot bispectrum.

        Args:
            bispectrum: Bispectrum matrix
            k1_mat: k1 frequency matrix (optional)
            k2_mat: k2 frequency matrix (optional)
            mask: Mask for fitting region (optional)
            title: Plot title
            log_scale: Whether to use log scale

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Apply mask if provided
        if mask is not None:
            bispectrum_plot = np.ma.masked_where(~mask, bispectrum)
        else:
            bispectrum_plot = bispectrum

        # Plot bispectrum
        if log_scale:
            im = ax.imshow(
                np.abs(bispectrum_plot), cmap="viridis", norm=LogNorm(), origin="lower"
            )
        else:
            im = ax.imshow(bispectrum_plot, cmap="viridis", origin="lower")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("|Bispectrum|")

        ax.set_xlabel("k1")
        ax.set_ylabel("k2")
        ax.set_title(title)

        plt.tight_layout()
        return fig

    def plot_fit_results(
        self, data: Dict[str, Any], fitted_params: Dict[str, Any]
    ) -> plt.Figure:
        """
        Plot fitting results.

        Args:
            data: Dictionary containing experimental data
            fitted_params: Dictionary containing fitted parameters

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Trace
        if "trace" in data and "dt" in fitted_params:
            time_points = np.arange(len(data["trace"])) * fitted_params["dt"]
            axes[0, 0].plot(time_points, data["trace"], "b-", linewidth=0.5)
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Intensity")
            axes[0, 0].set_title("Fluorescence Trace")
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Power Spectrum
        if "power_spectrum" in data and "power_spectrum_theory" in fitted_params:
            freq_vec = data.get("freq_vec", np.arange(len(data["power_spectrum"])))
            axes[0, 1].plot(freq_vec, data["power_spectrum"], "b-", label="Data")
            axes[0, 1].plot(
                freq_vec, fitted_params["power_spectrum_theory"], "r--", label="Fit"
            )
            axes[0, 1].set_xlabel("Frequency (Hz)")
            axes[0, 1].set_ylabel("Power")
            axes[0, 1].set_title("Power Spectrum")
            axes[0, 1].set_yscale("log")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Parameter Summary
        axes[1, 0].axis("off")
        param_text = "Fitted Parameters:\\n"
        for key, value in fitted_params.items():
            if isinstance(value, (int, float)) and not key.endswith("_theory"):
                param_text += f"{key}: {value:.3f}\\n"
        axes[1, 0].text(
            0.1,
            0.9,
            param_text,
            transform=axes[1, 0].transAxes,
            verticalalignment="top",
            fontfamily="monospace",
        )

        # Plot 4: Residuals (if available)
        if "power_spectrum" in data and "power_spectrum_theory" in fitted_params:
            residuals = data["power_spectrum"] - fitted_params["power_spectrum_theory"]
            axes[1, 1].plot(freq_vec, residuals, "g-", linewidth=1)
            axes[1, 1].set_xlabel("Frequency (Hz)")
            axes[1, 1].set_ylabel("Residuals")
            axes[1, 1].set_title("Power Spectrum Residuals")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_parameter_uncertainty(
        self,
        param_names: list,
        param_values: list,
        param_errors: list,
        title: str = "Parameter Uncertainties",
    ) -> plt.Figure:
        """
        Plot parameter uncertainties.

        Args:
            param_names: List of parameter names
            param_values: List of parameter values
            param_errors: List of parameter errors
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(param_names))

        ax.errorbar(param_values, y_pos, xerr=param_errors, fmt="o", capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.set_xlabel("Parameter Value")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(
        self,
        correlation_matrix: np.ndarray,
        param_names: list,
        title: str = "Parameter Correlation Matrix",
    ) -> plt.Figure:
        """
        Plot parameter correlation matrix.

        Args:
            correlation_matrix: Correlation matrix
            param_names: List of parameter names
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation")

        # Set ticks and labels
        ax.set_xticks(range(len(param_names)))
        ax.set_yticks(range(len(param_names)))
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_yticklabels(param_names)

        # Add text annotations
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                ax.text(
                    j,
                    i,
                    f"{correlation_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        ax.set_title(title)

        plt.tight_layout()
        return fig
