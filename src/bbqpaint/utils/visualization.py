"""
Visualization utilities for bbqPAINT.

This module contains classes for plotting fluorescence traces, power spectra,
bispectra, and analysis results.
"""

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


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
            ax.set_xscale("log")
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
                np.log10(np.abs(bispectrum_plot)),
                cmap="viridis",
                origin="lower",
                vmin=1e-8,
                vmax=0,
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

    def plot_comprehensive_analysis(
        self,
        intensity: np.ndarray,
        time_points: np.ndarray,
        results: Any,
        analysis_params: Any,
        sim_params: Optional[Any] = None,
    ) -> plt.Figure:
        """
        Create comprehensive 6-panel analysis figure.

        Panels:
        1. Intensity trace with mean line
        2. Intensity histogram (horizontal)
        3. Power spectrum (log-log) with blocked data, errors, and fit
        4. Power spectrum value distribution
        5. Bispectrum (half experimental, half theoretical)
        6. Bispectrum value distribution

        Args:
            intensity: Intensity time trace
            time_points: Time points array
            results: AnalysisResults object with fitted parameters
            analysis_params: AnalysisParameters object
            sim_params: Optional SimulationParameters (for displaying true values)

        Returns:
            Matplotlib figure
        """
        from scipy import stats

        from bbqpaint.analysis.bispectrum import BispectrumAnalyzer
        from bbqpaint.analysis.power_spectrum import PowerSpectrumAnalyzer
        from bbqpaint.utils.statistics import block_bispectrum, block_power_spectrum

        fig = plt.figure(figsize=(15, 10))

        # Panel 1: Intensity trace
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(time_points, intensity, "k-", linewidth=0.5, alpha=0.7)
        ax1.axhline(np.mean(intensity), color="r", linestyle="--", linewidth=1.5)
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Intensity (a.u.)")
        ax1.set_title("Fluorescence Trace")
        ax1.grid(True, alpha=0.3)

        # Panel 2: Intensity histogram (rotated)
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(
            intensity,
            bins=np.linspace(np.min(intensity), np.max(intensity), 100),
            orientation="horizontal",
            color="skyblue",
        )
        ax2.set_ylabel("Intensity (a.u.)")
        ax2.set_xlabel("Count")
        ax2.set_title("Intensity Distribution")
        ax2.grid(True, alpha=0.3)

        # Panel 3: Power spectrum with blocking
        if results.power_spectrum is not None:
            ax3 = plt.subplot(2, 3, 3)

            # Calculate frequency vector
            n_points = len(intensity)
            freq_vec = np.arange(1, len(results.power_spectrum) + 1) / (
                n_points * analysis_params.dt
            )

            # Block the power spectrum for cleaner visualization
            blocked_freq, blocked_ps, blocked_err = block_power_spectrum(
                freq_vec, results.power_spectrum
            )

            # Plot blocked data with error bars
            ax3.errorbar(
                blocked_freq,
                blocked_ps,
                yerr=blocked_err,
                fmt="o",
                markersize=4,
                color="blue",
                ecolor="lightblue",
                capsize=3,
                label="Data (blocked)",
            )

            # Calculate and plot theoretical fit
            if results.k_sum_fit is not None and results.s2_fit is not None:
                ps_analyzer = PowerSpectrumAnalyzer(
                    dt=analysis_params.dt, n_chops=analysis_params.n_chops
                )
                # Calculate k_vec for theoretical model
                k_vec = freq_vec * n_points * analysis_params.dt
                theoretical_ps = ps_analyzer.calculate_theoretical_ps(
                    k_vec,
                    results.k_sum_fit,
                    results.s2_fit,
                    results.pk_bg_fit,
                    n_points,
                )
                ax3.plot(freq_vec, theoretical_ps, "r-", linewidth=2, label="Fit")

            ax3.set_xscale("log")
            ax3.set_yscale("log")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.set_ylabel("Power Spectrum")
            ax3.set_title("Power Spectrum Analysis")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Panel 4: Power spectrum value distribution
        if results.power_spectrum is not None:
            ax4 = plt.subplot(2, 3, 4)

            # Normalize each P_k by its EXPECTED (fitted) value P_k^(avg).
            # The raw periodogram follows an exponential distribution
            # around the Lorentzian fit, so P_k / P_k^(fit) ~ Exp(1).
            n_points = len(intensity)
            freq_vec = np.arange(1, len(results.power_spectrum) + 1) / (
                n_points * analysis_params.dt
            )
            if results.k_sum_fit is not None and results.s2_fit is not None:
                ps_analyzer = PowerSpectrumAnalyzer(
                    dt=analysis_params.dt, n_chops=analysis_params.n_chops
                )
                k_vec = freq_vec * n_points * analysis_params.dt
                theoretical_ps = ps_analyzer.calculate_theoretical_ps(
                    k_vec,
                    results.k_sum_fit,
                    results.s2_fit,
                    results.pk_bg_fit,
                    n_points,
                )
                ps_normalized = results.power_spectrum / theoretical_ps
            else:
                ps_normalized = results.power_spectrum / np.mean(results.power_spectrum)

            bin_width = 1.0
            bin_edges = np.arange(0, 11 + bin_width, bin_width)
            bin_centers = bin_edges[:-1] + bin_width / 2
            bin_counts, _ = np.histogram(ps_normalized, bins=bin_edges)

            # Expected counts for an Exp(1) distribution (+/- Poisson error)
            expected_counts = len(ps_normalized) * (
                stats.expon.cdf(bin_edges[1:]) - stats.expon.cdf(bin_edges[:-1])
            )

            ax4.bar(
                bin_centers,
                bin_counts,
                width=bin_width,
                color="skyblue",
                edgecolor="black",
                label="$P_k/P_k^{(\\mathrm{avg})}$",
            )
            ax4.errorbar(
                bin_centers,
                expected_counts,
                yerr=np.sqrt(expected_counts),
                fmt=".",
                color="red",
                markersize=10,
                linewidth=2,
                label="Expected counts",
            )
            ax4.set_yscale("log")
            ax4.set_xlabel("$P_k / P_k^{(\\mathrm{avg})}$")
            ax4.set_ylabel("Counts")
            ax4.set_title("PS Value Distribution")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # Panels 5 & 6 share the FULL bispectrum (B_raw), its theoretical model
        # (B_fit) and variance, so compute them once here.
        if results.bispectrum is not None:
            chopped_traces = PowerSpectrumAnalyzer(
                analysis_params.dt, analysis_params.n_chops
            ).chop_trace(intensity)
            chop_length = chopped_traces.shape[0]

            bs_analyzer = BispectrumAnalyzer(dt=analysis_params.dt)

            # w_max = 6 rad/s (for frequency cutoff)
            freq_max_rad = 6.0  # TODO: hardcoded value
            bs_full, freq_1_full, freq_2_full = bs_analyzer.calculate_full_bispectrum(
                chopped_traces, freq_max=freq_max_rad
            )

            have_fit = results.k_sum_fit is not None and results.c3_fit is not None
            if have_fit:
                # Convert freq matrices to k-indices for theoretical calculation
                k1_indices = freq_1_full * chop_length * analysis_params.dt
                k2_indices = freq_2_full * chop_length * analysis_params.dt

                theoretical_bs_full = (
                    results.n_emitters_fit
                    * bs_analyzer.calculate_theoretical_bs(
                        k1_indices,
                        k2_indices,
                        results.k_sum_fit,
                        results.c3_fit,
                        chop_length,
                    )
                )
                bs_var_full = bs_analyzer.calculate_variance(
                    k1_indices,
                    k2_indices,
                    analysis_params.n_chops,
                    results.k_sum_fit,
                    results.s2_fit,
                    results.pk_bg_fit,
                    chop_length,
                )

        # Panel 5: Bispectrum (half experimental, half theoretical)
        if results.bispectrum is not None:
            ax5 = plt.subplot(2, 3, 5)

            # Block for visualization (num_blocks_goal=75)
            blocked_bs, blocked_f1, blocked_f2 = block_bispectrum(
                bs_full,
                freq_1_full,
                freq_2_full,
                num_blocks_goal=75,  # TODO: hardcoded value
            )

            if have_fit:
                # Block theoretical bispectrum
                blocked_bs_theo, _, _ = block_bispectrum(
                    theoretical_bs_full, freq_1_full, freq_2_full, num_blocks_goal=75
                )

                # Create half-half plot using LOG SCALE
                bs_exp_log = np.log(np.abs(blocked_bs) + 1e-10)
                bs_theo_log = np.log(np.abs(blocked_bs_theo) + 1e-10)

                # Flip upside down
                bs_exp_flipped = np.flipud(bs_exp_log)
                bs_theo_flipped = np.flipud(bs_theo_log)

                # Create the half-half: upper triangle from exp, lower from theory
                combined_bs = (
                    np.triu(bs_exp_flipped, k=1) + np.triu(bs_theo_flipped, k=0).T
                )

                # Also flip the frequency axes to match
                blocked_f1_flipped = np.flipud(blocked_f1)
                blocked_f2_flipped = np.flipud(blocked_f2)
            else:
                combined_bs = np.log(np.abs(blocked_bs) + 1e-10)
                blocked_f1_flipped = np.flipud(blocked_f1)
                blocked_f2_flipped = np.flipud(blocked_f2)

            im = ax5.pcolormesh(
                blocked_f1_flipped,
                blocked_f2_flipped,
                combined_bs,
                shading="auto",
                cmap="viridis",
            )

            # Add diagonal line to show separation between exp and theory
            f_extent = [blocked_f1_flipped.min(), blocked_f1_flipped.max()]
            ax5.plot(f_extent, f_extent, "w-", linewidth=2, alpha=0.7)

            ax5.set_xlabel("$f_{k_1}$ [s$^{-1}$]")
            ax5.set_ylabel("$f_{k_2}$ [s$^{-1}$]")
            ax5.set_title("Bispectrum $B(k_1,k_2)$")
            ax5.set_aspect("equal")
            cbar = plt.colorbar(im, ax=ax5)
            cbar.set_label("[s$^2$]")

        # Panel 6: Bispectrum value distribution
        if results.bispectrum is not None and have_fit:
            ax6 = plt.subplot(2, 3, 6)

            # Per-point z-score: (B_raw - B_fit) / sqrt(variance * L*dt/2).
            # For the noise-dominated region this is standard normal, N(0,1).
            cor_fac = chop_length * analysis_params.dt / 2
            bs_normalized = (bs_full - theoretical_bs_full) / np.sqrt(
                bs_var_full * cor_fac
            )
            bs_normalized = bs_normalized.flatten()
            bs_normalized = bs_normalized[np.isfinite(bs_normalized)]

            # 14 bins over the data range (15 edges)
            edges = np.linspace(np.min(bs_normalized), np.max(bs_normalized), 15)
            bin_width = edges[1] - edges[0]
            counts, _ = np.histogram(bs_normalized, bins=edges)

            ax6.hist(
                bs_normalized,
                bins=edges,
                color="skyblue",
                edgecolor="black",
                label="Data",
            )

            # N(0,1) scaled from density to counts
            x = np.linspace(-5, 5, 100)
            gauss_counts = stats.norm.pdf(x, 0, 1) * bin_width * counts.sum()
            ax6.plot(x, gauss_counts, "r-", linewidth=2, label="N(0,1)")

            ax6.set_xlim(-3, 3)
            ax6.set_xticks(range(-3, 4))
            ax6.set_xlabel("$(B - B^{\\mathrm{(avg)}})/\\sigma_B$")
            ax6.set_ylabel("Counts")
            ax6.set_title("BS Value Distribution")
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
