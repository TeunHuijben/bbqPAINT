"""
Parameter estimation example for mCOAST.

This script demonstrates comprehensive analysis and visualization of fluorescence traces
to estimate molecular parameters. Includes 6-panel figure similar to MATLAB demo.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mcoast.analysis import AnalysisParameters, ParameterEstimator
from mcoast.analysis.fitting import FittingEngine
from mcoast.simulation import SimulationParameters, TraceGenerator
from mcoast.utils.statistics import block_bispectrum, block_power_spectrum


def main():
    """Main function demonstrating parameter estimation with comprehensive visualization."""

    print("=== mCOAST Parameter Estimation Example ===")
    print()

    # Step 1: Generate test data
    print("1. Generating synthetic fluorescence trace...")
    sim_params = SimulationParameters()
    sim_params.k_on = 0.15
    sim_params.k_off = 0.3
    sim_params.n_emitters = 3
    sim_params.dt = 0.2
    sim_params.measurement_time = 5000
    sim_params.single_molecule_intensity = 1.0
    sim_params.sigma_noise = 0.01

    generator = TraceGenerator(sim_params)
    time_points, intensity = generator.generate_trace()
    print(f"   ✓ Generated {len(intensity)} data points")
    print()

    # Step 2: Analyze the data
    print("2. Performing spectral analysis...")
    analysis_params = AnalysisParameters()
    analysis_params.dt = sim_params.dt
    analysis_params.measurement_time = sim_params.measurement_time
    analysis_params.n_chops = 5
    analysis_params.fit_power_spectrum = True
    analysis_params.fit_bispectrum = True

    estimator = ParameterEstimator(intensity, analysis_params)
    results = estimator.estimate_parameters()
    print("   ✓ Analysis completed")
    print()

    # Step 3: Print results
    print("3. Estimation Results:")
    print(f"   N = {results.n_emitters_fit:.2f} (true: {sim_params.n_emitters})")
    print(f"   k_on = {results.k_on_fit:.3f} Hz (true: {sim_params.k_on})")
    print(f"   k_off = {results.k_off_fit:.3f} Hz (true: {sim_params.k_off})")
    print(
        f"   I_single = {results.single_molecule_intensity_fit:.2f} (true: {sim_params.single_molecule_intensity})"
    )
    print()

    # Step 4: Create comprehensive 6-panel visualization
    print("4. Creating comprehensive visualization...")
    create_comprehensive_figure(
        intensity, time_points, results, sim_params, analysis_params
    )
    print("   ✓ Figure created")

    plt.show()


def create_comprehensive_figure(
    intensity, time_points, results, sim_params, analysis_params
):
    """
    Create 6-panel comprehensive visualization similar to MATLAB demo.

    Panels:
    1. Intensity trace with mean line
    2. Intensity histogram (horizontal)
    3. Power spectrum (log-log) with blocked data, errors, and fit
    4. Power spectrum value distribution
    5. Bispectrum (half experimental, half theoretical)
    6. Bispectrum value distribution
    """
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
        intensity, bins=50, orientation="horizontal", color="skyblue", edgecolor="black"
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
            fitting_engine = FittingEngine()
            theoretical_ps = fitting_engine._calculate_theoretical_ps(
                freq_vec,
                results.k_sum_fit,
                results.s2_fit,
                results.pk_bg_fit,
                analysis_params.dt,
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

        # Normalize power spectrum values
        ps_normalized = results.power_spectrum / np.mean(results.power_spectrum)

        # Plot histogram
        ax4.hist(
            ps_normalized,
            bins=30,
            density=True,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            label="Data",
        )

        # Plot theoretical exponential distribution
        x = np.linspace(0, np.max(ps_normalized), 100)
        ax4.plot(x, np.exp(-x), "r-", linewidth=2, label="Exp(-x)")

        ax4.set_xlabel("Normalized Power Spectrum")
        ax4.set_ylabel("Probability Density")
        ax4.set_title("PS Value Distribution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Panel 5: Bispectrum (half experimental, half theoretical)
    if results.bispectrum is not None:
        ax5 = plt.subplot(2, 3, 5)

        # Get bispectrum data
        freq_vec = np.arange(1, results.bispectrum.shape[0] + 1) / (
            len(intensity) * analysis_params.dt
        )
        k1_mat, k2_mat = np.meshgrid(freq_vec, freq_vec)

        # Block for visualization
        blocked_bs, blocked_k1, blocked_k2 = block_bispectrum(
            results.bispectrum, k1_mat, k2_mat, num_blocks_goal=50
        )

        # Create half-half plot (experimental on one side, theoretical on other)
        combined_bs = blocked_bs.copy()
        if results.k_sum_fit is not None and results.c3_fit is not None:
            fitting_engine = FittingEngine()
            # Calculate theoretical bispectrum for all points
            k1_flat = k1_mat.flatten()
            k2_flat = k2_mat.flatten()
            theoretical_bs_flat = fitting_engine._calculate_theoretical_bs(
                k1_flat,
                k2_flat,
                results.k_sum_fit,
                results.c3_fit,
                results.c3_offset_fit,
                analysis_params.dt,
            )
            theoretical_bs = theoretical_bs_flat.reshape(k1_mat.shape)

            # Block theoretical bispectrum
            blocked_bs_theo, _, _ = block_bispectrum(
                theoretical_bs, k1_mat, k2_mat, num_blocks_goal=50
            )

            # Use lower triangle for experimental, upper triangle for theoretical
            mask = np.triu_indices_from(combined_bs, k=1)
            combined_bs[mask] = blocked_bs_theo[mask]

        im = ax5.pcolormesh(
            blocked_k1, blocked_k2, combined_bs, shading="auto", cmap="viridis"
        )
        ax5.set_xlabel("k1 (Hz)")
        ax5.set_ylabel("k2 (Hz)")
        ax5.set_title("Bispectrum (Exp/Theory)")
        plt.colorbar(im, ax=ax5)

    # Panel 6: Bispectrum value distribution
    if results.bispectrum is not None:
        ax6 = plt.subplot(2, 3, 6)

        # Flatten bispectrum and normalize
        bs_values = results.bispectrum.flatten()
        bs_normalized = (bs_values - np.mean(bs_values)) / np.std(bs_values)

        # Plot histogram
        ax6.hist(
            bs_normalized,
            bins=30,
            density=True,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            label="Data",
        )

        # Plot theoretical normal distribution
        x = np.linspace(np.min(bs_normalized), np.max(bs_normalized), 100)
        ax6.plot(x, stats.norm.pdf(x, 0, 1), "r-", linewidth=2, label="N(0,1)")

        ax6.set_xlabel("Normalized Bispectrum")
        ax6.set_ylabel("Probability Density")
        ax6.set_title("BS Value Distribution")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
