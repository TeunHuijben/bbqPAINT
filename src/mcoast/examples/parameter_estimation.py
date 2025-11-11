"""
Parameter estimation example for mCOAST.

This script demonstrates comprehensive analysis and visualization of fluorescence traces
to estimate molecular parameters. Includes 6-panel figure similar to MATLAB demo.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mcoast.analysis import AnalysisParameters, ParameterEstimator
from mcoast.analysis.bispectrum import BispectrumAnalyzer
from mcoast.analysis.power_spectrum import PowerSpectrumAnalyzer
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
    sim_params.n_emitters = 4
    sim_params.dt = 0.2
    sim_params.measurement_time = 3600
    sim_params.single_molecule_intensity = 1.0
    sim_params.sigma_noise = 0.2

    generator = TraceGenerator(sim_params)
    time_points, intensity = generator.generate_trace()
    print(f"   ✓ Generated {len(intensity)} data points")
    print()

    # Step 2: Analyze the data
    print("2. Performing spectral analysis...")
    analysis_params = AnalysisParameters()
    analysis_params.dt = sim_params.dt
    analysis_params.measurement_time = sim_params.measurement_time
    analysis_params.n_chops = 20
    analysis_params.fit_power_spectrum = True
    analysis_params.fit_bispectrum = True
    analysis_params.fit_k_sum_free = True

    estimator = ParameterEstimator(intensity, analysis_params)
    results = estimator.estimate_parameters()
    print("   ✓ Analysis completed")
    print()

    # Step 3: Print results
    print("3. Estimation Results:")
    print("   Power Spectrum Fit:")
    print(
        f"      k_sum = {results.k_sum_fit:.3f} Hz (true: {sim_params.k_on + sim_params.k_off})"
    )
    print(f"      s2 = {results.s2_fit:.3f}")
    print(f"      pk_bg = {results.pk_bg_fit:.3f}")
    print("   Bispectrum Fit:")
    print(f"      C3 = {results.c3_fit:.3f}")
    print(f"      C3_offset = {results.c3_offset_fit:.3f}")
    print("   Derived Parameters:")
    print(f"      N = {results.n_emitters_fit:.2f} (true: {sim_params.n_emitters})")
    print(f"      k_on = {results.k_on_fit:.3f} Hz (true: {sim_params.k_on})")
    print(f"      k_off = {results.k_off_fit:.3f} Hz (true: {sim_params.k_off})")
    print(
        f"      I_single = {results.single_molecule_intensity_fit:.2f} (true: {sim_params.single_molecule_intensity})"
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
    # MATLAB uses FULL bispectrum for visualization (func_calc_BS_full), not triangular!
    if results.bispectrum is not None:
        ax5 = plt.subplot(2, 3, 5)

        # Calculate FULL bispectrum for visualization (matching MATLAB)
        chopped_traces = PowerSpectrumAnalyzer(
            analysis_params.dt, analysis_params.n_chops
        ).chop_trace(intensity)

        bs_analyzer = BispectrumAnalyzer(dt=analysis_params.dt)

        # w_max = 6 rad/s in MATLAB (for frequency cutoff)
        freq_max_rad = 6.0  # Match MATLAB w_max
        bs_full, freq_1_full, freq_2_full = bs_analyzer.calculate_full_bispectrum(
            chopped_traces, freq_max=freq_max_rad
        )

        # Block for visualization (num_blocks_goal=75 to match MATLAB)
        blocked_bs, blocked_f1, blocked_f2 = block_bispectrum(
            bs_full, freq_1_full, freq_2_full, num_blocks_goal=75
        )

        # Calculate theoretical full bispectrum for comparison
        if results.k_sum_fit is not None and results.c3_fit is not None:
            chop_length = chopped_traces.shape[0]

            # Convert freq matrices to k-indices for theoretical calculation
            k1_indices = freq_1_full * chop_length * analysis_params.dt
            k2_indices = freq_2_full * chop_length * analysis_params.dt

            # CRITICAL: Multiply by N_fit to match MATLAB (BS scales with N emitters)
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

            # Block theoretical bispectrum
            blocked_bs_theo, _, _ = block_bispectrum(
                theoretical_bs_full, freq_1_full, freq_2_full, num_blocks_goal=75
            )

            # Create half-half plot like MATLAB using LOG SCALE
            # MATLAB: triu(flipud(real(log(B_exp))), 1) + triu(flipud(real(log(B_theo))))'
            bs_exp_log = np.log(np.abs(blocked_bs) + 1e-10)
            bs_theo_log = np.log(np.abs(blocked_bs_theo) + 1e-10)

            # Flip upside down (MATLAB uses flipud)
            bs_exp_flipped = np.flipud(bs_exp_log)
            bs_theo_flipped = np.flipud(bs_theo_log)

            # Create the half-half: upper triangle from exp, lower from theory
            # MATLAB: BS_halfhalf1 = triu(flipud(log(B_raw)), 1)
            #         BS_halfhalf2 = triu(flipud(log(B_theo)))
            #         BS_halfhalf = BS_halfhalf1 + BS_halfhalf2'
            combined_bs = np.triu(bs_exp_flipped, k=1) + np.triu(bs_theo_flipped, k=0).T

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
