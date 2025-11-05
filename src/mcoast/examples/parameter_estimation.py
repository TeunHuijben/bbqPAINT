"""
Parameter estimation example for mCOAST.

This script demonstrates how to analyze fluorescence traces and estimate
molecular parameters using the mCOAST analysis module.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path to import mcoast
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from mcoast.analysis import AnalysisParameters, ParameterEstimator  # noqa: E402
from mcoast.simulation import SimulationParameters, TraceGenerator  # noqa: E402
from mcoast.utils import Plotter  # noqa: E402


def main():
    """Main function demonstrating parameter estimation."""

    print("=== mCOAST Parameter Estimation Example ===")

    # First, generate a synthetic trace
    print("1. Generating synthetic trace...")

    sim_params = SimulationParameters()
    sim_params.k_on = 1.5
    sim_params.k_off = 2.5
    sim_params.n_emitters = 3
    sim_params.dt = 0.1
    sim_params.measurement_time = 2000
    sim_params.single_molecule_intensity = 50.0
    sim_params.sigma_noise = 10.0

    sim_params.validate()

    generator = TraceGenerator(sim_params)
    time_points, intensity = generator.generate_trace()

    print(f"✓ Generated trace with {len(intensity)} points")
    print(
        f"  True parameters: k_on={sim_params.k_on}, k_off={sim_params.k_off}, n_emitters={sim_params.n_emitters}"
    )

    # Now analyze the trace
    print("\n2. Analyzing trace...")

    analysis_params = AnalysisParameters()
    analysis_params.dt = sim_params.dt
    analysis_params.measurement_time = sim_params.measurement_time
    analysis_params.n_chops = 5
    analysis_params.fit_power_spectrum = True
    analysis_params.fit_bispectrum = True
    analysis_params.calculate_uncertainties = True

    # Set initial guesses based on data
    analysis_params.initial_guess.s2 = np.var(intensity)
    analysis_params.initial_guess.c3 = np.mean((intensity - np.mean(intensity)) ** 3)

    analysis_params.validate()

    # Create parameter estimator
    estimator = ParameterEstimator(intensity, analysis_params)

    # Perform analysis
    print("Performing parameter estimation...")
    results = estimator.estimate_parameters()

    print("✓ Analysis completed")

    # Print results
    print("\n=== Analysis Results ===")
    results.print_summary()

    # Compare with true values
    print("\n=== Comparison with True Values ===")
    print("Number of emitters:")
    print(f"  True: {sim_params.n_emitters}")
    print(f"  Estimated: {results.n_emitters_fit:.2f}")

    print("On rate (k_on):")
    print(f"  True: {sim_params.k_on}")
    print(f"  Estimated: {results.k_on_fit:.3f}")

    print("Off rate (k_off):")
    print(f"  True: {sim_params.k_off}")
    print(f"  Estimated: {results.k_off_fit:.3f}")

    print("Single molecule intensity:")
    print(f"  True: {sim_params.single_molecule_intensity}")
    print(f"  Estimated: {results.single_molecule_intensity_fit:.2f}")

    # Calculate errors
    n_emitters_error = (
        abs(results.n_emitters_fit - sim_params.n_emitters)
        / sim_params.n_emitters
        * 100
    )
    k_on_error = abs(results.k_on_fit - sim_params.k_on) / sim_params.k_on * 100
    k_off_error = abs(results.k_off_fit - sim_params.k_off) / sim_params.k_off * 100
    i_single_error = (
        abs(
            results.single_molecule_intensity_fit - sim_params.single_molecule_intensity
        )
        / sim_params.single_molecule_intensity
        * 100
    )

    print("\n=== Relative Errors ===")
    print(f"Number of emitters: {n_emitters_error:.1f}%")
    print(f"On rate: {k_on_error:.1f}%")
    print(f"Off rate: {k_off_error:.1f}%")
    print(f"Single molecule intensity: {i_single_error:.1f}%")

    # Create visualizations
    print("\n3. Creating visualizations...")

    plotter = Plotter()

    # Plot trace
    fig1 = plotter.plot_trace(
        intensity, sim_params.dt, title="Analyzed Fluorescence Trace"
    )

    # Plot power spectrum if available
    if results.power_spectrum is not None:
        freq_vec = np.arange(1, len(results.power_spectrum) + 1) / (
            len(intensity) * sim_params.dt
        )
        fig2 = plotter.plot_power_spectrum(
            freq_vec, results.power_spectrum, title="Power Spectrum Analysis"
        )

    # Plot bispectrum if available
    if results.bispectrum is not None:
        fig3 = plotter.plot_bispectrum(results.bispectrum, title="Bispectrum Analysis")

    # Save plots
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    fig1.savefig(
        os.path.join(output_dir, "analyzed_trace.png"), dpi=300, bbox_inches="tight"
    )
    if results.power_spectrum is not None:
        fig2.savefig(
            os.path.join(output_dir, "power_spectrum.png"), dpi=300, bbox_inches="tight"
        )
    if results.bispectrum is not None:
        fig3.savefig(
            os.path.join(output_dir, "bispectrum.png"), dpi=300, bbox_inches="tight"
        )

    print(f"✓ Plots saved to {output_dir}/")

    plt.show()


if __name__ == "__main__":
    main()
