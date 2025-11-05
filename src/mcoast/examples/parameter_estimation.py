"""
Parameter estimation example for mCOAST.

This script demonstrates how to analyze fluorescence traces and estimate
molecular parameters using the mCOAST analysis module.
"""

import matplotlib.pyplot as plt
import numpy as np

from mcoast.analysis import AnalysisParameters, ParameterEstimator
from mcoast.simulation import SimulationParameters, TraceGenerator
from mcoast.utils import Plotter


def main():
    """Main function demonstrating parameter estimation."""

    print("=== mCOAST Parameter Estimation Example ===")
    print(
        "This example shows how to analyze fluorescence data to estimate molecular properties."
    )
    print()

    # Step 1: Generate some test data (normally you'd load your own data)
    print("1. Creating test data...")
    sim_params = SimulationParameters()
    sim_params.k_on = 1.5  # True ON rate
    sim_params.k_off = 2.5  # True OFF rate
    sim_params.n_emitters = 3  # True number of molecules
    sim_params.dt = 0.1
    sim_params.measurement_time = 2000
    sim_params.single_molecule_intensity = 50.0
    sim_params.sigma_noise = 10.0

    generator = TraceGenerator(sim_params)
    time_points, intensity = generator.generate_trace()

    print(f"   • Generated {len(intensity)} data points")
    print(
        f"   • We know the true values: {sim_params.n_emitters} molecules, rates {sim_params.k_on}/{sim_params.k_off}"
    )
    print()

    # Step 2: Analyze the data (pretend we don't know the true values)
    print("2. Analyzing the data to estimate parameters...")
    analysis_params = AnalysisParameters()
    analysis_params.dt = sim_params.dt
    analysis_params.measurement_time = sim_params.measurement_time
    analysis_params.n_chops = 5
    analysis_params.fit_power_spectrum = True
    analysis_params.fit_bispectrum = True

    # Let the analysis estimate parameters from the data
    estimator = ParameterEstimator(intensity, analysis_params)
    results = estimator.estimate_parameters()

    print("   ✓ Analysis completed!")
    print()

    # Step 3: Show what we found
    print("3. Results - What did we find?")
    print(f"   • Estimated number of molecules: {results.n_emitters_fit:.1f}")
    print(f"   • Estimated ON rate: {results.k_on_fit:.2f} per second")
    print(f"   • Estimated OFF rate: {results.k_off_fit:.2f} per second")
    print(
        f"   • Estimated single molecule brightness: {results.single_molecule_intensity_fit:.1f}"
    )
    print()

    # Step 4: Compare with what we actually put in
    print("4. How accurate were our estimates?")
    n_error = (
        abs(results.n_emitters_fit - sim_params.n_emitters)
        / sim_params.n_emitters
        * 100
    )
    k_on_error = abs(results.k_on_fit - sim_params.k_on) / sim_params.k_on * 100
    k_off_error = abs(results.k_off_fit - sim_params.k_off) / sim_params.k_off * 100

    print(
        f"   • Number of molecules - True: {sim_params.n_emitters}, "
        f"Found: {results.n_emitters_fit:.1f} (Error: {n_error:.1f}%)"
    )
    print(
        f"   • ON rate - True: {sim_params.k_on}, Found: {results.k_on_fit:.2f} (Error: {k_on_error:.1f}%)"
    )
    print(
        f"   • OFF rate - True: {sim_params.k_off}, Found: {results.k_off_fit:.2f} (Error: {k_off_error:.1f}%)"
    )
    print()

    # Step 5: Show the plots
    print("5. Creating plots...")
    plotter = Plotter()

    # Plot the original data
    plotter.plot_trace(intensity, sim_params.dt, title="Fluorescence Trace We Analyzed")

    # Plot analysis results if available
    if results.power_spectrum is not None:
        freq_vec = np.arange(1, len(results.power_spectrum) + 1) / (
            len(intensity) * sim_params.dt
        )
        plotter.plot_power_spectrum(
            freq_vec, results.power_spectrum, title="Power Spectrum Analysis"
        )

    if results.bispectrum is not None:
        plotter.plot_bispectrum(results.bispectrum, title="Bispectrum Analysis")

    print("   ✓ Plots created - these show the raw data and analysis results")
    print("   ✓ The analysis worked by looking at patterns in the blinking!")

    plt.show()


if __name__ == "__main__":
    main()
