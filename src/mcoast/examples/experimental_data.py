"""
Example: Analyzing Real Experimental Data

This example demonstrates how to analyze actual experimental fluorescence trace data
using the mCOAST analysis pipeline. The data is loaded from a CSV file containing
time-stamped intensity measurements.
"""

import os

import numpy as np

from mcoast.analysis import AnalysisParameters, ParameterEstimator
from mcoast.utils import Plotter


def load_experimental_data(filename="experimental_data_example.csv"):
    """Load experimental trace data from CSV file."""
    # Get the path to the data directory relative to this file
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "..", "data")
    filepath = os.path.join(data_dir, filename)

    print(f"Loading experimental data from {filename}...")

    # Load data (skip header if present)
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)
        time_points = data[:, 0]
        intensity = data[:, 1]
    except (ValueError, IndexError):
        # Fallback if no header
        data = np.loadtxt(filepath, delimiter=",")
        time_points = data[:, 0]
        intensity = data[:, 1]

    print(f"✓ Loaded {len(intensity)} data points")
    print(f"✓ Duration: {time_points[-1] - time_points[0]:.0f} seconds")
    print(f"✓ Average intensity: {np.mean(intensity):.1f}")

    return {"time": time_points, "intensity": intensity}


def analyze_experimental_data():
    """Complete analysis pipeline for experimental data."""

    # Step 1: Load experimental data
    trace_data = load_experimental_data()

    # Calculate sampling time from data
    dt = trace_data["time"][1] - trace_data["time"][0]
    print(f"✓ Sampling time: {dt:.3f} seconds")

    # Step 3: Set up analysis parameters
    print("\nSetting up analysis parameters...")
    analysis_params = AnalysisParameters()
    analysis_params.dt = dt
    analysis_params.measurement_time = trace_data["time"][-1] - trace_data["time"][0]
    analysis_params.n_chops = 5
    analysis_params.fit_power_spectrum = True
    analysis_params.fit_bispectrum = True

    # Step 4: Run complete analysis
    print("\nPerforming complete analysis...")
    estimator = ParameterEstimator(trace_data["intensity"], analysis_params)
    results = estimator.estimate_parameters()

    print("✓ Analysis completed!")

    # Step 5: Display results
    print("\nEstimated parameters:")
    if results.n_emitters_fit is not None:
        print(f"  • Number of molecules: {results.n_emitters_fit:.1f}")
    if results.k_on_fit is not None:
        print(f"  • ON rate: {results.k_on_fit:.2f} per second")
    if results.k_off_fit is not None:
        print(f"  • OFF rate: {results.k_off_fit:.2f} per second")
    if results.single_molecule_intensity_fit is not None:
        print(
            f"  • Single molecule brightness: {results.single_molecule_intensity_fit:.1f}"
        )

    # Step 6: Generate summary plots
    print("\nGenerating analysis plots...")
    plotter = Plotter()

    print("✓ Trace plot")
    plotter.plot_trace(trace_data["intensity"], dt)

    if results.power_spectrum is not None:
        print("✓ Power spectrum plot")
        freq_vec = np.arange(1, len(results.power_spectrum) + 1) / (
            len(trace_data["intensity"]) * analysis_params.dt
        )
        plotter.plot_power_spectrum(freq_vec, results.power_spectrum)

    if results.bispectrum is not None:
        print("✓ Bispectrum plot")
        plotter.plot_bispectrum(results.bispectrum)

    print("\nAnalysis complete!")
    return results


if __name__ == "__main__":
    print("=== mCOAST Experimental Data Analysis ===")
    analyze_experimental_data()
