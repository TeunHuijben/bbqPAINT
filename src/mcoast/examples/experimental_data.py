"""
Experimental data analysis example for mCOAST.

This script demonstrates how to analyze experimental fluorescence traces
using the mCOAST analysis module.
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
from mcoast.utils import Plotter, TracePreprocessor  # noqa: E402


def load_experimental_data(filename: str) -> tuple:
    """
    Load experimental data from file.

    This is a placeholder function - in practice, you would load
    your experimental data from the appropriate file format.

    Args:
        filename: Path to data file

    Returns:
        Tuple of (time_points, intensity_values)
    """
    # For demonstration, generate synthetic data that mimics experimental conditions
    print(f"Loading experimental data from {filename}...")

    # Simulate experimental conditions
    dt = 0.1  # 100 ms sampling
    measurement_time = 3000  # 5 minutes
    n_points = int(measurement_time / dt)

    # Generate realistic experimental trace
    np.random.seed(42)  # For reproducibility

    # Multiple emitters with different rates
    k_on_values = [0.8, 1.2, 1.5]
    k_off_values = [1.5, 2.0, 2.5]

    intensity = np.zeros(n_points)
    time_points = np.arange(n_points) * dt

    for i, (k_on, k_off) in enumerate(zip(k_on_values, k_off_values)):
        # Generate individual emitter trace
        emitter_intensity = generate_emitter_trace(k_on, k_off, dt, measurement_time)
        intensity += emitter_intensity

    # Add realistic experimental noise
    noise_level = 0.1 * np.mean(intensity)
    intensity += np.random.normal(0, noise_level, n_points)

    # Add occasional outliers
    outlier_indices = np.random.choice(
        n_points, size=int(0.001 * n_points), replace=False
    )
    intensity[outlier_indices] += np.random.normal(
        0, 2 * noise_level, len(outlier_indices)
    )

    print(f"✓ Loaded {n_points} data points")
    print(f"  Duration: {measurement_time} s")
    print(f"  Sampling rate: {1/dt} Hz")
    print(f"  Mean intensity: {np.mean(intensity):.2f}")
    print(f"  Std intensity: {np.std(intensity):.2f}")

    return time_points, intensity


def generate_emitter_trace(
    k_on: float, k_off: float, dt: float, measurement_time: float
) -> np.ndarray:
    """Generate trace for a single emitter (simplified version)"""
    n_points = int(measurement_time / dt)
    trace = np.zeros(n_points)

    # Simple two-state model
    current_state = np.random.binomial(1, k_on / (k_on + k_off))

    for i in range(n_points):
        trace[i] = current_state

        # Check for state transition
        if current_state == 0:
            # In off state, check for on transition
            if np.random.exponential(1 / k_on) < dt:
                current_state = 1
        else:
            # In on state, check for off transition
            if np.random.exponential(1 / k_off) < dt:
                current_state = 0

    return trace * 30  # Scale intensity


def main():
    """Main function demonstrating experimental data analysis."""

    print("=== mCOAST Experimental Data Analysis Example ===")

    # Load experimental data
    print("1. Loading experimental data...")
    time_points, intensity = load_experimental_data("experimental_data.txt")

    # Preprocess the data
    print("\n2. Preprocessing data...")
    preprocessor = TracePreprocessor()

    # Remove outliers
    cleaned_intensity, outlier_mask = preprocessor.remove_outliers(
        intensity, method="iqr"
    )
    n_outliers = np.sum(outlier_mask)
    print(f"✓ Removed {n_outliers} outliers ({n_outliers/len(intensity)*100:.1f}%)")

    # Interpolate NaN values
    cleaned_intensity = preprocessor.interpolate_nans(cleaned_intensity)
    print("✓ Interpolated missing values")

    # Detrend the data
    detrended_intensity = preprocessor.detrend(cleaned_intensity, method="linear")
    print("✓ Detrended data")

    # Analyze the preprocessed trace
    print("\n3. Analyzing preprocessed trace...")

    analysis_params = AnalysisParameters()
    analysis_params.dt = time_points[1] - time_points[0]  # Calculate dt from data
    analysis_params.measurement_time = time_points[-1]
    analysis_params.n_chops = 5
    analysis_params.fit_power_spectrum = True
    analysis_params.fit_bispectrum = True
    analysis_params.calculate_uncertainties = True

    # Set initial guesses based on preprocessed data
    analysis_params.initial_guess.s2 = np.var(detrended_intensity)
    analysis_params.initial_guess.c3 = np.mean(
        (detrended_intensity - np.mean(detrended_intensity)) ** 3
    )

    analysis_params.validate()

    # Create parameter estimator
    estimator = ParameterEstimator(detrended_intensity, analysis_params)

    # Perform analysis
    print("Performing parameter estimation...")
    results = estimator.estimate_parameters()

    print("✓ Analysis completed")

    # Print results
    print("\n=== Analysis Results ===")
    results.print_summary()

    # Create visualizations
    print("\n4. Creating visualizations...")

    plotter = Plotter()

    # Plot original vs preprocessed trace
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time_points, intensity, "b-", linewidth=0.5, alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.set_title("Original Experimental Trace")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_points, detrended_intensity, "g-", linewidth=0.5, alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.set_title("Preprocessed Trace")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Plot power spectrum if available
    if results.power_spectrum is not None:
        freq_vec = np.arange(1, len(results.power_spectrum) + 1) / (
            len(detrended_intensity) * analysis_params.dt
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
        os.path.join(output_dir, "experimental_trace.png"), dpi=300, bbox_inches="tight"
    )
    if results.power_spectrum is not None:
        fig2.savefig(
            os.path.join(output_dir, "experimental_power_spectrum.png"),
            dpi=300,
            bbox_inches="tight",
        )
    if results.bispectrum is not None:
        fig3.savefig(
            os.path.join(output_dir, "experimental_bispectrum.png"),
            dpi=300,
            bbox_inches="tight",
        )

    print(f"✓ Plots saved to {output_dir}/")

    # Print analysis summary
    print("\n=== Analysis Summary ===")
    print(f"Data points analyzed: {len(detrended_intensity)}")
    print(f"Measurement duration: {time_points[-1]:.1f} s")
    print(f"Sampling rate: {1/analysis_params.dt:.1f} Hz")
    print(f"Outliers removed: {n_outliers}")

    if results.n_emitters_fit is not None:
        print(f"Estimated number of emitters: {results.n_emitters_fit:.2f}")
    if results.k_on_fit is not None:
        print(f"Estimated on rate: {results.k_on_fit:.3f} Hz")
    if results.k_off_fit is not None:
        print(f"Estimated off rate: {results.k_off_fit:.3f} Hz")

    plt.show()


if __name__ == "__main__":
    main()
