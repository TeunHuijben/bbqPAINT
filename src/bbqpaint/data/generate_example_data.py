"""
Generate example experimental data for bbqPAINT demo.

This script creates a realistic fluorescence trace and saves it as a CSV file
for use in the experimental_data.py example.
"""

import numpy as np

from bbqpaint.simulation import SimulationParameters, TraceGenerator


def main():
    """Generate and save example experimental data."""

    print("Generating example experimental data...")

    # Set up realistic experimental parameters
    sim_params = SimulationParameters()
    sim_params.k_on = 0.15
    sim_params.k_off = 0.3
    sim_params.n_emitters = 4
    sim_params.dt = 0.2  # 100 ms sampling
    sim_params.measurement_time = 5000  # 50 minutes
    sim_params.single_molecule_intensity = 10.0
    sim_params.sigma_noise = 1.0  # Realistic noise level

    # Generate the trace
    np.random.seed(42)  # For reproducible example data
    generator = TraceGenerator(sim_params)
    time_points, intensity = generator.generate_trace()

    # Add some experimental artifacts
    # Add occasional outliers (as happens in real experiments)
    n_outliers = int(0.001 * len(intensity))
    outlier_indices = np.random.choice(len(intensity), size=n_outliers, replace=False)
    intensity[outlier_indices] += np.random.normal(0, 10, n_outliers)

    # Add slow drift (common in real experiments)
    drift = (
        0.5 * np.sin(2 * np.pi * time_points / 1000)
        + 0.3 * time_points / time_points[-1]
    )
    intensity += drift

    # Save as CSV file
    data = np.column_stack((time_points, intensity))
    np.savetxt(
        "experimental_data_example.csv",
        data,
        delimiter=",",
        comments="",
        fmt="%.6f",
    )

    print(f"✓ Generated {len(intensity)} data points")
    print(f"✓ Duration: {time_points[-1]:.0f} seconds")
    print(f"✓ Average intensity: {np.mean(intensity):.1f}")
    print("✓ Saved to experimental_data.csv")
    print()
    print("True parameters used:")
    print(f"  • Number of molecules: {sim_params.n_emitters}")
    print(f"  • ON rate: {sim_params.k_on} per second")
    print(f"  • OFF rate: {sim_params.k_off} per second")
    print(f"  • Single molecule brightness: {sim_params.single_molecule_intensity}")


if __name__ == "__main__":
    main()
