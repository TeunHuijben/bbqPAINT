"""
Basic simulation example for mCOAST.

This script demonstrates how to generate synthetic fluorescence traces
using the mCOAST simulation module.
"""

import matplotlib.pyplot as plt
import numpy as np

from mcoast.simulation import SimulationParameters, TraceGenerator
from mcoast.utils import Plotter


def main():
    """Main function demonstrating basic simulation."""

    print("=== mCOAST Basic Simulation Example ===")
    print(
        "This example shows how to simulate fluorescence traces from blinking molecules."
    )
    print()

    # Set up simulation parameters
    print("1. Setting up simulation parameters...")
    sim_params = SimulationParameters()
    sim_params.k_on = 0.15  # How fast molecules turn ON (per second)
    sim_params.k_off = 0.3  # How fast molecules turn OFF (per second)
    sim_params.n_emitters = 4  # Number of molecules
    sim_params.dt = 0.2  # Time between measurements (seconds)
    sim_params.measurement_time = 5000  # Total measurement time (seconds)
    sim_params.single_molecule_intensity = 1.0  # Brightness of one molecule
    sim_params.sigma_noise = 0.2  # Amount of background noise

    print(f"   • Number of molecules: {sim_params.n_emitters}")
    print(f"   • ON rate: {sim_params.k_on} per second")
    print(f"   • OFF rate: {sim_params.k_off} per second")
    print(f"   • Measurement time: {sim_params.measurement_time} seconds")
    print(f"   • Single molecule brightness: {sim_params.single_molecule_intensity}")
    print()

    # Generate the fluorescence trace
    print("2. Generating fluorescence trace...")
    generator = TraceGenerator(sim_params)
    time_points, intensity = generator.generate_trace()

    print(f"   ✓ Generated {len(intensity)} data points")
    print(f"   ✓ Average intensity: {np.mean(intensity):.1f}")
    print(f"   ✓ Intensity variation: {np.std(intensity):.1f}")
    print()

    # Show what we expect vs what we got
    print("3. Comparing results with theory...")
    k_sum_theory = sim_params.k_on + sim_params.k_off
    p_up_theory = sim_params.k_on / k_sum_theory  # Fraction of time molecules are ON
    expected_intensity = (
        sim_params.n_emitters * sim_params.single_molecule_intensity * p_up_theory
    )

    print(f"   • Expected average intensity: {expected_intensity:.1f}")
    print(f"   • Actual average intensity: {np.mean(intensity):.1f}")
    print(f"   • Difference: {abs(np.mean(intensity) - expected_intensity):.1f}")
    print()

    # Create and show the plot
    print("4. Creating plot...")
    plotter = Plotter()
    plotter.plot_trace(intensity, sim_params.dt, title="Simulated Fluorescence Trace")

    print("   ✓ Plot created - showing fluorescence over time")
    print("   ✓ You should see the molecules blinking on and off!")

    plt.show()


if __name__ == "__main__":
    main()
