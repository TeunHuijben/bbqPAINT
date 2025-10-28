"""
Basic simulation example for mCOAST.

This script demonstrates how to generate synthetic fluorescence traces
using the mCOAST simulation module.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from mcoast.simulation import SimulationParameters, TraceGenerator  # noqa: E402
from mcoast.utils import Plotter  # noqa: E402


def main():
    """Main function demonstrating basic simulation."""

    print("=== mCOAST Basic Simulation Example ===")

    # Create simulation parameters
    sim_params = SimulationParameters()
    sim_params.k_on = 1.0  # On rate (Hz)
    sim_params.k_off = 2.0  # Off rate (Hz)
    sim_params.n_emitters = 4  # Number of emitters
    sim_params.dt = 0.1  # Sampling time (s)
    sim_params.measurement_time = 1000  # Total measurement time (s)
    sim_params.single_molecule_intensity = 40.0  # Single molecule intensity
    sim_params.snr = 20.0  # Signal-to-noise ratio

    # Validate parameters
    sim_params.validate()
    print("✓ Simulation parameters validated")

    # Create trace generator
    generator = TraceGenerator(sim_params)
    print("✓ Trace generator created")

    # Generate trace
    print("Generating fluorescence trace...")
    time_points, intensity = generator.generate_trace()
    print(f"✓ Generated trace with {len(intensity)} points")
    print(f"  Duration: {time_points[-1]:.1f} s")
    print(f"  Mean intensity: {np.mean(intensity):.2f}")
    print(f"  Std intensity: {np.std(intensity):.2f}")

    # Create plotter and visualize
    plotter = Plotter()

    # Plot the trace
    fig = plotter.plot_trace(
        intensity, sim_params.dt, title="Generated Fluorescence Trace"
    )

    # Save the plot
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "simulated_trace.png"), dpi=300, bbox_inches="tight"
    )
    print(f"✓ Plot saved to {output_dir}/simulated_trace.png")

    # Print parameter summary
    print("\n=== Simulation Parameters ===")
    print(f"On rate (k_on): {sim_params.k_on} Hz")
    print(f"Off rate (k_off): {sim_params.k_off} Hz")
    print(f"Number of emitters: {sim_params.n_emitters}")
    print(f"Sampling time: {sim_params.dt} s")
    print(f"Measurement time: {sim_params.measurement_time} s")
    print(f"Single molecule intensity: {sim_params.single_molecule_intensity}")
    print(f"Signal-to-noise ratio: {sim_params.snr}")

    # Calculate theoretical values
    k_sum_theory = sim_params.k_on + sim_params.k_off
    p_up_theory = sim_params.k_on / k_sum_theory
    i_mean_theory = (
        sim_params.n_emitters * sim_params.single_molecule_intensity * p_up_theory
    )

    print("\n=== Theoretical Values ===")
    print(f"Total rate (k_sum): {k_sum_theory} Hz")
    print(f"On probability: {p_up_theory:.3f}")
    print(f"Theoretical mean intensity: {i_mean_theory:.2f}")
    print(f"Actual mean intensity: {np.mean(intensity):.2f}")

    plt.show()


if __name__ == "__main__":
    main()
