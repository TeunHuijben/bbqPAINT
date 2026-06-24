# %% [markdown]
"""
# Basic Simulation Example for bbqPAINT

This notebook demonstrates how to generate synthetic fluorescence traces
using the bbqPAINT simulation module.

We'll simulate fluorescence from blinking molecules and visualize the results.
"""

# %% tags=["colab"]
# Install bbqpaint package (for Google Colab)
# !pip install git+https://github.com/TeunHuijben/bbqPAINT.git

# %% [markdown]
# ### Load dependencies

# %%
import matplotlib.pyplot as plt
import numpy as np

from bbqpaint.simulation import SimulationParameters, TraceGenerator
from bbqpaint.utils import Plotter

# %% [markdown]
# ### Set up simulation parameters

# %%
# Configure the properties of our simulated molecules
sim_params = SimulationParameters()
sim_params.k_on = 0.15  # How fast molecules turn ON (per second)
sim_params.k_off = 0.3  # How fast molecules turn OFF (per second)
sim_params.n_emitters = 4  # Number of molecules
sim_params.dt = 0.2  # Time between measurements (seconds)
sim_params.measurement_time = 5000  # Total measurement time (seconds)
sim_params.single_molecule_intensity = 1.0  # Brightness of one molecule
sim_params.sigma_noise = 0.2  # Amount of background noise

# Print simulation parameters
sim_params.summary()

# %% [markdown]
# ### Simulate the fluorescence intensity trace

# %%
# Create a trace generator and simulate the blinking
generator = TraceGenerator(sim_params)
time_points, intensity = generator.generate_trace()

print(f"✓ Generated {len(intensity)} data points")
print(f"✓ Average intensity: {np.mean(intensity):.1f}")
print(f"✓ Intensity variation (std): {np.std(intensity):.1f}")

# Compare with theoretical expectation
k_sum_theory = sim_params.k_on + sim_params.k_off
p_up_theory = sim_params.k_on / k_sum_theory  # Fraction of time molecules are ON
expected_intensity = (
    sim_params.n_emitters * sim_params.single_molecule_intensity * p_up_theory
)
print(f"\nTheoretical average intensity: {expected_intensity:.1f}")
print(f"Difference from theory: {abs(np.mean(intensity) - expected_intensity):.1f}")

# %% [markdown]
# ### Plot the trace

# %%
# Visualize the simulated trace to see molecules blinking on and off
plotter = Plotter()
plotter.plot_trace(intensity, sim_params.dt, title="Simulated Fluorescence Trace")
# plt.xlim(0,50) #uncomment to see zoom-in of trace
plt.show()
