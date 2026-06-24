# %% [markdown]
"""
# Parameter Estimation Example - Simulated Data

This notebook demonstrates comprehensive analysis and visualization of fluorescence traces
to estimate molecular parameters from simulated data.

"""

# %% tags=["colab"]
# Install bbqpaint package (for Google Colab)
# !pip install git+https://github.com/TeunHuijben/bbqPAINT.git

# %% [markdown]
# ### Load dependencies

# %%
import matplotlib.pyplot as plt

from bbqpaint.analysis import AnalysisParameters, ParameterEstimator
from bbqpaint.simulation import SimulationParameters, TraceGenerator
from bbqpaint.utils import Plotter

# %% [markdown]
# ### Simulate the fluorescence intensity trace

# %%
# Set up simulation parameters
sim_params = SimulationParameters()
sim_params.k_on = 0.15  # ON rate (Hz)
sim_params.k_off = 0.3  # OFF rate (Hz)
sim_params.n_emitters = 4  # Number of molecules
sim_params.dt = 0.2  # Sampling time (seconds)
sim_params.measurement_time = 3600  # Total time (seconds)
sim_params.single_molecule_intensity = 1.0  # Brightness per molecule
sim_params.sigma_noise = 0.2  # Background noise

# Print simulation parameters
sim_params.summary()

# Generate the trace
generator = TraceGenerator(sim_params)
time_points, intensity = generator.generate_trace()

print(f"✓ Generated {len(intensity)} data points")
print(f"✓ Average intensity: {intensity.mean():.2f}")

# %% [markdown]
# ### Perform bbqPAINT analysis

# %%
# Configure analysis parameters
analysis_params = AnalysisParameters()
analysis_params.dt = sim_params.dt
analysis_params.measurement_time = sim_params.measurement_time
analysis_params.n_chops = 20  # Number of segments for bispectrum
analysis_params.fit_power_spectrum = True  # Fit power spectrum
analysis_params.fit_bispectrum = True  # Fit bispectrum

# Run the analysis
estimator = ParameterEstimator(intensity, analysis_params)
results = estimator.estimate_parameters()

print("✓ Analysis completed\n")

# Print fitted parameters with comparison to true values
results.summary(true_params=sim_params)

# %% [markdown]
# ### Visualize results

# %%
# Generate 6-panel figure showing all analysis results
plotter = Plotter()
plotter.plot_comprehensive_analysis(
    intensity, time_points, results, analysis_params, sim_params
)
plt.show()
