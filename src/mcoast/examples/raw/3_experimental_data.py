# %% [markdown]
"""
# Parameter Estimation Example - Experimental Data

This notebook demonstrates comprehensive analysis and visualization of fluorescence traces
to estimate molecular parameters from experimental data.

We'll:
1. Load experimental fluorescence trace from a CSV file
2. Analyze it using power spectrum and bispectrum methods
3. Extract molecular parameters (k_on, k_off, N, I_single)
4. Visualize the results in a comprehensive 6-panel figure
"""

# %% tags=["colab"]
# Install mcoast package (for Google Colab)
# !pip install git+https://github.com/TeunHuijben/mcoast.git

# %% [markdown]
# ## Load dependencies

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from mcoast.analysis import AnalysisParameters, ParameterEstimator
from mcoast.utils import Plotter

# %% [markdown]
# ## Load experimental fluorescence trace

# %%
# Load data from CSV file (time, intensity columns)
current_dir = os.path.dirname(__file__)
data_dir = os.path.join(current_dir, "..", "data")
filepath = os.path.join(data_dir, "experimental_data_example.csv")

# Load the data
data = np.loadtxt(filepath, delimiter=",")
time_points = data[:, 0]
intensity = data[:, 1]

print(f"✓ Loaded {len(intensity)} data points")
print(f"✓ Duration: {time_points[-1] - time_points[0]:.0f} seconds")
print(f"✓ Sampling time (dt): {time_points[1] - time_points[0]:.3f} seconds")
print(f"✓ Average intensity: {intensity.mean():.2f}")

# %% [markdown]
# ## Perform mCOAST analysis

# %%
# Configure analysis parameters
analysis_params = AnalysisParameters()
analysis_params.dt = time_points[1] - time_points[0]  # Calculate from data
analysis_params.measurement_time = time_points[-1] - time_points[0]
analysis_params.n_chops = 20  # Number of segments for bispectrum
analysis_params.fit_power_spectrum = True  # Fit power spectrum
analysis_params.fit_bispectrum = True  # Fit bispectrum
analysis_params.fit_k_sum_free = True  # Refine k_sum in bispectrum fit

# Run the analysis
estimator = ParameterEstimator(intensity, analysis_params)
results = estimator.estimate_parameters()

print("✓ Analysis completed\n")

# Print fitted parameters
results.summary()

# %% [markdown]
# ## Visualize results

# %%
# Generate 6-panel figure showing all analysis results
plotter = Plotter()
plotter.plot_comprehensive_analysis(intensity, time_points, results, analysis_params)
plt.show()
