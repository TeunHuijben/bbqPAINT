# %% [markdown]
"""
# Parameter Estimation Example - Experimental Data

This notebook demonstrates comprehensive analysis and visualization of fluorescence traces
to estimate molecular parameters from experimental data.

"""

# %% tags=["colab"]
# Install mcoast package (for Google Colab)
# !pip install git+https://github.com/TeunHuijben/mcoast.git

# %% [markdown]
# ### Load dependencies

# %%
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np

from mcoast.analysis import AnalysisParameters, ParameterEstimator
from mcoast.utils import Plotter

# %% [markdown]
# ### Load experimental fluorescence trace

# %%
# Load experimental trace from mCOAST package
data_path = resources.files("mcoast").joinpath("data", "experimental_data_example.csv")
data = np.loadtxt(data_path, delimiter=",")
time_points = data[:, 0]
intensity = data[:, 1]

print(f"✓ Loaded {len(intensity)} data points")
print(f"✓ Duration: {time_points[-1] - time_points[0]:.0f} seconds")
print(f"✓ Sampling time (dt): {time_points[1] - time_points[0]:.3f} seconds")
print(f"✓ Average intensity: {intensity.mean():.2f}")

# %% [markdown]
# ### Perform mCOAST analysis

# %%
# Configure analysis parameters
analysis_params = AnalysisParameters()
analysis_params.dt = time_points[1] - time_points[0]  # Calculate from data
analysis_params.measurement_time = time_points[-1] - time_points[0]
analysis_params.n_chops = 20  # Number of segments for bispectrum
analysis_params.fit_power_spectrum = True  # Fit power spectrum
analysis_params.fit_bispectrum = True  # Fit bispectrum

# Run the analysis
estimator = ParameterEstimator(intensity, analysis_params)
results = estimator.estimate_parameters()

print("✓ Analysis completed\n")

# Print fitted parameters
results.summary()

# %% [markdown]
# ### Visualize results

# %%
# Generate 6-panel figure showing all analysis results
plotter = Plotter()
plotter.plot_comprehensive_analysis(intensity, time_points, results, analysis_params)
plt.show()
