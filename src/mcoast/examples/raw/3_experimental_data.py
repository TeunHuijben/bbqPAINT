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

# %% Imports
import os

import matplotlib.pyplot as plt
import numpy as np

from mcoast.analysis import AnalysisParameters, ParameterEstimator
from mcoast.utils import Plotter

# %% Load experimental fluorescence trace
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

# %% Perform spectral analysis
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

print("✓ Analysis completed")

# Print fitted parameters
print("\n=== Estimation Results ===")
print("Power Spectrum Fit:")
print(f"  k_sum = {results.k_sum_fit:.3f} Hz")
print(f"  s2 = {results.s2_fit:.3f}")
print(f"  pk_bg = {results.pk_bg_fit:.3f}")
print("\nBispectrum Fit:")
print(f"  C3 = {results.c3_fit:.3f}")
print(f"  C3_offset = {results.c3_offset_fit:.3f}")
print("\nDerived Parameters:")
print(f"  N = {results.n_emitters_fit:.2f}")
print(f"  k_on = {results.k_on_fit:.3f} Hz")
print(f"  k_off = {results.k_off_fit:.3f} Hz")
print(f"  I_single = {results.single_molecule_intensity_fit:.2f}")

# %% Create comprehensive visualization
# Generate 6-panel figure showing all analysis results
plotter = Plotter()
plotter.plot_comprehensive_analysis(intensity, time_points, results, analysis_params)
plt.show()
