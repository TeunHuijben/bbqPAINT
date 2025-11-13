# mCOAST

[![License](https://img.shields.io/github/license/TeunHuijben/mcoast.svg?color=green)](https://github.com/TeunHuijben/mcoast/raw/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![CI](https://github.com/TeunHuijben/mcoast/actions/workflows/ci.yml/badge.svg)](https://github.com/TeunHuijben/mcoast/actions/workflows/ci.yml)

**M**olecular **CO**unting from **A** **S**ingle intensity **T**race

A Python package for counting the number of blinking molecules from fluorescence intensity traces using power spectrum and bispectrum analysis. mCOAST extracts molecular parameters (k_on, k_off, number of emitters) from fluorescence intensity traces.

## Installation

#### From Source

```bash
conda create -n mcoast python=3.11
conda activate mcoast
git clone https://github.com/TeunHuijben/mcoast.git
cd mcoast
pip install -e ".[dev]"  # Install with development dependencies
```

#### From PyPI (Coming Soon)

```bash
pip install mcoast
```

## Quick Start

#### Basic Simulation

```python
from mcoast.simulation import SimulationParameters, TraceGenerator

# Create simulation parameters
sim_params = SimulationParameters()
sim_params.k_on = 0.15                      # On rate (Hz)
sim_params.k_off = 0.3                      # Off rate (Hz)
sim_params.n_emitters = 4                   # Number of emitters
sim_params.dt = 0.2                         # Sampling time (s)
sim_params.measurement_time = 3600          # Total measurement time (s)
sim_params.single_molecule_intensity = 1.0  # Single molecule brightness
sim_params.sigma_noise = 0.2                # Noise standard deviation

# Generate trace
generator = TraceGenerator(sim_params)
time_points, intensity = generator.generate_trace()
```

#### Parameter Estimation

```python
from mcoast.analysis import AnalysisParameters, ParameterEstimator

# Create analysis parameters
analysis_params = AnalysisParameters()
analysis_params.dt = 0.2
analysis_params.measurement_time = 3600
analysis_params.n_chops = 20                # Number of trace segments for bispectrum
analysis_params.fit_power_spectrum = True   # Fit power spectrum (k_sum, s2, pk_bg)
analysis_params.fit_bispectrum = True       # Fit bispectrum (C3)
analysis_params.fit_k_sum_free = True       # Refine k_sum in bispectrum fit

# Analyze the trace
estimator = ParameterEstimator(intensity, analysis_params)
results = estimator.estimate_parameters()

# Access results
print(f"Number of emitters: {results.n_emitters_fit:.2f}")
print(f"k_on: {results.k_on_fit:.3f} Hz")
print(f"k_off: {results.k_off_fit:.3f} Hz")
print(f"Single molecule intensity: {results.single_molecule_intensity_fit:.2f}")
```

## Examples

Three example notebooks demonstrating different use cases are available in `src/mcoast/examples/`. Either run the notebook in Google Colab by using the "Open in Colab" button, or run them locally.

- **`1_basic_simulation.ipynb`**: Generate synthetic fluorescence traces <a href="https://colab.research.google.com/github/TeunHuijben/mCOAST/blob/main/src/mcoast/examples/colab/1_basic_simulation.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- **`2_simulated_data.ipynb`**: Complete mCOAST pipeline simulated data <a href="https://colab.research.google.com/github/TeunHuijben/mCOAST/blob/main/src/mcoast/examples/colab/2_simulated_data.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- **`3_experimental_data.ipynb`**: Complete mCOAST pipeline on experimental data <a href="https://colab.research.google.com/github/TeunHuijben/mCOAST/blob/main/src/mcoast/examples/colab/3_experimental_data.ipynb" target="_blank">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**For developers**: Source files are in `src/mcoast/examples/raw/` (jupytext format). Edit those files, and a GitHub Action will automatically generate the `.ipynb` notebooks.

## What mCOAST Extracts

From a single fluorescence intensity trace, mCOAST estimates:

1. **k_on**: Fluorophore on-rate (binding/activation rate)
2. **k_off**: Fluorophore off-rate (unbinding/deactivation rate)
3. **N**: Number of independently blinking emitters
4. **I_single**: Single-molecule brightness

#### How It Works

1. **Power Spectrum Analysis** (2nd order):
   - Extracts k_sum = k_on + k_off
   - Extracts variance (s2) and background noise (pk_bg)
   - Uses maximum likelihood estimation (MLE)

2. **Bispectrum Analysis** (3rd order):
   - Extracts third cumulant (C3)
   - Uses weighted least squares (WLS) with full theoretical model

3. **Molecular Parameters**:
   - Combines power spectrum and bispectrum results
   - Solves analytical equations to extract all four parameters
   - Provides uncertainty estimates from covariance matrices


## Testing and Development

```bash
# Install development dependencies
pip install -e ".[dev]"  # Install with development dependencies

# Run all tests
pytest

# Run with coverage
pytest --cov=src/mcoast --cov-report=html

# Run specific test file
pytest tests/test_parameter_estimation.py

# Run pre-commit hooks
pre-commit run --all-files
```

## Package Structure

```
mcoast/
├── simulation/              # Trace generation and simulation
│   ├── parameters.py       # SimulationParameters dataclass
│   ├── trace_generator.py  # TraceGenerator for blinking dynamics
│   └── noise_models.py     # Gaussian noise injection
├── analysis/                # Parameter estimation and analysis
│   ├── parameters.py       # AnalysisParameters, AnalysisResults dataclasses
│   ├── power_spectrum.py   # PowerSpectrumAnalyzer with MLE fitting
│   ├── bispectrum.py       # BispectrumAnalyzer with WLS fitting
│   └── parameter_estimation.py  # ParameterEstimator (main orchestrator)
├── utils/                   # Utility functions
│   ├── statistics.py       # Blocking analysis for visualization
│   ├── preprocessing.py    # TracePreprocessor (detrending, smoothing, outlier removal)
│   └── visualization.py    # Plotter for traces and spectra
└── examples/                # Example scripts
    ├── 1_basic_simulation.ipynb
    ├── 2_simulated_data.ipynb
    └── 3_experimental_data.ipynb
```

## Citation

If you use mCOAST in your research, please cite:

```bibtex
...
```
