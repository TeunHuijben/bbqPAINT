# mCOAST

[![License](https://img.shields.io/pypi/l/mcoast.svg?color=green)](https://github.com/TeunHuijben/mcoast/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mcoast.svg?color=green)](https://pypi.org/project/mcoast)
[![Python Version](https://img.shields.io/pypi/pyversions/mcoast.svg?color=green)](https://python.org)
[![CI](https://github.com/TeunHuijben/mcoast/actions/workflows/ci.yml/badge.svg)](https://github.com/TeunHuijben/mcoast/actions/workflows/ci.yml)

Molecular counting from a single intensity trace

## Installation

```bash
git clone https://github.com/TeunHuijben/mcoast.git
cd mcoast
pip install -e .
```

## Quick Start

### Basic Simulation

```python
from mcoast.simulation import SimulationParameters, TraceGenerator

# Create simulation parameters
sim_params = SimulationParameters()
sim_params.k_on = 1.0          # On rate (Hz)
sim_params.k_off = 2.0         # Off rate (Hz)
sim_params.n_emitters = 4      # Number of emitters
sim_params.dt = 0.1           # Sampling time (s)
sim_params.measurement_time = 1000  # Total measurement time (s)
sim_params.single_molecule_intensity = 40.0
sim_params.sigma_noise = 2.0  # Noise standard deviation

# Generate trace
generator = TraceGenerator(sim_params)
time_points, intensity = generator.generate_trace()
```

### Parameter Estimation

```python
from mcoast.analysis import AnalysisParameters, ParameterEstimator

# Create analysis parameters
analysis_params = AnalysisParameters()
analysis_params.dt = 0.1
analysis_params.measurement_time = 1000
analysis_params.n_chops = 5

# Analyze the trace
estimator = ParameterEstimator(intensity, analysis_params)
results = estimator.estimate_parameters()

# Print results
results.print_summary()
```

## Examples

The package includes several example scripts demonstrating different use cases:

- `basic_simulation.py`: Generate synthetic fluorescence traces
- `parameter_estimation.py`: Analyze traces and estimate parameters
- `experimental_data.py`: Analyze experimental data

Run examples:

```bash
python mcoast/examples/basic_simulation.py
python mcoast/examples/parameter_estimation.py
python mcoast/examples/experimental_data.py
```


## Package Structure

```
mcoast/
├── simulation/          # Trace generation and simulation
│   ├── parameters.py   # Simulation parameter classes
│   ├── trace_generator.py
│   └── noise_models.py
├── analysis/            # Parameter estimation and analysis
│   ├── parameters.py   # Analysis parameter classes
│   ├── spectral_analysis.py
│   ├── bispectrum.py
│   ├── parameter_estimation.py
│   └── fitting.py
├── utils/               # Utility functions
│   ├── statistics.py
│   └── visualization.py
└── examples/            # Example scripts
```


## Citation

If you use mCOAST in your research, please cite:

```bibtex
...
```
