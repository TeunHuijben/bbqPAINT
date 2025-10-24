from mcoast.analysis.parameter_estimation import ParameterEstimator
from mcoast.analysis.parameters import AnalysisParameters, AnalysisResults
from mcoast.simulation.parameters import SimulationParameters
from mcoast.simulation.trace_generator import TraceGenerator

"""
mCOAST: Multi-emitter Correlation Analysis of Single-molecule Time traces

A Python implementation for analyzing single-molecule fluorescence blinking dynamics
using power spectrum and bispectrum analysis.
"""

"""Molecular counting from a single iintensity trace"""

__version__ = "0.0.1"
__author__ = "Teun Huijben"
__email__ = "teunhuijben@hotmail.com"


__all__ = [
    "TraceGenerator",
    "ParameterEstimator",
    "SimulationParameters",
    "AnalysisParameters",
    "AnalysisResults",
]
