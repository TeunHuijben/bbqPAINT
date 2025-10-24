"""
Analysis module for mCOAST parameter estimation.

This module contains classes for analyzing fluorescence traces and estimating
molecular parameters using power spectrum and bispectrum analysis.
"""

from mcoast.analysis.bispectrum import BispectrumAnalyzer
from mcoast.analysis.fitting import FittingEngine
from mcoast.analysis.parameter_estimation import ParameterEstimator
from mcoast.analysis.parameters import (
    AnalysisParameters,
    AnalysisResults,
    FittingBounds,
    InitialGuessParameters,
)
from mcoast.analysis.spectral_analysis import SpectralAnalyzer

__all__ = [
    "AnalysisParameters",
    "AnalysisResults",
    "InitialGuessParameters",
    "FittingBounds",
    "SpectralAnalyzer",
    "BispectrumAnalyzer",
    "ParameterEstimator",
    "FittingEngine",
]
