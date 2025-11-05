"""
Simulation module for generating fluorescence traces.

This module contains classes for generating synthetic fluorescence traces
with blinking behavior and noise characteristics.
"""

from mcoast.simulation.noise_models import add_gaussian_noise
from mcoast.simulation.parameters import SimulationParameters
from mcoast.simulation.trace_generator import TraceGenerator

__all__ = [
    "SimulationParameters",
    "TraceGenerator",
    "add_gaussian_noise",
]
