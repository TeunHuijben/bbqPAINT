"""
Simulation module for generating fluorescence traces.

This module contains classes for generating synthetic fluorescence traces
with blinking behavior and noise characteristics.
"""

from bbqpaint.simulation.noise_models import add_gaussian_noise
from bbqpaint.simulation.parameters import SimulationParameters
from bbqpaint.simulation.trace_generator import TraceGenerator

__all__ = [
    "SimulationParameters",
    "TraceGenerator",
    "add_gaussian_noise",
]
