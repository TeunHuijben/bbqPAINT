"""
Simulation module for generating fluorescence traces.

This module contains classes for generating synthetic fluorescence traces
with various blinking models and noise characteristics.
"""

from mcoast.simulation.noise_models import GaussianNoise, NoiseModel, OutlierModel
from mcoast.simulation.parameters import (
    DynamicDisorderParameters,
    SimulationParameters,
    TwoExcitedStatesParameters,
)
from mcoast.simulation.state_machine import BlinkingStateMachine
from mcoast.simulation.trace_generator import TraceGenerator

__all__ = [
    "SimulationParameters",
    "TwoExcitedStatesParameters",
    "DynamicDisorderParameters",
    "TraceGenerator",
    "BlinkingStateMachine",
    "NoiseModel",
    "GaussianNoise",
    "OutlierModel",
]
