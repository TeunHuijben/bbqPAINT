"""
Trace generation classes for mCOAST.

This module contains the main classes for generating fluorescence traces
with various blinking models and noise characteristics.
"""

from typing import Tuple

import numpy as np

from mcoast.simulation.noise_models import GaussianNoise, NoiseModel
from mcoast.simulation.parameters import SimulationParameters
from mcoast.simulation.state_machine import BlinkingStateMachine


class TraceGenerator:
    """Main class for generating fluorescence traces"""

    def __init__(self, params: SimulationParameters):
        """
        Initialize trace generator with simulation parameters.

        Args:
            params: SimulationParameters object containing all simulation settings
        """
        self.params = params
        self.params.validate()

        # Initialize state machine based on model type
        self.state_machine = BlinkingStateMachine(params.model_type)

        # Initialize noise models
        self.noise_model = self._setup_noise_model()

    def _setup_noise_model(self) -> NoiseModel:
        """Setup noise model based on parameters"""
        # For now, return a simple Gaussian noise model
        # This can be extended to composite noise models
        if (
            self.params.snr is not None
            and self.params.single_molecule_intensity is not None
        ):
            sigma = (1 / self.params.snr) * self.params.single_molecule_intensity
            return GaussianNoise(sigma)
        elif self.params.noise_sigma is not None:
            return GaussianNoise(self.params.noise_sigma)
        else:
            return GaussianNoise(0.0)  # No noise

    def generate_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete fluorescence trace.

        Returns:
            Tuple of (time_points, intensity_values)
        """
        # Calculate number of frames
        n_frames = int(self.params.measurement_time / self.params.dt)

        # Generate time vector
        time_points = np.arange(n_frames) * self.params.dt

        # Initialize intensity array
        intensity = np.zeros(n_frames)

        # Generate traces for each emitter
        for emitter_idx in range(self.params.n_emitters):
            emitter_trace = self._generate_single_emitter_trace(n_frames)
            intensity += emitter_trace

        # Add noise
        intensity = self.noise_model.add_noise(intensity)

        return time_points, intensity

    def _generate_single_emitter_trace(self, n_frames: int) -> np.ndarray:
        """Generate trace for a single emitter"""
        # This is a placeholder - will be implemented with state machine
        # For now, return a simple random trace
        return np.random.poisson(self.params.single_molecule_intensity, n_frames)

    def generate_ensemble_trace(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trace from multiple emitters (alias for generate_trace)"""
        return self.generate_trace()


class SingleEmitterGenerator:
    """Generate traces for individual emitters"""

    def __init__(self, k_on: float, k_off: float, dt: float, measurement_time: float):
        """
        Initialize single emitter generator.

        Args:
            k_on: On transition rate
            k_off: Off transition rate
            dt: Sampling time
            measurement_time: Total measurement time
        """
        self.k_on = k_on
        self.k_off = k_off
        self.dt = dt
        self.measurement_time = measurement_time

        # Calculate derived parameters
        self.k_sum = k_on + k_off
        self.p_up = k_on / self.k_sum
        self.char_time = 1 / k_on + 1 / k_off

    def generate_state_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate on/off state sequence.

        Returns:
            Tuple of (time_points, state_values)
        """
        # Placeholder implementation
        n_frames = int(self.measurement_time / self.dt)
        time_points = np.arange(n_frames) * self.dt

        # Simple random state sequence for now
        states = np.random.binomial(1, self.p_up, n_frames)

        return time_points, states

    def apply_time_averaging(self, state_sequence: np.ndarray) -> np.ndarray:
        """
        Apply time averaging to discrete states.

        Args:
            state_sequence: Binary state sequence

        Returns:
            Time-averaged intensity values
        """
        # Placeholder implementation
        # In the real implementation, this would handle the integration
        # over the sampling time window
        return state_sequence.astype(float)
