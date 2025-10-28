"""
Trace generation classes for mCOAST.

This module contains the main classes for generating fluorescence traces
with blinking behavior and noise characteristics.
"""

from typing import Tuple

import numpy as np

from mcoast.simulation.noise_models import GaussianNoise
from mcoast.simulation.parameters import SimulationParameters


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

        # Initialize noise models
        self.noise_model = self._setup_noise_model()

    def _setup_noise_model(self) -> GaussianNoise:
        """Setup noise model based on parameters"""
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
        """
        Generate trace for a single emitter using simple two-state blinking.

        Args:
            n_frames: Number of frames to generate

        Returns:
            Intensity trace for a single emitter
        """
        # Generate blinking state sequence
        _, states = self._simple_two_state_blinking(
            self.params.k_on,
            self.params.k_off,
            self.params.dt,
            self.params.measurement_time,
        )

        # Convert states to intensity (on=intensity, off=0)
        intensity = states[:n_frames] * self.params.single_molecule_intensity

        return intensity

    def _simple_two_state_blinking(
        self, k_on: float, k_off: float, dt: float, measurement_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple on/off blinking model.

        Args:
            k_on: On transition rate
            k_off: Off transition rate
            dt: Sampling time
            measurement_time: Total measurement time

        Returns:
            Tuple of (time_points, state_values)
        """
        # Calculate parameters
        k_sum = k_on + k_off
        p_up = k_on / k_sum
        char_time = 1 / k_on + 1 / k_off

        # Generate enough transitions
        n_frames = int(measurement_time / dt)
        max_transitions = int(2 * measurement_time / char_time)

        # Generate transition times
        transition_times = []
        current_time = 0
        current_state = np.random.binomial(1, p_up)

        while (
            current_time < measurement_time and len(transition_times) < max_transitions
        ):
            if current_state == 0:
                # In off state, wait for on transition
                wait_time = np.random.exponential(1 / k_on)
                current_state = 1
            else:
                # In on state, wait for off transition
                wait_time = np.random.exponential(1 / k_off)
                current_state = 0

            current_time += wait_time
            transition_times.append((current_time, current_state))

        # Convert to discrete time series
        time_points = np.arange(n_frames) * dt
        states = np.zeros(n_frames, dtype=int)

        current_state = transition_times[0][1] if transition_times else 0
        transition_idx = 0

        for i, t in enumerate(time_points):
            # Update state if we've passed a transition
            while (
                transition_idx < len(transition_times)
                and transition_times[transition_idx][0] <= t
            ):
                current_state = transition_times[transition_idx][1]
                transition_idx += 1

            states[i] = current_state

        return time_points, states

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
