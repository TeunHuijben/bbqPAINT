"""
Trace generation classes for bbqPAINT.

This module contains the main classes for generating fluorescence traces
with blinking behavior and noise characteristics.
"""

from typing import Tuple

import numpy as np

from bbqpaint.simulation.noise_models import add_gaussian_noise
from bbqpaint.simulation.parameters import SimulationParameters


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

        # Setup noise parameters
        self.noise_sigma = self._calculate_noise_sigma()

    def _calculate_noise_sigma(self) -> float:
        """Calculate noise sigma based on parameters"""
        return self.params.sigma_noise

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
        intensity = add_gaussian_noise(intensity, self.noise_sigma)

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
