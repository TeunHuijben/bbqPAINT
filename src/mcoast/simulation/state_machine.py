"""
State machine classes for different blinking models.

This module contains classes that implement different blinking models
including simple two-state, two excited states, and dynamic disorder models.
"""

from typing import Tuple

import numpy as np

from mcoast.simulation.parameters import (
    DynamicDisorderParameters,
    SimulationParameters,
    TwoExcitedStatesParameters,
)


class BlinkingStateMachine:
    """Handles different blinking models"""

    def __init__(self, model_type: str = "simple"):
        """
        Initialize state machine.

        Args:
            model_type: Type of blinking model ("simple", "two_excited", "dynamic_disorder")
        """
        self.model_type = model_type

    def simple_two_state(
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

    def two_excited_states(
        self,
        k_on: float,
        k_off1: float,
        k_off2: float,
        w1: float,
        dt: float,
        measurement_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two excited states model.

        Args:
            k_on: On transition rate
            k_off1: Off rate from first excited state
            k_off2: Off rate from second excited state
            w1: Weight for first excited state
            dt: Sampling time
            measurement_time: Total measurement time

        Returns:
            Tuple of (time_points, state_values)
        """
        # Placeholder implementation
        # This would implement the more complex two-state model
        return self.simple_two_state(k_on, k_off1, dt, measurement_time)

    def dynamic_disorder(
        self,
        k_on: float,
        k_off: float,
        gamma_shape_on: float,
        gamma_shape_off: float,
        dt: float,
        measurement_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dynamic disorder model with rate fluctuations.

        Args:
            k_on: Mean on transition rate
            k_off: Mean off transition rate
            gamma_shape_on: Shape parameter for on rate distribution
            gamma_shape_off: Shape parameter for off rate distribution
            dt: Sampling time
            measurement_time: Total measurement time

        Returns:
            Tuple of (time_points, state_values)
        """
        # Placeholder implementation
        # This would implement the dynamic disorder model with gamma-distributed rates
        return self.simple_two_state(k_on, k_off, dt, measurement_time)

    def generate_trace(
        self, params: SimulationParameters
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trace based on model type and parameters.

        Args:
            params: SimulationParameters object

        Returns:
            Tuple of (time_points, state_values)
        """
        if self.model_type == "simple":
            return self.simple_two_state(
                params.k_on, params.k_off, params.dt, params.measurement_time
            )
        elif self.model_type == "two_excited":
            if isinstance(params, TwoExcitedStatesParameters):
                return self.two_excited_states(
                    params.k_on,
                    params.k_off1,
                    params.k_off2,
                    params.w1,
                    params.dt,
                    params.measurement_time,
                )
            else:
                raise ValueError(
                    "TwoExcitedStatesParameters required for two_excited model"
                )
        elif self.model_type == "dynamic_disorder":
            if isinstance(params, DynamicDisorderParameters):
                return self.dynamic_disorder(
                    params.k_on,
                    params.k_off,
                    params.gamma_shape_on,
                    params.gamma_shape_off,
                    params.dt,
                    params.measurement_time,
                )
            else:
                raise ValueError(
                    "DynamicDisorderParameters required for dynamic_disorder model"
                )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
