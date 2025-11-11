"""
Simple parameter classes for mCOAST simulation.
"""

from dataclasses import dataclass


@dataclass
class SimulationParameters:
    """Simple simulation parameters"""

    # Core blinking parameters
    k_on: float = 0.15  # On rate [1/s]
    k_off: float = 0.3  # Off rate [1/s]
    n_emitters: int = 4  # Number of targets/emitters

    # Timing parameters
    dt: float = 0.2  # Sampling time [s]
    measurement_time: float = 3600  # Measurement time [s]

    # Intensity and noise
    single_molecule_intensity: float = 1.0  # Isingle
    sigma_noise: float = 0.2  # std of background noise

    def validate(self) -> None:
        """Basic validation"""
        if self.k_on <= 0:
            raise ValueError("k_on must be positive")
        if self.k_off <= 0:
            raise ValueError("k_off must be positive")
        if self.n_emitters <= 0:
            raise ValueError("n_emitters must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.measurement_time <= 0:
            raise ValueError("measurement_time must be positive")
