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

    def summary(self) -> None:
        """Print a summary of the simulation parameters"""
        print("=== Simulation Parameters ===")
        print(f"  N (emitters):     {self.n_emitters}")
        print(f"  k_on:             {self.k_on:.3f} Hz")
        print(f"  k_off:            {self.k_off:.3f} Hz")
        print(f"  k_sum:            {self.k_on + self.k_off:.3f} Hz")
        print(f"  I_single:         {self.single_molecule_intensity:.2f}")
        print(f"  dt:               {self.dt:.3f} s")
        print(f"  Measurement time: {self.measurement_time:.1f} s")
        print(f"  Noise (sigma):    {self.sigma_noise:.3f}")
        print()
