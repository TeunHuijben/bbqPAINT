"""
Parameter classes for mCOAST simulation and analysis.

This module contains all parameter containers used throughout the mCOAST package,
including base classes, simulation parameters, analysis parameters, and result containers.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseParameters:
    """Base class for all parameter containers"""

    def __init__(self):
        self.dt: Optional[float] = None  # Sampling time
        self.measurement_time: Optional[float] = None  # Total measurement time

    def validate(self) -> None:
        """Validate parameter values"""
        if self.dt is not None and self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.measurement_time is not None and self.measurement_time <= 0:
            raise ValueError("measurement_time must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> "BaseParameters":
        """Create from dictionary"""
        instance = cls()
        for key, value in params_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    def update(self, **kwargs) -> None:
        """Update parameters from keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class SimulationParameters(BaseParameters):
    """Parameters for trace simulation"""

    def __init__(self):
        super().__init__()

        # Core blinking parameters
        self.k_on: Optional[float] = None  # On rate
        self.k_off: Optional[float] = None  # Off rate
        self.n_emitters: Optional[int] = None  # Number of emitters

        # Physical parameters
        self.single_molecule_intensity: Optional[float] = None  # Isingle
        self.snr: Optional[float] = None  # Signal-to-noise ratio

        # Noise parameters
        self.noise_sigma: Optional[float] = None  # Background noise std
        self.outlier_rate: Optional[float] = None  # Outlier contamination rate
        self.outlier_value: Optional[float] = None  # Outlier intensity value

        # Advanced models
        self.model_type: str = "simple"  # simple, two_excited, dynamic_disorder
        self.k_off1: Optional[float] = None  # For two excited states
        self.k_off2: Optional[float] = None  # For two excited states
        self.w1: Optional[float] = None  # Weight for two excited states
        self.gamma_shape_on: Optional[float] = None  # For dynamic disorder
        self.gamma_shape_off: Optional[float] = None  # For dynamic disorder

    def validate(self) -> None:
        """Validate simulation parameters"""
        super().validate()
        if self.k_on is not None and self.k_on <= 0:
            raise ValueError("k_on must be positive")
        if self.k_off is not None and self.k_off <= 0:
            raise ValueError("k_off must be positive")
        if self.n_emitters is not None and self.n_emitters <= 0:
            raise ValueError("n_emitters must be positive")
        if (
            self.single_molecule_intensity is not None
            and self.single_molecule_intensity <= 0
        ):
            raise ValueError("single_molecule_intensity must be positive")


@dataclass
class InitialGuessParameters:
    """Initial parameter guesses for fitting"""

    k_sum: float = 0.3
    s2: Optional[float] = None  # Will be calculated from data
    pk_bg: float = 10.0
    c3: Optional[float] = None  # Will be calculated from data
    c3_offset: float = 0.0


@dataclass
class FittingBounds:
    """Parameter bounds for fitting"""

    k_sum: Tuple[float, float] = (0, np.inf)
    s2: Tuple[float, float] = (0, np.inf)
    pk_bg: Tuple[float, float] = (0, np.inf)
    c3: Tuple[float, float] = (-np.inf, np.inf)
    c3_offset: Tuple[float, float] = (-np.inf, np.inf)


class AnalysisParameters(BaseParameters):
    """Parameters for trace analysis"""

    def __init__(self):
        super().__init__()

        # Analysis control
        self.n_chops: int = 5  # Number of trace chops (K)
        self.max_iterations: int = 1000  # Max fitting iterations
        self.tolerance: float = 1e-6  # Fitting tolerance

        # Initial guesses for fitting
        self.initial_guess: InitialGuessParameters = InitialGuessParameters()

        # Fitting bounds
        self.bounds: FittingBounds = FittingBounds()

        # Analysis options
        self.fit_power_spectrum: bool = True
        self.fit_bispectrum: bool = True
        self.fit_k_sum_free: bool = False  # Whether to fit kSum in bispectrum
        self.calculate_uncertainties: bool = True

    def validate(self) -> None:
        """Validate analysis parameters"""
        super().validate()
        if self.n_chops <= 0:
            raise ValueError("n_chops must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


class AnalysisResults:
    """Container for analysis results"""

    def __init__(self):
        # Fitted parameters
        self.k_sum_fit: Optional[float] = None
        self.s2_fit: Optional[float] = None
        self.pk_bg_fit: Optional[float] = None
        self.c3_fit: Optional[float] = None
        self.c3_offset_fit: Optional[float] = None

        # Derived parameters
        self.n_emitters_fit: Optional[float] = None
        self.k_on_fit: Optional[float] = None
        self.k_off_fit: Optional[float] = None
        self.single_molecule_intensity_fit: Optional[float] = None

        # Uncertainties
        self.k_sum_std: Optional[float] = None
        self.s2_std: Optional[float] = None
        self.pk_bg_std: Optional[float] = None
        self.c3_std: Optional[float] = None
        self.c3_offset_std: Optional[float] = None

        # Raw data
        self.power_spectrum: Optional[np.ndarray] = None
        self.bispectrum: Optional[np.ndarray] = None
        self.trace: Optional[np.ndarray] = None

        # Analysis metadata
        self.dt: Optional[float] = None
        self.n_chops: Optional[int] = None
        self.measurement_time: Optional[float] = None
        self.n_frames: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def print_summary(self) -> None:
        """Print a summary of the analysis results"""
        print("=== mCOAST Analysis Results ===")
        print(f"Number of emitters: {self.n_emitters_fit:.2f}")
        print(f"On rate (k_on): {self.k_on_fit:.3f}")
        print(f"Off rate (k_off): {self.k_off_fit:.3f}")
        print(f"Single molecule intensity: {self.single_molecule_intensity_fit:.2f}")
        print(f"Total rate (k_sum): {self.k_sum_fit:.3f}")
        print(f"Variance (s2): {self.s2_fit:.2f}")
        print(f"Third cumulant (C3): {self.c3_fit:.2f}")


# Specialized parameter classes for different models
class TwoExcitedStatesParameters(SimulationParameters):
    """Parameters for two excited states model"""

    def __init__(self):
        super().__init__()
        self.model_type = "two_excited"
        self.k_off1: Optional[float] = None
        self.k_off2: Optional[float] = None
        self.w1: Optional[float] = None

    def validate(self) -> None:
        """Validate two excited states parameters"""
        super().validate()
        if self.k_off1 is not None and self.k_off1 <= 0:
            raise ValueError("k_off1 must be positive")
        if self.k_off2 is not None and self.k_off2 <= 0:
            raise ValueError("k_off2 must be positive")
        if self.w1 is not None and not (0 <= self.w1 <= 1):
            raise ValueError("w1 must be between 0 and 1")


class DynamicDisorderParameters(SimulationParameters):
    """Parameters for dynamic disorder model"""

    def __init__(self):
        super().__init__()
        self.model_type = "dynamic_disorder"
        self.gamma_shape_on: Optional[float] = None
        self.gamma_shape_off: Optional[float] = None

    def validate(self) -> None:
        """Validate dynamic disorder parameters"""
        super().validate()
        if self.gamma_shape_on is not None and self.gamma_shape_on <= 0:
            raise ValueError("gamma_shape_on must be positive")
        if self.gamma_shape_off is not None and self.gamma_shape_off <= 0:
            raise ValueError("gamma_shape_off must be positive")
