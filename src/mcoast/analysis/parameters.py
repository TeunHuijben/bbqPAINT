"""
Parameter classes for mCOAST analysis.

This module contains parameter containers specific to the analysis workflow,
including analysis parameters, initial guesses, fitting bounds, and results.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


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


class AnalysisParameters:
    """Parameters for trace analysis"""

    def __init__(self):
        # Analysis control
        self.dt: Optional[float] = None
        self.measurement_time: Optional[float] = None
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
        if self.dt is not None and self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.measurement_time is not None and self.measurement_time <= 0:
            raise ValueError("measurement_time must be positive")
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
        if self.n_emitters_fit is not None:
            print(f"Number of emitters: {self.n_emitters_fit:.2f}")
        if self.k_on_fit is not None:
            print(f"On rate (k_on): {self.k_on_fit:.3f}")
        if self.k_off_fit is not None:
            print(f"Off rate (k_off): {self.k_off_fit:.3f}")
        if self.single_molecule_intensity_fit is not None:
            print(
                f"Single molecule intensity: {self.single_molecule_intensity_fit:.2f}"
            )
        if self.k_sum_fit is not None:
            print(f"Total rate (k_sum): {self.k_sum_fit:.3f}")
        if self.s2_fit is not None:
            print(f"Variance (s2): {self.s2_fit:.2f}")
        if self.c3_fit is not None:
            print(f"Third cumulant (C3): {self.c3_fit:.2f}")

    def summary(self, true_params=None) -> None:
        """
        Print a comprehensive summary of the analysis results.

        Parameters
        ----------
        true_params : SimulationParameters, optional
            If provided, will show comparison with true values
        """
        print("=== Analysis Results ===")

        # Power Spectrum Fit
        print("Power Spectrum Fit:")
        if self.k_sum_fit is not None:
            true_str = ""
            if true_params is not None:
                true_k_sum = true_params.k_on + true_params.k_off
                true_str = f" (true: {true_k_sum:.3f})"
            print(f"  k_sum = {self.k_sum_fit:.3f} Hz{true_str}")

        if self.s2_fit is not None:
            print(f"  s2 = {self.s2_fit:.3f}")

        if self.pk_bg_fit is not None:
            print(f"  pk_bg = {self.pk_bg_fit:.3f}")

        # Bispectrum Fit
        print("\nBispectrum Fit:")
        if self.c3_fit is not None:
            print(f"  C3 = {self.c3_fit:.3f}")

        if self.c3_offset_fit is not None:
            print(f"  C3_offset = {self.c3_offset_fit:.3f}")

        # Derived Parameters
        print("\nDerived Parameters:")
        if self.n_emitters_fit is not None:
            true_str = ""
            if true_params is not None:
                true_str = f" (true: {true_params.n_emitters})"
            print(f"  N = {self.n_emitters_fit:.2f}{true_str}")

        if self.k_on_fit is not None:
            true_str = ""
            if true_params is not None:
                true_str = f" (true: {true_params.k_on:.3f})"
            print(f"  k_on = {self.k_on_fit:.3f} Hz{true_str}")

        if self.k_off_fit is not None:
            true_str = ""
            if true_params is not None:
                true_str = f" (true: {true_params.k_off:.3f})"
            print(f"  k_off = {self.k_off_fit:.3f} Hz{true_str}")

        if self.single_molecule_intensity_fit is not None:
            true_str = ""
            if true_params is not None:
                true_str = f" (true: {true_params.single_molecule_intensity:.2f})"
            print(f"  I_single = {self.single_molecule_intensity_fit:.2f}{true_str}")

        print()
