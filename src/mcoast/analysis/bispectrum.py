"""
Bispectrum analysis classes for mCOAST.

This module contains classes for calculating and analyzing bispectra
from fluorescence traces, with integrated fitting using the full
frequency-dependent theoretical model.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


class BispectrumAnalyzer:
    """Bispectrum calculation and analysis"""

    def __init__(self, dt: float):
        """
        Initialize bispectrum analyzer.

        Args:
            dt: Sampling time
        """
        self.dt = dt

    def calculate_bispectrum(
        self, chopped_traces: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate bispectrum from chopped traces.

        Args:
            chopped_traces: Matrix of chopped traces [chop_length x n_chops]

        Returns:
            Tuple of (bispectrum, k1_matrix, k2_matrix)
        """
        chop_length, n_chops = chopped_traces.shape

        # Ensure even number of frames
        if chop_length % 2 == 1:
            chopped_traces = chopped_traces[:-1, :]
            chop_length = len(chopped_traces)

        # Calculate FFT for each chop
        fft_chops = self.dt * np.fft.fftshift(
            np.fft.fft(chopped_traces, axis=0), axes=0
        )

        # Take positive frequencies only
        fft_pos = fft_chops[chop_length // 2 :, :]

        # Calculate bispectrum dimensions
        n_b = chop_length // 2

        # Initialize bispectrum matrices
        bispectrum = np.zeros((n_b // 2, n_b))
        k1_mat = np.zeros((n_b // 2, n_b))
        k2_mat = np.zeros((n_b // 2, n_b))

        # Calculate bispectrum in triangular region
        for i in range(n_b // 2):
            for j in range(n_b):
                k1 = j
                k2 = i
                k3 = k1 + k2

                # Check if k3 is within bounds
                if k3 < n_b and k1 >= k2:
                    # Calculate bispectrum component
                    fx = fft_pos[k1, :]
                    fy = fft_pos[k2, :]
                    fz = fft_pos[k3, :]

                    # Bispectrum formula
                    bs_component = np.real(fx * fy * np.conj(fz))
                    bispectrum[i, j] = np.mean(bs_component) / (chop_length * self.dt)

                    # Store frequency indices
                    k1_mat[i, j] = k1
                    k2_mat[i, j] = k2

        return bispectrum, k1_mat, k2_mat

    def calculate_full_bispectrum(
        self, chopped_traces: np.ndarray, freq_max: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate FULL bispectrum including negative frequencies (for visualization).

        Creating a square matrix that includes both positive and negative frequencies up to freq_max.

        Args:
            chopped_traces: Matrix of chopped traces [chop_length x n_chops]
            freq_max: Maximum frequency to include (in rad/s). If None, uses full range.

        Returns:
            Tuple of (bispectrum, freq_1_matrix, freq_2_matrix) in Hz
        """
        chop_length, n_chops = chopped_traces.shape

        # Ensure even number of frames
        if chop_length % 2 == 1:
            chopped_traces = chopped_traces[:-1, :]
            chop_length = chopped_traces.shape[0]

        # Calculate FFT for each chop (fftshift for zero-centered frequencies)
        fft_chops = self.dt * np.fft.fftshift(
            np.fft.fft(chopped_traces, axis=0), axes=0
        )

        # Create frequency vector (zero-centered)
        freq_vec_hz = np.fft.fftshift(np.fft.fftfreq(chop_length, self.dt))
        freq_vec_rad = freq_vec_hz * 2 * np.pi  # Convert to rad/s

        # Determine frequency range to calculate
        if freq_max is None:
            freq_max = np.max(np.abs(freq_vec_rad))

        # Find indices within freq_max range
        idx_want = np.where(np.abs(freq_vec_rad) <= freq_max)[0]
        n_b = len(idx_want)

        # Create index matrices for k1, k2, k3
        idx_0 = np.where(freq_vec_rad == 0)[0][0]  # Zero frequency index
        idx1_mat = np.tile(idx_want, (n_b, 1))
        idx2_mat = np.tile(idx_want[:, np.newaxis], (1, n_b))
        idx3_mat = idx1_mat + idx2_mat - idx_0

        # Initialize bispectrum
        bispectrum = np.zeros((n_b, n_b))

        # Calculate bispectrum
        for chop in range(n_chops):
            # Extract Fourier components for this chop
            fx = fft_chops[idx1_mat.flatten(), chop]
            fy = fft_chops[idx2_mat.flatten(), chop]
            fz = fft_chops[idx3_mat.flatten(), chop]

            # Calculate bispectrum component
            bs_component = np.real(fx * fy * np.conj(fz))
            bispectrum += bs_component.reshape(n_b, n_b)

        # Average over chops and normalize
        bispectrum = bispectrum / n_chops / (chop_length * self.dt)

        # Create frequency matrices for output
        freq_1_mat = np.tile(freq_vec_hz[idx_want], (n_b, 1))
        freq_2_mat = np.tile(freq_vec_hz[idx_want][:, np.newaxis], (1, n_b))

        return bispectrum, freq_1_mat, freq_2_mat

    def apply_mask(
        self, bispectrum: np.ndarray, k1_mat: np.ndarray, k2_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply masking for fitting region.

        Args:
            bispectrum: Bispectrum matrix
            k1_mat: k1 frequency matrix
            k2_mat: k2 frequency matrix

        Returns:
            Tuple of (masked_bispectrum, masked_k1, masked_k2)
        """
        # Apply triangular mask: k2 > 0, k1 >= k2, k1 + k2 < n_b/2
        mask = (
            (k2_mat > 0) & (k1_mat >= k2_mat) & (k1_mat + k2_mat < bispectrum.shape[0])
        )

        bs_masked = bispectrum[mask]
        k1_masked = k1_mat[mask]
        k2_masked = k2_mat[mask]

        return bs_masked, k1_masked, k2_masked

    def calculate_theoretical_bs(
        self,
        k1_mat: np.ndarray,
        k2_mat: np.ndarray,
        k_sum: float,
        c3: float,
        chop_length: int,
    ) -> np.ndarray:
        """
        Calculate theoretical bispectrum.

        Args:
            k1_mat: k1 frequency matrix
            k2_mat: k2 frequency matrix
            k_sum: Total transition rate
            c3: Third cumulant
            chop_length: Length of each chop

        Returns:
            Theoretical bispectrum
        """
        c = np.exp(-k_sum * self.dt)
        k3_mat = k1_mat + k2_mat

        # Theoretical bispectrum formula
        temp1 = (
            (np.cos(2 * np.pi * k1_mat / chop_length) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length))
            + (np.cos(2 * np.pi * k2_mat / chop_length) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length))
            + (np.cos(2 * np.pi * k3_mat / chop_length) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / chop_length))
        )

        temp2 = (
            np.cos(2 * np.pi * (k1_mat - k2_mat) / chop_length)
            - c
            * (
                np.cos(2 * np.pi * k1_mat / chop_length)
                + np.cos(2 * np.pi * k2_mat / chop_length)
            )
            + c**2
        ) / (
            (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length))
            * (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length))
        )

        temp3 = (
            np.cos(2 * np.pi * (k1_mat + k3_mat) / chop_length)
            - c
            * (
                np.cos(2 * np.pi * k1_mat / chop_length)
                + np.cos(2 * np.pi * k3_mat / chop_length)
            )
            + c**2
        ) / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length)) + (
            np.cos(2 * np.pi * (k2_mat + k3_mat) / chop_length)
            - c
            * (
                np.cos(2 * np.pi * k2_mat / chop_length)
                + np.cos(2 * np.pi * k3_mat / chop_length)
            )
            + c**2
        ) / (
            1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length)
        )

        bs_theory = (
            self.dt**2
            * c3
            * (
                1
                + 2 * c * temp1
                + 2
                * c**2
                * (
                    temp2
                    + temp3
                    / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / chop_length))
                )
            )
        )

        return bs_theory

    def calculate_variance(
        self,
        k1_mat: np.ndarray,
        k2_mat: np.ndarray,
        n_chops: int,
        k_sum: float,
        s2: float,
        pk_bg: float,
        chop_length: int,
    ) -> np.ndarray:
        """
        Calculate bispectrum variance for weighting.

        Args:
            k1_mat: k1 frequency matrix
            k2_mat: k2 frequency matrix
            n_chops: Number of chops
            k_sum: Total transition rate
            s2: Variance parameter
            pk_bg: Background noise level
            chop_length: Length of each chop

        Returns:
            Variance matrix for weighting
        """
        k3_mat = k1_mat + k2_mat
        c = np.exp(-k_sum * self.dt)

        # Calculate power spectrum values
        pk1 = (
            s2
            * (1 - c**2)
            * self.dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length))
            + pk_bg
        )
        pk2 = (
            s2
            * (1 - c**2)
            * self.dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length))
            + pk_bg
        )
        pk3 = (
            s2
            * (1 - c**2)
            * self.dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / chop_length))
            + pk_bg
        )

        # Bispectrum variance formula
        variance = pk1 * pk2 * pk3 / n_chops

        return variance

    def fit_bispectrum_wls(
        self,
        bs_data: np.ndarray,
        k1_data: np.ndarray,
        k2_data: np.ndarray,
        weights: np.ndarray,
        chop_length: int,
        initial_guess: List[float],
        max_iterations: int = 1000,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Weighted least squares fitting for bispectrum using the full theoretical model.

        Args:
            bs_data: Bispectrum data (flattened)
            k1_data: k1 frequency data (flattened)
            k2_data: k2 frequency data (flattened)
            weights: Weights for fitting (inverse variance)
            chop_length: Length of each chop
            initial_guess: Initial parameter guess [c3, c3_offset, k_sum]
            max_iterations: Maximum number of iterations

        Returns:
            Tuple of (fitted_parameters, covariance_matrix)
                fitted_parameters: [c3, c3_offset, k_sum]
                covariance_matrix: 3x3 covariance matrix (or None if calculation fails)
        """

        def bispectrum_model(k12_data, *params):
            """
            Bispectrum model function using the full frequency-dependent theoretical model.

            This uses the complete theoretical bispectrum calculation, NOT a constant approximation.
            """
            c3, c3_offset, k_sum = params

            # Extract k1 and k2
            k1_mat = k12_data[:, 0]
            k2_mat = k12_data[:, 1]

            # Calculate theoretical bispectrum using the FULL model
            c = np.exp(-k_sum * self.dt)
            k3_mat = k1_mat + k2_mat

            # Full theoretical bispectrum formula (frequency-dependent)
            temp1 = (
                (np.cos(2 * np.pi * k1_mat / chop_length) - c)
                / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length))
                + (np.cos(2 * np.pi * k2_mat / chop_length) - c)
                / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length))
                + (np.cos(2 * np.pi * k3_mat / chop_length) - c)
                / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / chop_length))
            )

            temp2 = (
                np.cos(2 * np.pi * (k1_mat - k2_mat) / chop_length)
                - c
                * (
                    np.cos(2 * np.pi * k1_mat / chop_length)
                    + np.cos(2 * np.pi * k2_mat / chop_length)
                )
                + c**2
            ) / (
                (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length))
                * (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length))
            )

            temp3 = (
                np.cos(2 * np.pi * (k1_mat + k3_mat) / chop_length)
                - c
                * (
                    np.cos(2 * np.pi * k1_mat / chop_length)
                    + np.cos(2 * np.pi * k3_mat / chop_length)
                )
                + c**2
            ) / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_mat / chop_length)) + (
                np.cos(2 * np.pi * (k2_mat + k3_mat) / chop_length)
                - c
                * (
                    np.cos(2 * np.pi * k2_mat / chop_length)
                    + np.cos(2 * np.pi * k3_mat / chop_length)
                )
                + c**2
            ) / (
                1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_mat / chop_length)
            )

            bs_theory = (
                self.dt**2
                * c3
                * (
                    1
                    + 2 * c * temp1
                    + 2
                    * c**2
                    * (
                        temp2
                        + temp3
                        / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / chop_length))
                    )
                )
                + c3_offset**2
            )

            return bs_theory

        # Prepare data
        k12_data = np.column_stack([k1_data, k2_data])

        # Use all three parameters: c3, c3_offset, k_sum
        p0 = initial_guess[:3]

        # Perform weighted least squares fitting
        try:
            fitted_params, cov_matrix = curve_fit(
                bispectrum_model,
                k12_data,
                bs_data,
                p0=p0,
                sigma=weights,
                maxfev=max_iterations,
                absolute_sigma=True,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
            print(f"Warning: Bispectrum fitting failed, using initial guess: {e}")
            fitted_params = np.array(p0)
            cov_matrix = None

        return fitted_params, cov_matrix
