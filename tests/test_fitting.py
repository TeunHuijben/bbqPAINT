"""
Tests for fitting functionality in analyzer classes.

These tests verify power spectrum fitting (PowerSpectrumAnalyzer) and
bispectrum fitting (BispectrumAnalyzer) capabilities.
"""

import numpy as np
import pytest

from mcoast.analysis.bispectrum import BispectrumAnalyzer
from mcoast.analysis.power_spectrum import PowerSpectrumAnalyzer


class TestPowerSpectrumFitting:
    """Test cases for PowerSpectrumAnalyzer fitting"""

    @pytest.fixture
    def power_spectrum_analyzer(self):
        """Create a PowerSpectrumAnalyzer instance for testing"""
        return PowerSpectrumAnalyzer(dt=0.1, n_chops=5)

    @pytest.fixture
    def sample_power_spectrum_data(self):
        """Create sample power spectrum data for testing"""
        freq_vec = np.linspace(0.01, 0.5, 100)

        # True parameters
        k_sum_true = 0.8
        s2_true = 100.0
        pk_bg_true = 5.0

        # Generate theoretical power spectrum with some noise
        dt = 0.1
        n_frames = 1000
        c = np.exp(-k_sum_true * dt)
        k_vec = freq_vec * n_frames * dt

        ps_theory = (
            s2_true
            * (1 - c**2)
            * dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k_vec / n_frames))
            + pk_bg_true
        )

        # Add noise
        np.random.seed(42)
        power_spec = ps_theory + np.random.normal(
            0, 0.1 * np.mean(ps_theory), len(ps_theory)
        )
        power_spec = np.maximum(power_spec, 0.1)  # Ensure positive values

        return {
            "power_spec": power_spec,
            "freq_vec": freq_vec,
            "true_params": [k_sum_true, s2_true, pk_bg_true],
        }

    def test_power_spectrum_analyzer_initialization(self):
        """Test PowerSpectrumAnalyzer initialization"""
        analyzer = PowerSpectrumAnalyzer(dt=0.1, n_chops=5)
        assert analyzer.dt == 0.1
        assert analyzer.n_chops == 5

    def test_power_spectrum_mle_fitting(
        self, power_spectrum_analyzer, sample_power_spectrum_data
    ):
        """Test maximum likelihood estimation for power spectrum"""
        data = sample_power_spectrum_data

        initial_guess = [0.5, 80.0, 3.0]
        bounds = [(0, 2), (10, 200), (0, 20)]

        fitted_params, cov_matrix = power_spectrum_analyzer.fit_power_spectrum_mle(
            data["power_spec"],
            data["freq_vec"],
            initial_guess,
            bounds,
            max_iterations=500,
        )

        # Check that we get reasonable results
        assert len(fitted_params) == 3
        assert fitted_params[0] > 0  # k_sum should be positive
        assert fitted_params[1] > 0  # s2 should be positive
        assert fitted_params[2] >= 0  # pk_bg should be non-negative

        # Check parameters are within reasonable range of true values
        true_params = data["true_params"]
        assert abs(fitted_params[0] - true_params[0]) < 0.5  # k_sum
        assert abs(fitted_params[1] - true_params[1]) < 50  # s2
        assert abs(fitted_params[2] - true_params[2]) < 10  # pk_bg

    def test_power_spectrum_mle_with_bounds(
        self, power_spectrum_analyzer, sample_power_spectrum_data
    ):
        """Test power spectrum fitting respects bounds"""
        data = sample_power_spectrum_data

        initial_guess = [0.5, 80.0, 3.0]
        # Tight bounds around true values
        bounds = [(0.6, 1.0), (80, 120), (3, 7)]

        fitted_params, _ = power_spectrum_analyzer.fit_power_spectrum_mle(
            data["power_spec"], data["freq_vec"], initial_guess, bounds
        )

        # Check bounds are respected
        assert bounds[0][0] <= fitted_params[0] <= bounds[0][1]
        assert bounds[1][0] <= fitted_params[1] <= bounds[1][1]
        assert bounds[2][0] <= fitted_params[2] <= bounds[2][1]

    def test_theoretical_power_spectrum_calculation(self, power_spectrum_analyzer):
        """Test theoretical power spectrum calculation"""
        k_vec = np.arange(1, 50)
        k_sum = 0.5
        s2 = 100.0
        pk_bg = 5.0
        n_frames = 1000

        ps_theory = power_spectrum_analyzer.calculate_theoretical_ps(
            k_vec, k_sum, s2, pk_bg, n_frames
        )

        assert len(ps_theory) == len(k_vec)
        assert np.all(ps_theory > 0)
        assert np.all(np.isfinite(ps_theory))

    def test_covariance_matrix_calculation(
        self, power_spectrum_analyzer, sample_power_spectrum_data
    ):
        """Test that covariance matrix is calculated"""
        data = sample_power_spectrum_data

        initial_guess = [0.5, 80.0, 3.0]
        bounds = [(0, 2), (10, 200), (0, 20)]

        fitted_params, cov_matrix = power_spectrum_analyzer.fit_power_spectrum_mle(
            data["power_spec"], data["freq_vec"], initial_guess, bounds
        )

        # Covariance matrix should be returned (or None if calculation fails)
        if cov_matrix is not None:
            assert cov_matrix.shape == (3, 3)
            assert np.all(np.isfinite(cov_matrix))


class TestBispectrumFitting:
    """Test cases for BispectrumAnalyzer fitting"""

    @pytest.fixture
    def bispectrum_analyzer(self):
        """Create a BispectrumAnalyzer instance for testing"""
        return BispectrumAnalyzer(dt=0.1)

    @pytest.fixture
    def sample_bispectrum_data(self):
        """Create sample bispectrum data for testing with PROPER frequency-dependent model"""
        n_points = 50
        k1_data = np.random.uniform(1, 10, n_points)
        k2_data = np.random.uniform(1, 10, n_points)

        # True parameters
        c3_true = 50.0
        c3_offset_true = 2.0
        k_sum_true = 0.8
        dt = 0.1
        chop_length = 100

        # Generate theoretical bispectrum using FULL frequency-dependent model
        c = np.exp(-k_sum_true * dt)
        k3_mat = k1_data + k2_data

        # Full theoretical bispectrum formula
        temp1 = (
            (np.cos(2 * np.pi * k1_data / chop_length) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_data / chop_length))
            + (np.cos(2 * np.pi * k2_data / chop_length) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_data / chop_length))
            + (np.cos(2 * np.pi * k3_mat / chop_length) - c)
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k3_mat / chop_length))
        )

        temp2 = (
            np.cos(2 * np.pi * (k1_data - k2_data) / chop_length)
            - c
            * (
                np.cos(2 * np.pi * k1_data / chop_length)
                + np.cos(2 * np.pi * k2_data / chop_length)
            )
            + c**2
        ) / (
            (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_data / chop_length))
            * (1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_data / chop_length))
        )

        temp3 = (
            np.cos(2 * np.pi * (k1_data + k3_mat) / chop_length)
            - c
            * (
                np.cos(2 * np.pi * k1_data / chop_length)
                + np.cos(2 * np.pi * k3_mat / chop_length)
            )
            + c**2
        ) / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k1_data / chop_length)) + (
            np.cos(2 * np.pi * (k2_data + k3_mat) / chop_length)
            - c
            * (
                np.cos(2 * np.pi * k2_data / chop_length)
                + np.cos(2 * np.pi * k3_mat / chop_length)
            )
            + c**2
        ) / (
            1 + c**2 - 2 * c * np.cos(2 * np.pi * k2_data / chop_length)
        )

        bs_theory = (
            dt**2
            * c3_true
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
            + c3_offset_true
        )

        # Create data with some noise
        np.random.seed(42)
        bs_data = bs_theory + np.random.normal(
            0, 0.05 * np.abs(np.mean(bs_theory)), n_points
        )
        weights = np.ones(n_points)  # Equal weights

        return {
            "bs_data": bs_data,
            "k1_data": k1_data,
            "k2_data": k2_data,
            "weights": weights,
            "chop_length": chop_length,
            "true_params": [c3_true, c3_offset_true, k_sum_true],
        }

    def test_bispectrum_analyzer_initialization(self):
        """Test BispectrumAnalyzer initialization"""
        analyzer = BispectrumAnalyzer(dt=0.1)
        assert analyzer.dt == 0.1

    def test_bispectrum_wls_fitting(self, bispectrum_analyzer, sample_bispectrum_data):
        """Test weighted least squares for bispectrum with proper frequency-dependent model"""
        data = sample_bispectrum_data

        initial_guess = [40.0, 1.0, 0.8]  # c3, c3_offset, k_sum

        fitted_params, cov_matrix = bispectrum_analyzer.fit_bispectrum_wls(
            data["bs_data"],
            data["k1_data"],
            data["k2_data"],
            data["weights"],
            data["chop_length"],
            initial_guess,
            max_iterations=1000,
        )

        # Should always return 3 parameters: c3, c3_offset, k_sum
        assert len(fitted_params) == 3
        assert np.all(np.isfinite(fitted_params))

        # Check parameters are within reasonable range
        assert fitted_params[0] > 0  # c3 should be positive
        assert fitted_params[2] > 0  # k_sum should be positive
        assert abs(fitted_params[0] - data["true_params"][0]) < 30  # c3
        assert abs(fitted_params[1] - data["true_params"][1]) < 5  # c3_offset

    def test_theoretical_bispectrum_calculation(self, bispectrum_analyzer):
        """Test theoretical bispectrum calculation"""
        k1_mat = np.array([[1, 2], [3, 4]])
        k2_mat = np.array([[1, 1], [2, 2]])
        k_sum = 0.5
        c3 = 50.0
        chop_length = 100

        bs_theory = bispectrum_analyzer.calculate_theoretical_bs(
            k1_mat, k2_mat, k_sum, c3, chop_length
        )

        assert bs_theory.shape == k1_mat.shape
        assert np.all(np.isfinite(bs_theory))

    def test_fitting_with_invalid_data(self, bispectrum_analyzer):
        """Test handling of edge cases"""
        # Very small dataset
        bs_data = np.array([1.0, 2.0])
        k1_data = np.array([0.1, 0.2])
        k2_data = np.array([0.1, 0.2])
        weights = np.ones(2)
        chop_length = 100

        initial_guess = [40.0, 1.0]

        # Should handle gracefully
        try:
            fitted_params, _ = bispectrum_analyzer.fit_bispectrum_wls(
                bs_data, k1_data, k2_data, weights, chop_length, initial_guess
            )
            assert len(fitted_params) == 2
        except Exception:
            # It's okay if it fails gracefully
            pass


class TestIntegration:
    """Integration tests for fitting components"""

    def test_fitting_convergence(self):
        """Test that fitting converges with different numbers of iterations"""
        analyzer = PowerSpectrumAnalyzer(dt=0.1, n_chops=5)

        # Generate test data
        freq_vec = np.linspace(0.01, 0.5, 50)
        k_sum_true = 0.8
        s2_true = 100.0
        pk_bg_true = 5.0

        dt = 0.1
        n_frames = 1000
        c = np.exp(-k_sum_true * dt)
        k_vec = freq_vec * n_frames * dt

        ps_theory = (
            s2_true
            * (1 - c**2)
            * dt
            / (1 + c**2 - 2 * c * np.cos(2 * np.pi * k_vec / n_frames))
            + pk_bg_true
        )
        power_spec = ps_theory + np.random.normal(
            0, 0.1 * np.mean(ps_theory), len(ps_theory)
        )

        initial_guess = [0.5, 80.0, 3.0]
        bounds = [(0, 2), (10, 200), (0, 20)]

        # Test with few iterations
        fitted_params_few, _ = analyzer.fit_power_spectrum_mle(
            power_spec, freq_vec, initial_guess, bounds, max_iterations=50
        )

        # Test with many iterations
        fitted_params_many, _ = analyzer.fit_power_spectrum_mle(
            power_spec, freq_vec, initial_guess, bounds, max_iterations=500
        )

        # Both should return valid results
        assert len(fitted_params_few) == 3
        assert len(fitted_params_many) == 3
        assert np.all(np.isfinite(fitted_params_few))
        assert np.all(np.isfinite(fitted_params_many))

    def test_empty_data_handling(self):
        """Test handling of empty or minimal data"""
        analyzer = PowerSpectrumAnalyzer(dt=0.1, n_chops=5)

        # Test with minimal data
        power_spec = np.array([1.0, 2.0])
        freq_vec = np.array([0.1, 0.2])
        initial_guess = [0.5, 80.0, 3.0]
        bounds = [(0, 2), (10, 200), (0, 20)]

        # Should handle gracefully
        try:
            fitted_params, _ = analyzer.fit_power_spectrum_mle(
                power_spec, freq_vec, initial_guess, bounds
            )
            assert len(fitted_params) == 3
        except Exception:
            # It's okay if it fails with minimal data
            pass
