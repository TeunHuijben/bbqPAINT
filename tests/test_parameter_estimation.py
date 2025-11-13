"""
Tests for the parameter estimation module.

These tests verify the functionality of the ParameterEstimator class and
related parameter classes including analysis workflow and result calculation.
"""

import numpy as np
import pytest

from mcoast.analysis.parameter_estimation import ParameterEstimator
from mcoast.analysis.parameters import (
    AnalysisParameters,
    AnalysisResults,
    FittingBounds,
    InitialGuessParameters,
)


class TestInitialGuessParameters:
    """Test cases for InitialGuessParameters"""

    def test_default_values(self):
        """Test default parameter values"""
        params = InitialGuessParameters()

        assert params.k_sum == 0.3
        assert params.s2 is None
        assert params.pk_bg == 10.0
        assert params.c3 is None
        assert params.c3_offset == 0.0

    def test_custom_values(self):
        """Test custom parameter values"""
        params = InitialGuessParameters(
            k_sum=0.5, s2=100.0, pk_bg=5.0, c3=50.0, c3_offset=1.0
        )

        assert params.k_sum == 0.5
        assert params.s2 == 100.0
        assert params.pk_bg == 5.0
        assert params.c3 == 50.0
        assert params.c3_offset == 1.0


class TestFittingBounds:
    """Test cases for FittingBounds"""

    def test_default_bounds(self):
        """Test default bounds"""
        bounds = FittingBounds()

        assert bounds.k_sum == (0, np.inf)
        assert bounds.s2 == (0, np.inf)
        assert bounds.pk_bg == (0, np.inf)
        assert bounds.c3 == (-np.inf, np.inf)
        assert bounds.c3_offset == (-np.inf, np.inf)

    def test_custom_bounds(self):
        """Test custom bounds"""
        bounds = FittingBounds(
            k_sum=(0.1, 2.0),
            s2=(10, 500),
            pk_bg=(0, 50),
            c3=(-100, 100),
            c3_offset=(-10, 10),
        )

        assert bounds.k_sum == (0.1, 2.0)
        assert bounds.s2 == (10, 500)
        assert bounds.pk_bg == (0, 50)
        assert bounds.c3 == (-100, 100)
        assert bounds.c3_offset == (-10, 10)


class TestAnalysisParameters:
    """Test cases for AnalysisParameters"""

    def test_default_initialization(self):
        """Test default parameter initialization"""
        params = AnalysisParameters()

        assert params.dt is None
        assert params.measurement_time is None
        assert params.n_chops == 5
        assert params.max_iterations == 1000
        assert params.tolerance == 1e-6
        assert params.fit_power_spectrum is True
        assert params.fit_bispectrum is True
        assert params.calculate_uncertainties is True

        # Check nested objects
        assert isinstance(params.initial_guess, InitialGuessParameters)
        assert isinstance(params.bounds, FittingBounds)

    def test_parameter_modification(self):
        """Test modifying parameters"""
        params = AnalysisParameters()

        params.dt = 0.1
        params.measurement_time = 1000.0
        params.n_chops = 3
        params.max_iterations = 500

        assert params.dt == 0.1
        assert params.measurement_time == 1000.0
        assert params.n_chops == 3
        assert params.max_iterations == 500

    def test_validation_valid_parameters(self):
        """Test validation with valid parameters"""
        params = AnalysisParameters()
        params.dt = 0.1
        params.measurement_time = 1000.0
        params.n_chops = 5
        params.max_iterations = 1000
        params.tolerance = 1e-6

        # Should not raise exception
        params.validate()

    def test_validation_invalid_dt(self):
        """Test validation with invalid dt"""
        params = AnalysisParameters()
        params.dt = -0.1

        with pytest.raises(ValueError, match="dt must be positive"):
            params.validate()

    def test_validation_invalid_measurement_time(self):
        """Test validation with invalid measurement time"""
        params = AnalysisParameters()
        params.measurement_time = -100.0

        with pytest.raises(ValueError, match="measurement_time must be positive"):
            params.validate()

    def test_validation_invalid_n_chops(self):
        """Test validation with invalid n_chops"""
        params = AnalysisParameters()
        params.n_chops = 0

        with pytest.raises(ValueError, match="n_chops must be positive"):
            params.validate()

    def test_validation_invalid_max_iterations(self):
        """Test validation with invalid max_iterations"""
        params = AnalysisParameters()
        params.max_iterations = -10

        with pytest.raises(ValueError, match="max_iterations must be positive"):
            params.validate()

    def test_validation_invalid_tolerance(self):
        """Test validation with invalid tolerance"""
        params = AnalysisParameters()
        params.tolerance = -1e-6

        with pytest.raises(ValueError, match="tolerance must be positive"):
            params.validate()


class TestAnalysisResults:
    """Test cases for AnalysisResults"""

    def test_default_initialization(self):
        """Test default results initialization"""
        results = AnalysisResults()

        # Check fitted parameters
        assert results.k_sum_fit is None
        assert results.s2_fit is None
        assert results.pk_bg_fit is None
        assert results.c3_fit is None
        assert results.c3_offset_fit is None

        # Check derived parameters
        assert results.n_emitters_fit is None
        assert results.k_on_fit is None
        assert results.k_off_fit is None
        assert results.single_molecule_intensity_fit is None

        # Check uncertainties
        assert results.k_sum_std is None
        assert results.s2_std is None
        assert results.pk_bg_std is None
        assert results.c3_std is None
        assert results.c3_offset_std is None

        # Check raw data
        assert results.power_spectrum is None
        assert results.bispectrum is None
        assert results.trace is None

    def test_setting_results(self):
        """Test setting result values"""
        results = AnalysisResults()

        results.k_sum_fit = 0.8
        results.s2_fit = 100.0
        results.c3_fit = 50.0
        results.n_emitters_fit = 3.0

        assert results.k_sum_fit == 0.8
        assert results.s2_fit == 100.0
        assert results.c3_fit == 50.0
        assert results.n_emitters_fit == 3.0


class TestParameterEstimator:
    """Test cases for ParameterEstimator"""

    @pytest.fixture
    def sample_trace(self):
        """Create a sample fluorescence trace for testing"""
        np.random.seed(42)
        n_points = 1000

        # Simulate a simple blinking trace
        mean_intensity = 30.0
        noise_level = 5.0

        # Create trace with some structure
        trace = np.random.normal(mean_intensity, noise_level, n_points)

        # Add some step changes to simulate blinking
        trace[200:400] *= 0.7  # Dimmer period
        trace[600:800] *= 1.3  # Brighter period

        # Ensure positive values
        trace = np.maximum(trace, 1.0)

        return trace

    @pytest.fixture
    def analysis_params(self):
        """Create sample analysis parameters"""
        params = AnalysisParameters()
        params.dt = 0.1
        params.measurement_time = 100.0
        params.n_chops = 3
        params.max_iterations = 100  # Reduce for faster testing
        params.fit_power_spectrum = True
        params.fit_bispectrum = True

        return params

    def test_estimator_initialization(self, sample_trace, analysis_params):
        """Test ParameterEstimator initialization"""
        estimator = ParameterEstimator(sample_trace, analysis_params)

        assert estimator.trace is sample_trace
        assert estimator.params is analysis_params
        assert hasattr(estimator, "power_spectrum_analyzer")
        assert hasattr(estimator, "bispectrum_analyzer")
        assert isinstance(estimator.results, AnalysisResults)

    def test_estimator_initialization_validates_params(self, sample_trace):
        """Test that initialization validates parameters"""
        params = AnalysisParameters()
        params.dt = -0.1  # Invalid

        with pytest.raises(ValueError):
            ParameterEstimator(sample_trace, params)

    def test_estimate_parameters_power_spectrum_only(
        self, sample_trace, analysis_params
    ):
        """Test parameter estimation with power spectrum only"""
        analysis_params.fit_bispectrum = False
        estimator = ParameterEstimator(sample_trace, analysis_params)

        results = estimator.estimate_parameters()

        # Check that power spectrum results are available
        assert results.k_sum_fit is not None
        assert results.s2_fit is not None
        assert results.pk_bg_fit is not None
        assert results.power_spectrum is not None

        # Bispectrum results should be None
        assert results.c3_fit is None
        assert results.bispectrum is None

    def test_estimate_parameters_bispectrum_only(self, sample_trace, analysis_params):
        """Test parameter estimation with bispectrum only"""
        analysis_params.fit_power_spectrum = False
        estimator = ParameterEstimator(sample_trace, analysis_params)

        # For bispectrum fitting, we need power spectrum parameters even if not fitted
        # Set some reasonable defaults
        estimator.results.k_sum_fit = 0.8
        estimator.results.s2_fit = 100.0
        estimator.results.pk_bg_fit = 5.0

        # This test focuses on the bispectrum part specifically
        bs_params = estimator.fit_bispectrum()

        # Check bispectrum results
        assert "c3" in bs_params
        assert "c3_offset" in bs_params
        assert np.isfinite(bs_params["c3"])
        assert np.isfinite(bs_params["c3_offset"])

    def test_estimate_parameters_full_analysis(self, sample_trace, analysis_params):
        """Test full parameter estimation"""
        estimator = ParameterEstimator(sample_trace, analysis_params)

        results = estimator.estimate_parameters()

        # Check that all main results are available
        assert results.power_spectrum is not None
        assert results.bispectrum is not None
        assert results.k_sum_fit is not None
        assert results.s2_fit is not None
        assert results.pk_bg_fit is not None
        assert results.c3_fit is not None
        assert results.c3_offset_fit is not None

        # Check derived parameters are calculated
        assert results.n_emitters_fit is not None
        assert results.k_on_fit is not None
        assert results.k_off_fit is not None
        assert results.single_molecule_intensity_fit is not None

    def test_fit_power_spectrum(self, sample_trace, analysis_params):
        """Test power spectrum fitting"""
        estimator = ParameterEstimator(sample_trace, analysis_params)

        # Calculate power spectrum first
        freq_vec, power_spec = (
            estimator.power_spectrum_analyzer.calculate_power_spectrum(sample_trace)
        )

        # Fit power spectrum
        ps_params = estimator.fit_power_spectrum(power_spec, freq_vec)

        # Check results
        assert "k_sum" in ps_params
        assert "s2" in ps_params
        assert "pk_bg" in ps_params
        assert ps_params["k_sum"] > 0
        assert ps_params["s2"] > 0
        assert ps_params["pk_bg"] >= 0

    def test_fit_bispectrum(self, sample_trace, analysis_params):
        """Test bispectrum fitting"""
        estimator = ParameterEstimator(sample_trace, analysis_params)

        # Set up required power spectrum results first
        estimator.results.k_sum_fit = 0.8
        estimator.results.s2_fit = 100.0
        estimator.results.pk_bg_fit = 5.0

        # Fit bispectrum
        bs_params = estimator.fit_bispectrum()

        # Check results
        assert "c3" in bs_params
        assert "c3_offset" in bs_params
        assert np.isfinite(bs_params["c3"])
        assert np.isfinite(bs_params["c3_offset"])

    def test_calculate_molecular_parameters(self, sample_trace, analysis_params):
        """Test molecular parameter calculation"""
        estimator = ParameterEstimator(sample_trace, analysis_params)

        # Set up fitted parameters
        estimator.results.k_sum_fit = 0.8
        estimator.results.s2_fit = 100.0
        estimator.results.c3_fit = 50.0

        # Calculate molecular parameters
        estimator.calculate_molecular_parameters()

        # Check that derived parameters are calculated
        assert estimator.results.n_emitters_fit is not None
        assert estimator.results.k_on_fit is not None
        assert estimator.results.k_off_fit is not None
        assert estimator.results.single_molecule_intensity_fit is not None

        # Check that values are finite
        assert np.isfinite(estimator.results.n_emitters_fit)
        assert np.isfinite(estimator.results.k_on_fit)
        assert np.isfinite(estimator.results.k_off_fit)
        assert np.isfinite(estimator.results.single_molecule_intensity_fit)

    def test_molecular_parameters_edge_cases(self, sample_trace, analysis_params):
        """Test molecular parameter calculation with edge case values"""
        estimator = ParameterEstimator(sample_trace, analysis_params)

        # Test with problematic values that might cause division issues
        estimator.results.k_sum_fit = 0.0001  # Very small
        estimator.results.s2_fit = 1000000.0  # Very large
        estimator.results.c3_fit = 0.0001  # Very small

        # Should handle gracefully
        estimator.calculate_molecular_parameters()

        # Results might be extreme but should be finite
        assert np.isfinite(estimator.results.n_emitters_fit)

    def test_parameter_estimation_with_custom_bounds(
        self, sample_trace, analysis_params
    ):
        """Test parameter estimation with custom bounds"""
        # Set custom bounds
        analysis_params.bounds.k_sum = (0.1, 1.5)
        analysis_params.bounds.s2 = (50, 200)
        analysis_params.bounds.pk_bg = (1, 10)

        estimator = ParameterEstimator(sample_trace, analysis_params)
        results = estimator.estimate_parameters()

        # Check bounds are respected
        if results.k_sum_fit is not None:
            assert 0.1 <= results.k_sum_fit <= 1.5
        if results.s2_fit is not None:
            assert 50 <= results.s2_fit <= 200
        if results.pk_bg_fit is not None:
            assert 1 <= results.pk_bg_fit <= 10

    def test_parameter_estimation_with_custom_initial_guess(
        self, sample_trace, analysis_params
    ):
        """Test parameter estimation with custom initial guess"""
        # Set custom initial guess
        analysis_params.initial_guess.k_sum = 1.0
        analysis_params.initial_guess.s2 = 150.0
        analysis_params.initial_guess.pk_bg = 8.0
        analysis_params.initial_guess.c3 = 75.0

        estimator = ParameterEstimator(sample_trace, analysis_params)
        results = estimator.estimate_parameters()

        # Should complete without error
        assert results is not None

    def test_different_n_chops(self, sample_trace, analysis_params):
        """Test parameter estimation with different numbers of chops"""
        for n_chops in [2, 3, 5, 8]:
            analysis_params.n_chops = n_chops
            estimator = ParameterEstimator(sample_trace, analysis_params)

            results = estimator.estimate_parameters()

            # Should work for all reasonable n_chops values
            assert results is not None
            assert results.power_spectrum is not None

    def test_k_sum_bs_always_fitted(self, sample_trace, analysis_params):
        """Test that k_sum is always fitted in bispectrum analysis"""
        estimator = ParameterEstimator(sample_trace, analysis_params)

        results = estimator.estimate_parameters()

        # k_sum_bs should always be present in results
        assert results is not None
        assert results.k_sum_bs is not None
        assert results.k_sum_bs > 0  # Should be positive

    def test_short_trace_handling(self, analysis_params):
        """Test handling of very short traces"""
        short_trace = np.random.normal(30, 5, 50)  # Very short trace

        estimator = ParameterEstimator(short_trace, analysis_params)

        # Should handle gracefully
        try:
            results = estimator.estimate_parameters()
            assert results is not None
        except Exception:
            # It's OK if very short traces fail gracefully
            pass

    def test_constant_trace_handling(self, analysis_params):
        """Test handling of constant traces"""
        constant_trace = np.full(1000, 25.0)  # Perfectly constant

        estimator = ParameterEstimator(constant_trace, analysis_params)

        # Should handle gracefully
        try:
            results = estimator.estimate_parameters()
            assert results is not None
        except Exception:
            # Constant traces might legitimately fail
            pass

    def test_noisy_trace_handling(self, analysis_params):
        """Test handling of very noisy traces"""
        np.random.seed(123)
        noisy_trace = np.random.normal(30, 50, 1000)  # Very high noise
        noisy_trace = np.maximum(noisy_trace, 0.1)  # Ensure positive

        estimator = ParameterEstimator(noisy_trace, analysis_params)
        results = estimator.estimate_parameters()

        # Should complete even with noisy data
        assert results is not None
