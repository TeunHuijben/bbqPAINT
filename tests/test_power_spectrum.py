"""
Tests for PowerSpectrumAnalyzer helpers not exercised by the estimation tests
(cumulant calculation and the odd-length-trace branch).
"""

import numpy as np

from bbqpaint.analysis.power_spectrum import PowerSpectrumAnalyzer


class TestCalculatePowerSpectrum:
    def test_odd_length_trace_trimmed(self):
        """An odd number of frames is trimmed to even before the FFT"""
        analyzer = PowerSpectrumAnalyzer(dt=0.1, n_chops=3)
        trace = np.random.RandomState(0).rand(1001)  # odd length

        freq_vec, power_spec = analyzer.calculate_power_spectrum(trace)

        assert len(freq_vec) == len(power_spec)
        assert np.all(np.isfinite(power_spec))

    def test_even_length_trace(self):
        analyzer = PowerSpectrumAnalyzer(dt=0.1, n_chops=3)
        trace = np.random.RandomState(0).rand(1000)

        freq_vec, power_spec = analyzer.calculate_power_spectrum(trace)

        assert len(freq_vec) == len(power_spec)


class TestCalculateCumulants:
    def test_returns_moments_and_cumulants(self):
        analyzer = PowerSpectrumAnalyzer(dt=0.1, n_chops=3)
        trace = np.random.RandomState(1).normal(10.0, 2.0, 2000)

        cumulants = analyzer.calculate_cumulants(trace, order=3)

        assert "moment_1" in cumulants
        assert "moment_3" in cumulants
        assert "mean" in cumulants
        assert "variance" in cumulants
        assert "skewness" in cumulants
        assert "third_moment" in cumulants

    def test_mean_and_variance_reasonable(self):
        analyzer = PowerSpectrumAnalyzer(dt=0.1, n_chops=3)
        trace = np.random.RandomState(2).normal(10.0, 2.0, 5000)

        cumulants = analyzer.calculate_cumulants(trace)

        assert abs(cumulants["mean"] - 10.0) < 0.5
        assert abs(cumulants["variance"] - 4.0) < 0.5

    def test_order_two_omits_third_moment(self):
        analyzer = PowerSpectrumAnalyzer(dt=0.1, n_chops=3)
        trace = np.random.RandomState(3).normal(10.0, 2.0, 1000)

        cumulants = analyzer.calculate_cumulants(trace, order=2)

        assert "third_moment" not in cumulants
        assert "moment_2" in cumulants
