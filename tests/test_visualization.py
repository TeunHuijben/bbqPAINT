"""
Tests for the visualization module.

These tests exercise the Plotter class. They use the non-interactive "Agg"
backend so no windows are opened, and verify that each method returns a
matplotlib Figure without raising.
"""

import matplotlib

matplotlib.use("Agg")  # noqa: E402  must precede pyplot import in Plotter

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from bbqpaint.analysis.parameters import AnalysisParameters, AnalysisResults  # noqa: E402
from bbqpaint.analysis.parameter_estimation import ParameterEstimator  # noqa: E402
from bbqpaint.simulation import SimulationParameters, TraceGenerator  # noqa: E402
from bbqpaint.utils.visualization import Plotter  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test to avoid resource warnings"""
    yield
    plt.close("all")


@pytest.fixture
def plotter():
    return Plotter()


@pytest.fixture
def sample_trace():
    np.random.seed(7)
    return np.maximum(np.random.normal(30.0, 5.0, 1000), 1.0)


class TestPlotterBasics:
    """Tests for the simple single-panel plot methods"""

    def test_init_default_figsize(self):
        p = Plotter()
        assert p.figsize == (12, 8)

    def test_init_custom_figsize(self):
        p = Plotter(figsize=(6, 4))
        assert p.figsize == (6, 4)

    def test_plot_trace(self, plotter, sample_trace):
        fig = plotter.plot_trace(sample_trace, dt=0.1)
        assert isinstance(fig, plt.Figure)

    def test_plot_trace_without_mean(self, plotter, sample_trace):
        fig = plotter.plot_trace(sample_trace, dt=0.1, show_mean=False)
        assert isinstance(fig, plt.Figure)

    def test_plot_power_spectrum(self, plotter):
        freq = np.linspace(0.1, 10, 200)
        power = 1.0 / freq
        fig = plotter.plot_power_spectrum(freq, power)
        assert isinstance(fig, plt.Figure)

    def test_plot_power_spectrum_with_theory(self, plotter):
        freq = np.linspace(0.1, 10, 200)
        power = 1.0 / freq
        theory = 1.0 / (freq + 0.05)
        fig = plotter.plot_power_spectrum(freq, power, ps_theory=theory)
        assert isinstance(fig, plt.Figure)

    def test_plot_power_spectrum_linear_scale(self, plotter):
        freq = np.linspace(0.1, 10, 200)
        power = 1.0 / freq
        fig = plotter.plot_power_spectrum(freq, power, log_scale=False)
        assert isinstance(fig, plt.Figure)

    def test_plot_bispectrum_log(self, plotter):
        bs = np.abs(np.random.RandomState(3).rand(50, 50)) + 1e-3
        fig = plotter.plot_bispectrum(bs)
        assert isinstance(fig, plt.Figure)

    def test_plot_bispectrum_linear(self, plotter):
        bs = np.abs(np.random.RandomState(3).rand(50, 50)) + 1e-3
        fig = plotter.plot_bispectrum(bs, log_scale=False)
        assert isinstance(fig, plt.Figure)

    def test_plot_bispectrum_with_mask(self, plotter):
        bs = np.abs(np.random.RandomState(4).rand(50, 50)) + 1e-3
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True
        fig = plotter.plot_bispectrum(bs, mask=mask)
        assert isinstance(fig, plt.Figure)

    def test_plot_parameter_uncertainty(self, plotter):
        names = ["k_sum", "s2", "pk_bg"]
        values = [0.45, 100.0, 5.0]
        errors = [0.05, 10.0, 0.5]
        fig = plotter.plot_parameter_uncertainty(names, values, errors)
        assert isinstance(fig, plt.Figure)

    def test_plot_correlation_matrix(self, plotter):
        names = ["a", "b", "c"]
        corr = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.1], [-0.2, 0.1, 1.0]])
        fig = plotter.plot_correlation_matrix(corr, names)
        assert isinstance(fig, plt.Figure)


class TestPlotFitResults:
    """Tests for the plot_fit_results 4-panel figure"""

    def test_full_data(self, plotter, sample_trace):
        freq = np.linspace(0.1, 10, 200)
        data = {
            "trace": sample_trace,
            "power_spectrum": 1.0 / freq,
            "freq_vec": freq,
        }
        fitted = {
            "dt": 0.1,
            "power_spectrum_theory": 1.0 / (freq + 0.05),
            "k_sum": 0.45,
            "s2": 100.0,
        }
        fig = plotter.plot_fit_results(data, fitted)
        assert isinstance(fig, plt.Figure)

    def test_minimal_data(self, plotter):
        """No trace / spectrum keys: only the parameter panel is drawn"""
        fig = plotter.plot_fit_results({}, {"k_sum": 0.45, "s2": 100.0})
        assert isinstance(fig, plt.Figure)


class TestComprehensiveAnalysis:
    """Tests for the comprehensive 6-panel analysis figure"""

    @pytest.fixture
    def full_results(self):
        """Run a full estimation to obtain populated results"""
        sim = SimulationParameters()
        sim.measurement_time = 200.0
        sim.dt = 0.2
        trace = TraceGenerator(sim).generate_trace()[1]

        params = AnalysisParameters()
        params.dt = sim.dt
        params.measurement_time = sim.measurement_time
        params.n_chops = 3
        params.max_iterations = 100

        estimator = ParameterEstimator(trace, params)
        results = estimator.estimate_parameters()
        time_points = np.arange(len(trace)) * params.dt
        return trace, time_points, results, params, sim

    def test_comprehensive_full(self, plotter, full_results):
        trace, time_points, results, params, sim = full_results
        fig = plotter.plot_comprehensive_analysis(
            trace, time_points, results, params, sim_params=sim
        )
        assert isinstance(fig, plt.Figure)

    def test_comprehensive_without_sim_params(self, plotter, full_results):
        trace, time_points, results, params, _ = full_results
        fig = plotter.plot_comprehensive_analysis(trace, time_points, results, params)
        assert isinstance(fig, plt.Figure)

    def test_comprehensive_bispectrum_without_fit(self, plotter):
        """Bispectrum panel present but no fit: hits the no-fit BS branch."""
        np.random.seed(13)
        trace = np.maximum(np.random.normal(30.0, 5.0, 600), 1.0)
        time_points = np.arange(len(trace)) * 0.2

        params = AnalysisParameters()
        params.dt = 0.2
        params.n_chops = 3

        results = AnalysisResults()
        results.power_spectrum = np.abs(np.random.rand(200)) + 0.1
        results.bispectrum = np.abs(np.random.rand(50, 50))  # present
        # no k_sum_fit / c3_fit -> have_fit is False

        fig = plotter.plot_comprehensive_analysis(trace, time_points, results, params)
        assert isinstance(fig, plt.Figure)

    def test_comprehensive_no_fit(self, plotter):
        """Power spectrum present but no fitted params, no bispectrum.

        Exercises the fallback branches (normalize by mean, skip theory/BS panels).
        """
        np.random.seed(11)
        trace = np.maximum(np.random.normal(30.0, 5.0, 500), 1.0)
        time_points = np.arange(len(trace)) * 0.2

        params = AnalysisParameters()
        params.dt = 0.2
        params.n_chops = 3

        results = AnalysisResults()
        results.power_spectrum = np.abs(np.random.rand(200)) + 0.1
        # bispectrum stays None, k_sum_fit/s2_fit stay None

        fig = plotter.plot_comprehensive_analysis(trace, time_points, results, params)
        assert isinstance(fig, plt.Figure)
