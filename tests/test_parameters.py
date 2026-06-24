"""
Tests for the result/parameter container methods (to_dict, print_summary, summary).
"""

from bbqpaint.analysis.parameters import AnalysisResults
from bbqpaint.simulation import SimulationParameters


def _populated_results():
    results = AnalysisResults()
    results.k_sum_fit = 0.45
    results.s2_fit = 100.0
    results.pk_bg_fit = 5.0
    results.c3_fit = 50.0
    results.c3_offset_fit = 1.0
    results.n_emitters_fit = 4.0
    results.k_on_fit = 0.15
    results.k_off_fit = 0.30
    results.single_molecule_intensity_fit = 1.0
    return results


class TestAnalysisResultsToDict:
    def test_to_dict_contains_fields(self):
        results = _populated_results()
        d = results.to_dict()

        assert d["k_sum_fit"] == 0.45
        assert d["n_emitters_fit"] == 4.0
        # all public attributes present
        assert "single_molecule_intensity_fit" in d

    def test_to_dict_excludes_private(self):
        results = AnalysisResults()
        results._private = 1  # noqa: SLF001
        assert "_private" not in results.to_dict()


class TestAnalysisResultsPrintSummary:
    def test_print_summary_populated(self, capsys):
        _populated_results().print_summary()
        out = capsys.readouterr().out
        assert "bbqPAINT Analysis Results" in out
        assert "Number of emitters" in out
        assert "Third cumulant" in out

    def test_print_summary_empty(self, capsys):
        AnalysisResults().print_summary()
        out = capsys.readouterr().out
        # Header always printed; no parameter lines for empty results
        assert "bbqPAINT Analysis Results" in out
        assert "Number of emitters" not in out


class TestAnalysisResultsSummary:
    def test_summary_without_true_params(self, capsys):
        _populated_results().summary()
        out = capsys.readouterr().out
        assert "Power Spectrum Fit" in out
        assert "Bispectrum Fit" in out
        assert "Derived Parameters" in out
        assert "true:" not in out

    def test_summary_with_true_params(self, capsys):
        true_params = SimulationParameters()
        _populated_results().summary(true_params=true_params)
        out = capsys.readouterr().out
        assert "true:" in out
        # k_sum comparison uses k_on + k_off of true params
        assert f"{true_params.k_on + true_params.k_off:.3f}" in out

    def test_summary_empty_results(self, capsys):
        AnalysisResults().summary()
        out = capsys.readouterr().out
        # Section headers still print even when all values are None
        assert "Analysis Results" in out


class TestSimulationParametersSummary:
    def test_summary_prints_all_fields(self, capsys):
        SimulationParameters().summary()
        out = capsys.readouterr().out
        assert "Simulation Parameters" in out
        assert "k_on" in out
        assert "k_sum" in out
        assert "Measurement time" in out

    def test_summary_reflects_custom_values(self, capsys):
        params = SimulationParameters(n_emitters=7)
        params.summary()
        out = capsys.readouterr().out
        assert "7" in out
