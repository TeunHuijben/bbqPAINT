"""
Tests for the statistics module (blocking helpers used for visualization).
"""

import numpy as np

from bbqpaint.utils.statistics import block_bispectrum, block_power_spectrum


class TestBlockPowerSpectrum:
    """Test cases for block_power_spectrum"""

    def test_returns_three_arrays(self):
        """Blocking returns (x, y, err) arrays of equal length"""
        x = np.arange(1, 1001, dtype=float)
        y = np.random.RandomState(0).rand(1000)

        blocked_x, blocked_y, blocked_err = block_power_spectrum(x, y)

        assert blocked_x.shape == blocked_y.shape == blocked_err.shape
        assert len(blocked_x) > 0

    def test_output_is_finite(self):
        """All blocked values are finite"""
        x = np.linspace(0.1, 10, 500)
        y = 1.0 / x  # decaying spectrum-like data

        blocked_x, blocked_y, blocked_err = block_power_spectrum(x, y)

        assert np.all(np.isfinite(blocked_x))
        assert np.all(np.isfinite(blocked_y))
        assert np.all(np.isfinite(blocked_err))

    def test_reduces_number_of_points(self):
        """Blocking compresses the data into fewer points"""
        x = np.arange(1, 2001, dtype=float)
        y = np.ones(2000)

        blocked_x, _, _ = block_power_spectrum(x, y)

        assert len(blocked_x) < len(x)

    def test_growth_factor_affects_block_count(self):
        """A larger growth factor yields fewer blocks"""
        x = np.arange(1, 2001, dtype=float)
        y = np.ones(2000)

        small_growth, _, _ = block_power_spectrum(x, y, growth_factor=1.1)
        large_growth, _, _ = block_power_spectrum(x, y, growth_factor=2.0)

        assert len(large_growth) <= len(small_growth)

    def test_short_data_with_large_growth(self):
        """Blocking works on short data with a large growth factor"""
        x = np.arange(1, 101, dtype=float)
        y = np.random.RandomState(5).rand(100)

        blocked_x, _, _ = block_power_spectrum(x, y, growth_factor=1.5)

        assert len(blocked_x) > 0
        assert np.all(np.isfinite(blocked_x))

    def test_errors_non_negative(self):
        """Blocked standard errors are non-negative"""
        x = np.arange(1, 1001, dtype=float)
        y = np.random.RandomState(1).rand(1000)

        _, _, blocked_err = block_power_spectrum(x, y)

        assert np.all(blocked_err >= 0)


class TestBlockBispectrum:
    """Test cases for block_bispectrum"""

    def test_returns_three_matrices(self):
        """Blocking returns (bs, k1, k2) matrices of equal shape"""
        bs = np.random.RandomState(2).rand(150, 150)
        k1 = np.tile(np.arange(150), (150, 1)).astype(float)
        k2 = k1.T.copy()

        blocked_bs, blocked_k1, blocked_k2 = block_bispectrum(bs, k1, k2)

        assert blocked_bs.shape == blocked_k1.shape == blocked_k2.shape

    def test_reduces_resolution(self):
        """Blocking reduces matrix size when input is large"""
        bs = np.ones((300, 300))
        k1 = np.tile(np.arange(300), (300, 1)).astype(float)
        k2 = k1.T.copy()

        blocked_bs, _, _ = block_bispectrum(bs, k1, k2, num_blocks_goal=75)

        assert blocked_bs.shape[0] < bs.shape[0]
        assert blocked_bs.shape[1] < bs.shape[1]

    def test_small_input_block_size_one(self):
        """When the matrix is smaller than the goal, block size stays 1"""
        bs = np.arange(100, dtype=float).reshape(10, 10)
        k1 = np.tile(np.arange(10), (10, 1)).astype(float)
        k2 = k1.T.copy()

        blocked_bs, _, _ = block_bispectrum(bs, k1, k2, num_blocks_goal=75)

        # block_size collapses to 1, so shape is preserved
        assert blocked_bs.shape == bs.shape

    def test_handles_nan_values(self):
        """A NaN inside a multi-element block is ignored via nanmean"""
        # 300x300 with goal 75 -> block size 3, so each block averages 9 points
        bs = np.ones((300, 300))
        bs[1, 1] = np.nan  # single NaN within the first block
        k1 = np.tile(np.arange(300), (300, 1)).astype(float)
        k2 = k1.T.copy()

        blocked_bs, _, _ = block_bispectrum(bs, k1, k2, num_blocks_goal=75)

        assert np.all(np.isfinite(blocked_bs))
