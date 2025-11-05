"""
Tests for the simulation module.
"""

import numpy as np
import pytest

from mcoast.simulation import SimulationParameters, TraceGenerator, add_gaussian_noise


class TestSimulationParameters:
    """Tests for SimulationParameters class"""

    def test_initialization(self):
        """Test that parameters initialize correctly"""
        params = SimulationParameters()
        assert params.k_on == 0.15
        assert params.k_off == 0.3
        assert params.n_emitters == 4
        assert params.dt == 0.2
        assert params.measurement_time == 3600
        assert params.single_molecule_intensity == 1.0
        assert params.sigma_noise == 0.2

    def test_valid_parameters(self):
        """Test validation passes with valid parameters"""
        params = SimulationParameters()
        params.k_on = 1.0
        params.k_off = 2.0
        params.n_emitters = 4
        params.dt = 0.1
        params.measurement_time = 100.0
        params.single_molecule_intensity = 40.0
        params.snr = 3.0

        # Should not raise
        params.validate()

    def test_negative_k_on_raises(self):
        """Test that negative k_on raises ValueError"""
        params = SimulationParameters()
        params.k_on = -1.0
        with pytest.raises(ValueError, match="k_on must be positive"):
            params.validate()

    def test_negative_k_off_raises(self):
        """Test that negative k_off raises ValueError"""
        params = SimulationParameters()
        params.k_off = -1.0
        with pytest.raises(ValueError, match="k_off must be positive"):
            params.validate()

    def test_negative_n_emitters_raises(self):
        """Test that negative n_emitters raises ValueError"""
        params = SimulationParameters()
        params.n_emitters = -1
        with pytest.raises(ValueError, match="n_emitters must be positive"):
            params.validate()

    def test_negative_dt_raises(self):
        """Test that negative dt raises ValueError"""
        params = SimulationParameters()
        params.dt = -0.1
        with pytest.raises(ValueError, match="dt must be positive"):
            params.validate()

    def test_negative_measurement_time_raises(self):
        """Test that negative measurement_time raises ValueError"""
        params = SimulationParameters()
        params.measurement_time = -100.0
        with pytest.raises(ValueError, match="measurement_time must be positive"):
            params.validate()

    def test_simple_parameter_access(self):
        """Test basic parameter access and modification"""
        params = SimulationParameters()
        params.k_on = 1.0
        params.k_off = 2.0
        params.n_emitters = 4
        params.dt = 0.1
        params.measurement_time = 100.0
        params.single_molecule_intensity = 40.0
        params.sigma_noise = 5.0

        assert params.k_on == 1.0
        assert params.k_off == 2.0
        assert params.n_emitters == 4
        assert params.dt == 0.1
        assert params.measurement_time == 100.0
        assert params.single_molecule_intensity == 40.0
        assert params.sigma_noise == 5.0


class TestGaussianNoise:
    """Tests for add_gaussian_noise function"""

    def test_add_noise_shape(self):
        """Test that noise preserves signal shape"""
        signal = np.ones(100)
        noisy_signal = add_gaussian_noise(signal, sigma=1.0)

        assert noisy_signal.shape == signal.shape

    def test_add_noise_changes_signal(self):
        """Test that noise actually changes the signal"""
        signal = np.ones(100)
        noisy_signal = add_gaussian_noise(signal, sigma=5.0)

        # With sigma=5, it's extremely unlikely all values remain 1.0
        assert not np.allclose(noisy_signal, signal)

    def test_add_noise_statistics(self):
        """Test that noise has correct statistical properties"""
        np.random.seed(42)
        sigma = 10.0
        signal = np.zeros(10000)  # Zero signal
        noisy_signal = add_gaussian_noise(signal, sigma=sigma)

        # The noise should have mean ~0 and std ~sigma
        assert np.abs(np.mean(noisy_signal)) < 0.5  # Mean close to 0
        assert np.abs(np.std(noisy_signal) - sigma) < 1.0  # Std close to sigma

    def test_zero_noise(self):
        """Test that zero sigma produces no noise"""
        signal = np.ones(100)
        noisy_signal = add_gaussian_noise(signal, sigma=0.0)

        assert np.allclose(noisy_signal, signal)


class TestTraceGenerator:
    """Tests for TraceGenerator class"""

    @pytest.fixture
    def basic_params(self):
        """Create basic valid parameters"""
        params = SimulationParameters()
        params.k_on = 1.0
        params.k_off = 2.0
        params.n_emitters = 4
        params.dt = 0.1
        params.measurement_time = 100.0
        params.single_molecule_intensity = 40.0
        params.sigma_noise = 0.2
        return params

    def test_initialization(self, basic_params):
        """Test generator initialization"""
        generator = TraceGenerator(basic_params)
        assert generator.params == basic_params
        assert generator.noise_sigma is not None

    def test_noise_sigma_from_snr(self, basic_params):
        """Test noise sigma equals sigma_noise parameter"""
        generator = TraceGenerator(basic_params)

        assert generator.noise_sigma == basic_params.sigma_noise

    def test_noise_sigma_from_direct_value(self):
        """Test noise sigma setup from direct sigma"""
        params = SimulationParameters()
        params.k_on = 1.0
        params.k_off = 2.0
        params.n_emitters = 4
        params.dt = 0.1
        params.measurement_time = 100.0
        params.single_molecule_intensity = 40.0
        params.sigma_noise = 5.0

        generator = TraceGenerator(params)
        assert generator.noise_sigma == 5.0

    def test_noise_model_no_noise(self):
        """Test noise model setup with no noise parameters"""
        params = SimulationParameters()
        params.k_on = 1.0
        params.k_off = 2.0
        params.n_emitters = 4
        params.dt = 0.1
        params.measurement_time = 100.0
        params.single_molecule_intensity = 40.0
        params.sigma_noise = 0.0

        generator = TraceGenerator(params)
        assert generator.noise_sigma == 0.0

    def test_generate_trace_shape(self, basic_params):
        """Test that generated trace has correct shape"""
        generator = TraceGenerator(basic_params)
        time_points, intensity = generator.generate_trace()

        expected_n_frames = int(basic_params.measurement_time / basic_params.dt)
        assert len(time_points) == expected_n_frames
        assert len(intensity) == expected_n_frames

    def test_generate_trace_time_points(self, basic_params):
        """Test that time points are correctly spaced"""
        generator = TraceGenerator(basic_params)
        time_points, _ = generator.generate_trace()

        # Check spacing
        dt_values = np.diff(time_points)
        assert np.allclose(dt_values, basic_params.dt)

        # Check start and end times
        assert time_points[0] == 0.0
        assert np.isclose(
            time_points[-1], basic_params.measurement_time - basic_params.dt
        )

    def test_generate_trace_intensity_range(self, basic_params):
        """Test that intensity values are in reasonable range"""
        generator = TraceGenerator(basic_params)
        _, intensity = generator.generate_trace()

        # Intensity should be non-negative (mostly)
        # Allow some negative values due to noise, but not too many
        assert np.sum(intensity < -50) < len(intensity) * 0.1

        # Maximum intensity should be roughly n_emitters * I_single + some noise
        max_expected = (
            basic_params.n_emitters * basic_params.single_molecule_intensity * 2
        )
        assert np.max(intensity) < max_expected

    def test_generate_trace_mean_intensity(self, basic_params):
        """Test that mean intensity is close to theoretical value"""
        np.random.seed(42)
        basic_params.measurement_time = 1000.0  # Longer trace for better statistics

        generator = TraceGenerator(basic_params)
        _, intensity = generator.generate_trace()

        # Theoretical mean: n * I_single * p_up
        k_sum = basic_params.k_on + basic_params.k_off
        p_up = basic_params.k_on / k_sum
        expected_mean = (
            basic_params.n_emitters * basic_params.single_molecule_intensity * p_up
        )

        # Allow 20% tolerance due to randomness
        assert np.abs(np.mean(intensity) - expected_mean) < 0.2 * expected_mean

    def test_single_emitter_trace(self, basic_params):
        """Test single emitter trace generation"""
        basic_params.n_emitters = 1
        basic_params.sigma_noise = 0.0  # No noise for easier testing
        basic_params.measurement_time = 100.0

        generator = TraceGenerator(basic_params)
        n_frames = int(basic_params.measurement_time / basic_params.dt)
        trace = generator._generate_single_emitter_trace(n_frames)

        assert len(trace) == n_frames
        # All values should be either 0 or I_single
        unique_values = np.unique(trace)
        assert all(
            np.isclose(v, 0.0) or np.isclose(v, basic_params.single_molecule_intensity)
            for v in unique_values
        )

    def test_simple_two_state_blinking(self, basic_params):
        """Test the blinking state generation"""
        generator = TraceGenerator(basic_params)
        time_points, states = generator._simple_two_state_blinking(
            basic_params.k_on,
            basic_params.k_off,
            basic_params.dt,
            basic_params.measurement_time,
        )

        expected_n_frames = int(basic_params.measurement_time / basic_params.dt)
        assert len(time_points) == expected_n_frames
        assert len(states) == expected_n_frames

        # States should be binary (0 or 1)
        assert np.all(np.isin(states, [0, 1]))

    def test_simple_two_state_blinking_probability(self, basic_params):
        """Test that blinking states have correct on probability"""
        np.random.seed(42)
        basic_params.measurement_time = 10000.0  # Long trace for statistics

        generator = TraceGenerator(basic_params)
        _, states = generator._simple_two_state_blinking(
            basic_params.k_on,
            basic_params.k_off,
            basic_params.dt,
            basic_params.measurement_time,
        )

        # Theoretical on probability
        k_sum = basic_params.k_on + basic_params.k_off
        p_up_expected = basic_params.k_on / k_sum

        p_up_actual = np.mean(states)

        # Allow 10% tolerance
        assert np.abs(p_up_actual - p_up_expected) < 0.1 * p_up_expected

    def test_generate_ensemble_trace(self, basic_params):
        """Test ensemble trace generation (alias)"""
        generator = TraceGenerator(basic_params)
        time1, intensity1 = generator.generate_trace()
        time2, intensity2 = generator.generate_ensemble_trace()

        # Should have same shape (but different values due to randomness)
        assert len(time1) == len(time2)
        assert len(intensity1) == len(intensity2)

    def test_reproducibility_with_seed(self, basic_params):
        """Test that traces are reproducible with same random seed"""
        np.random.seed(42)
        generator1 = TraceGenerator(basic_params)
        _, intensity1 = generator1.generate_trace()

        np.random.seed(42)
        generator2 = TraceGenerator(basic_params)
        _, intensity2 = generator2.generate_trace()

        assert np.allclose(intensity1, intensity2)

    def test_multiple_emitters_higher_intensity(self, basic_params):
        """Test that more emitters produce higher average intensity"""
        np.random.seed(42)
        basic_params.n_emitters = 1
        basic_params.measurement_time = 1000.0
        basic_params.sigma_noise = 0.0
        generator1 = TraceGenerator(basic_params)
        _, intensity1 = generator1.generate_trace()

        np.random.seed(42)
        basic_params.n_emitters = 4
        generator4 = TraceGenerator(basic_params)
        _, intensity4 = generator4.generate_trace()

        # 4 emitters should have roughly 4x the mean intensity
        assert np.mean(intensity4) > np.mean(intensity1) * 3.0
