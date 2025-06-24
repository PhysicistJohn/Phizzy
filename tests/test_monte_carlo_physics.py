import pytest
import numpy as np
from phizzy.monte_carlo import monte_carlo_physics


class TestMonteCarloPhysics:
    
    def test_ising_model_basic(self):
        """Test basic Ising model simulation"""
        result = monte_carlo_physics(
            'ising',
            n_samples=1000,
            temperature=2.0,
            parameters={'L': 10, 'J': 1.0},
            equilibration=100,
            random_seed=42
        )
        
        assert result.samples.shape == (1000, 10, 10)
        assert result.energies is not None
        assert result.magnetization is not None
        assert 0 < result.acceptance_rate < 1
        assert result.autocorrelation_time > 0
        assert 'mean_energy' in result.statistics
        assert 'specific_heat' in result.statistics
    
    def test_ising_phase_transition(self):
        """Test Ising model at different temperatures"""
        # Low temperature (ordered)
        result_low = monte_carlo_physics(
            'ising',
            n_samples=500,
            temperature=1.0,
            parameters={'L': 16, 'J': 1.0},
            equilibration=200,
            random_seed=42
        )
        
        # High temperature (disordered)
        result_high = monte_carlo_physics(
            'ising',
            n_samples=500,
            temperature=5.0,
            parameters={'L': 16, 'J': 1.0},
            equilibration=200,
            random_seed=42
        )
        
        # Magnetization should be higher at low temperature
        assert abs(result_low.statistics['mean_magnetization']) > abs(result_high.statistics['mean_magnetization'])
        # Acceptance rate should be lower at low temperature
        assert result_low.acceptance_rate < result_high.acceptance_rate
    
    def test_custom_energy_function(self):
        """Test with custom energy function"""
        def harmonic_energy(state, k=1.0, **kwargs):
            """Simple harmonic oscillator ensemble"""
            return 0.5 * k * np.sum(state**2)
        
        result = monte_carlo_physics(
            harmonic_energy,
            n_samples=1000,
            temperature=1.0,
            parameters={'dim': 10, 'k': 2.0},
            equilibration=100,
            random_seed=42
        )
        
        assert result.samples.shape == (1000, 10)
        # Mean energy should be approximately kT * dim / 2
        expected_energy = 1.0 * 10 / 2  # temperature * dim / 2
        assert abs(result.statistics['mean_energy'] - expected_energy) < 1.0
    
    def test_observables(self):
        """Test custom observables"""
        def radius_squared(state):
            return np.sum(state**2)
        
        def max_value(state):
            return np.max(np.abs(state))
        
        observables = {
            'r_squared': radius_squared,
            'max_val': max_value
        }
        
        result = monte_carlo_physics(
            'ising',
            n_samples=500,
            temperature=2.269,  # Critical temperature
            parameters={'L': 8},
            observables=observables,
            equilibration=100
        )
        
        assert 'r_squared' in result.observables
        assert 'max_val' in result.observables
        assert len(result.observables['r_squared']) == 500
        assert result.mean('r_squared') > 0
    
    def test_cluster_algorithm(self):
        """Test Wolff cluster algorithm"""
        result = monte_carlo_physics(
            'ising',
            n_samples=500,
            temperature=2.269,
            parameters={'L': 16},
            method='wolff',
            cluster_updates=True,
            equilibration=50,
            random_seed=42
        )
        
        # Cluster algorithm should have 100% acceptance
        assert result.acceptance_rate == 1.0
        assert result.samples.shape == (500, 16, 16)
    
    def test_hybrid_algorithm(self):
        """Test hybrid algorithm"""
        result = monte_carlo_physics(
            'ising',
            n_samples=500,
            temperature=2.269,
            parameters={'L': 12},
            method='hybrid',
            cluster_updates=True,
            equilibration=50
        )
        
        assert result.samples.shape == (500, 12, 12)
        # Hybrid should have intermediate acceptance rate
        assert 0.5 < result.acceptance_rate <= 1.0
    
    def test_parallel_tempering(self):
        """Test parallel tempering"""
        temperatures = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        result = monte_carlo_physics(
            'ising',
            n_samples=300,
            temperature=1.0,  # Base temperature
            parameters={'L': 8},
            parallel_tempering=temperatures,
            equilibration=50
        )
        
        assert result.samples.shape == (300, 8, 8)
        assert result.statistics['temperatures'] == temperatures
    
    def test_sampling_interval(self):
        """Test sampling interval option"""
        result = monte_carlo_physics(
            'ising',
            n_samples=100,
            temperature=2.0,
            parameters={'L': 8},
            sampling_interval=10,  # Sample every 10 steps
            equilibration=50
        )
        
        assert result.samples.shape == (100, 8, 8)
        # Should have done 100 * 10 = 1000 total steps
        assert result.acceptance_rate > 0
    
    def test_convergence_check(self):
        """Test convergence checking"""
        # Too few samples - should warn
        with pytest.warns(UserWarning, match="converged"):
            result = monte_carlo_physics(
                'ising',
                n_samples=50,  # Very few samples
                temperature=2.269,
                parameters={'L': 16},
                equilibration=10,
                convergence_check=True
            )
        
        assert 'converged' in result.statistics
        assert result.statistics['converged'] == False
    
    def test_polymer_system(self):
        """Test predefined polymer system"""
        result = monte_carlo_physics(
            'polymer',
            n_samples=500,
            temperature=1.0,
            parameters={'dim': 20, 'L': 1.0, 'k': 10.0},
            equilibration=100
        )
        
        assert result.samples.shape[0] == 500
        assert result.energies is not None
    
    def test_initial_state(self):
        """Test starting from specific initial state"""
        L = 10
        initial = np.ones((L, L))  # All spins up
        
        result = monte_carlo_physics(
            'ising',
            n_samples=100,
            temperature=1.0,
            parameters={'L': L},
            initial_state=initial,
            equilibration=0  # No equilibration
        )
        
        # First sample should be close to initial (low temperature)
        assert np.mean(result.samples[0]) > 0.9
    
    def test_result_methods(self):
        """Test MonteCarloResult methods"""
        result = monte_carlo_physics(
            'ising',
            n_samples=1000,
            temperature=2.0,
            parameters={'L': 8},
            equilibration=100
        )
        
        # Test mean and std methods
        mean_energy = result.mean()  # Mean of samples
        assert isinstance(mean_energy, np.ndarray)
        
        std_energy = result.std()
        assert isinstance(std_energy, np.ndarray)
        
        # Test error estimation
        error = result.error()
        assert error > 0
        assert error < result.std().mean()  # Error should be less than std due to autocorrelation
    
    def test_external_field(self):
        """Test Ising model with external field"""
        result = monte_carlo_physics(
            'ising',
            n_samples=500,
            temperature=2.0,
            parameters={'L': 10, 'J': 1.0, 'h': 0.5},
            equilibration=100
        )
        
        # With positive field, magnetization should be positive
        assert result.statistics['mean_magnetization'] > 0
    
    def test_error_handling(self):
        """Test error handling"""
        # Unknown system
        with pytest.raises(ValueError):
            monte_carlo_physics('unknown_system')
        
        # Invalid method (will fall back to metropolis)
        result = monte_carlo_physics(
            'ising',
            n_samples=100,
            method='invalid_method',
            parameters={'L': 5}
        )
        assert result.samples.shape == (100, 5, 5)