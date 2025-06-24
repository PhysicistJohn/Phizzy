import pytest
import numpy as np
from phizzy.integrators import symplectic_integrate


class TestSymplecticIntegrate:
    
    def test_harmonic_oscillator_energy_conservation(self):
        """Test energy conservation for simple harmonic oscillator"""
        def hamiltonian(q, p):
            k, m = 1.0, 1.0
            return 0.5 * p**2 / m + 0.5 * k * q**2
        
        q0, p0 = np.array([1.0]), np.array([0.0])
        t_span = (0, 10 * 2 * np.pi)  # 10 periods
        
        result = symplectic_integrate(
            hamiltonian, q0, p0, t_span, dt=0.01, method='leapfrog'
        )
        
        # Check energy conservation
        assert result.energy_error < 1e-6
        assert np.std(result.energy) / np.mean(result.energy) < 1e-6
        
    def test_kepler_problem(self):
        """Test Kepler problem (planetary motion)"""
        def hamiltonian(q, p):
            # 2D Kepler problem: H = |p|^2/2 - 1/|q|
            r = np.sqrt(q[0]**2 + q[1]**2)
            return 0.5 * (p[0]**2 + p[1]**2) - 1.0 / r
        
        # Circular orbit initial conditions
        q0 = np.array([1.0, 0.0])
        p0 = np.array([0.0, 1.0])
        t_span = (0, 10 * 2 * np.pi)
        
        result = symplectic_integrate(
            hamiltonian, q0, p0, t_span, dt=0.01, method='yoshida4'
        )
        
        # Check energy conservation (better for higher-order method)
        assert result.energy_error < 1e-10
        
        # Check angular momentum conservation
        L = result.q[:, 0] * result.p[:, 1] - result.q[:, 1] * result.p[:, 0]
        assert np.std(L) / np.mean(L) < 1e-10
        
    def test_double_pendulum(self):
        """Test double pendulum system"""
        def hamiltonian(q, p):
            # q = [theta1, theta2], p = [p_theta1, p_theta2]
            m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81
            
            c = np.cos(q[0] - q[1])
            s = np.sin(q[0] - q[1])
            den = m1 + m2 * s**2
            
            # Kinetic energy
            T = (p[0]**2 * m2 + p[1]**2 * (m1 + m2) - 
                 2 * p[0] * p[1] * m2 * c) / (2 * m2 * l1**2 * l2**2 * den)
            
            # Potential energy
            V = -(m1 + m2) * g * l1 * np.cos(q[0]) - m2 * g * l2 * np.cos(q[1])
            
            return T + V
        
        q0 = np.array([np.pi/4, -np.pi/4])
        p0 = np.array([0.0, 0.0])
        t_span = (0, 10.0)
        
        result = symplectic_integrate(
            hamiltonian, q0, p0, t_span, dt=0.001, method='forest-ruth'
        )
        
        # For chaotic systems, just check that energy is reasonably conserved
        assert result.energy_error < 1e-8
        
    def test_different_methods(self):
        """Test all available integration methods"""
        def hamiltonian(q, p):
            return 0.5 * (p**2 + q**2)
        
        q0, p0 = np.array([1.0]), np.array([0.0])
        t_span = (0, 2 * np.pi)
        
        methods = ['leapfrog', 'yoshida4', 'forest-ruth', 'pefrl']
        energy_errors = []
        
        for method in methods:
            result = symplectic_integrate(
                hamiltonian, q0, p0, t_span, dt=0.1, method=method
            )
            energy_errors.append(result.energy_error)
        
        # Higher-order methods should have better energy conservation
        assert energy_errors[0] > energy_errors[1]  # leapfrog > yoshida4
        assert energy_errors[0] > energy_errors[2]  # leapfrog > forest-ruth
        assert energy_errors[0] > energy_errors[3]  # leapfrog > pefrl
        
    def test_callback_function(self):
        """Test callback functionality"""
        callback_data = []
        
        def callback(t, q, p):
            callback_data.append((t, q.copy(), p.copy()))
        
        def hamiltonian(q, p):
            return 0.5 * (p**2 + q**2)
        
        q0, p0 = np.array([1.0]), np.array([0.0])
        t_span = (0, 1.0)
        
        result = symplectic_integrate(
            hamiltonian, q0, p0, t_span, dt=0.1, callback=callback
        )
        
        assert len(callback_data) == 10  # Should be called at each step
        assert callback_data[-1][0] == 1.0  # Final time
        
    def test_phase_space_volume_preservation(self):
        """Test Liouville's theorem - phase space volume preservation"""
        def hamiltonian(q, p):
            return 0.5 * (p[0]**2 + p[1]**2 + q[0]**2 + q[1]**2)
        
        # Initial conditions in a small box
        n_particles = 10
        q0 = np.random.uniform(-0.1, 0.1, (n_particles, 2))
        p0 = np.random.uniform(-0.1, 0.1, (n_particles, 2))
        
        results = []
        for i in range(n_particles):
            result = symplectic_integrate(
                hamiltonian, q0[i], p0[i], (0, 2*np.pi), dt=0.01
            )
            results.append(result)
        
        # Compute phase space volume at different times
        initial_volume = np.std(q0) * np.std(p0)
        final_q = np.array([r.q[-1] for r in results])
        final_p = np.array([r.p[-1] for r in results])
        final_volume = np.std(final_q) * np.std(final_p)
        
        # Volume should be approximately preserved
        assert abs(final_volume - initial_volume) / initial_volume < 0.1
        
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        def hamiltonian(q, p):
            return 0.5 * (p**2 + q**2)
        
        # Mismatched dimensions
        with pytest.raises(ValueError):
            symplectic_integrate(
                hamiltonian, 
                np.array([1.0, 2.0]), 
                np.array([1.0]),  # Wrong size
                (0, 1)
            )
        
        # Invalid method
        with pytest.raises(ValueError):
            symplectic_integrate(
                hamiltonian, 
                np.array([1.0]), 
                np.array([1.0]),
                (0, 1),
                method='invalid_method'
            )
    
    def test_high_dimensional_system(self):
        """Test with high-dimensional phase space"""
        dim = 10
        
        def hamiltonian(q, p):
            # Coupled harmonic oscillators
            K = np.eye(dim) + 0.1 * np.ones((dim, dim))
            return 0.5 * np.dot(p, p) + 0.5 * np.dot(q, np.dot(K, q))
        
        q0 = np.random.randn(dim)
        p0 = np.random.randn(dim)
        
        result = symplectic_integrate(
            hamiltonian, q0, p0, (0, 10.0), dt=0.01, method='pefrl'
        )
        
        # Energy should still be conserved in high dimensions
        assert result.energy_error < 1e-10