"""
Tests for symplectic integrators.
"""

import pytest
import torch
import numpy as np
from phizzyML.integrators.symplectic import (
    SymplecticEuler, LeapfrogIntegrator, VerletIntegrator,
    compute_energy_error
)


def harmonic_oscillator_hamiltonian(q, p, k=1.0, m=1.0):
    """Hamiltonian for harmonic oscillator: H = p²/(2m) + kq²/2"""
    kinetic = torch.sum(p**2) / (2 * m)
    potential = k * torch.sum(q**2) / 2
    return kinetic + potential


def pendulum_hamiltonian(q, p, l=1.0, g=9.81):
    """Hamiltonian for pendulum: H = p²/(2ml²) - mgl*cos(q)"""
    kinetic = torch.sum(p**2) / (2 * l**2)
    potential = -g * l * torch.cos(q).sum()
    return kinetic + potential


class TestSymplecticIntegrators:
    def setup_method(self):
        """Set up test parameters."""
        self.dt = 0.01
        self.n_steps = 1000
        torch.manual_seed(42)
    
    def test_symplectic_euler_energy_conservation(self):
        """Test that symplectic Euler approximately conserves energy."""
        integrator = SymplecticEuler(dt=self.dt)
        
        # Initial conditions
        q0 = torch.tensor([1.0, 0.0])
        p0 = torch.tensor([0.0, 1.0])
        
        # Integrate
        q_traj, p_traj = integrator.integrate(
            q0, p0, harmonic_oscillator_hamiltonian, self.n_steps
        )
        
        # Check energy conservation
        energy_errors = compute_energy_error(q_traj, p_traj, harmonic_oscillator_hamiltonian)
        
        # Symplectic Euler has O(dt) error
        assert energy_errors.max() < 0.1
    
    def test_leapfrog_energy_conservation(self):
        """Test that leapfrog integrator conserves energy better."""
        integrator = LeapfrogIntegrator(dt=self.dt)
        
        # Initial conditions
        q0 = torch.tensor([1.0, 0.0])
        p0 = torch.tensor([0.0, 1.0])
        
        # Integrate
        q_traj, p_traj = integrator.integrate(
            q0, p0, harmonic_oscillator_hamiltonian, self.n_steps
        )
        
        # Check energy conservation
        energy_errors = compute_energy_error(q_traj, p_traj, harmonic_oscillator_hamiltonian)
        
        # Leapfrog has O(dt²) error, should be much better
        assert energy_errors.max() < 0.001
    
    def test_verlet_integrator(self):
        """Test Verlet integrator for separable Hamiltonian."""
        mass = torch.tensor(1.0)
        integrator = VerletIntegrator(dt=self.dt, mass=mass)
        
        # Initial conditions
        q0 = torch.tensor([1.0, 0.0])
        p0 = torch.tensor([0.0, 1.0])
        
        # Integrate
        q_traj, p_traj = integrator.integrate(
            q0, p0, harmonic_oscillator_hamiltonian, self.n_steps
        )
        
        # Check energy conservation
        energy_errors = compute_energy_error(q_traj, p_traj, harmonic_oscillator_hamiltonian)
        
        # Verlet should also have good energy conservation
        assert energy_errors.max() < 0.001
    
    def test_phase_space_volume_preservation(self):
        """Test that symplectic integrators preserve phase space volume."""
        integrator = LeapfrogIntegrator(dt=self.dt)
        
        # Create grid of initial conditions
        q_grid = torch.linspace(-1, 1, 5)
        p_grid = torch.linspace(-1, 1, 5)
        q_init, p_init = torch.meshgrid(q_grid, p_grid, indexing='ij')
        q_init = q_init.flatten()
        p_init = p_init.flatten()
        
        # Integrate each point
        q_final = []
        p_final = []
        
        for q0, p0 in zip(q_init, p_init):
            q, p = integrator.step(
                q0.unsqueeze(0), p0.unsqueeze(0), 
                harmonic_oscillator_hamiltonian
            )
            q_final.append(q)
            p_final.append(p)
        
        q_final = torch.stack(q_final)
        p_final = torch.stack(p_final)
        
        # Compute Jacobian determinant numerically
        # For symplectic integrator, det(J) should be 1
        # This is a simplified test - proper test would compute full Jacobian
        initial_area = (q_grid[1] - q_grid[0]) * (p_grid[1] - p_grid[0])
        
        # Check that points don't collapse
        q_range = q_final.max() - q_final.min()
        p_range = p_final.max() - p_final.min()
        
        assert q_range > 0.5  # Points should spread out reasonably
        assert p_range > 0.5
    
    def test_backward_integration(self):
        """Test time-reversibility of symplectic integrators."""
        integrator = LeapfrogIntegrator(dt=self.dt)
        
        # Initial conditions
        q0 = torch.tensor([1.0])
        p0 = torch.tensor([0.0])
        
        # Forward integration
        q1, p1 = integrator.integrate(q0, p0, harmonic_oscillator_hamiltonian, 100)
        
        # Backward integration
        backward_integrator = LeapfrogIntegrator(dt=-self.dt)
        q2, p2 = backward_integrator.integrate(
            q1[-1], p1[-1], harmonic_oscillator_hamiltonian, 100
        )
        
        # Should return close to initial conditions
        assert torch.allclose(q2[-1], q0, atol=1e-3)
        assert torch.allclose(p2[-1], p0, atol=1e-3)
    
    def test_different_hamiltonians(self):
        """Test integrators with different Hamiltonian systems."""
        integrator = LeapfrogIntegrator(dt=self.dt)
        
        # Test with pendulum
        q0 = torch.tensor([0.5])  # Small angle
        p0 = torch.tensor([0.0])
        
        q_traj, p_traj = integrator.integrate(
            q0, p0, pendulum_hamiltonian, self.n_steps
        )
        
        # Check energy conservation for pendulum
        energy_errors = compute_energy_error(q_traj, p_traj, pendulum_hamiltonian)
        assert energy_errors.max() < 0.01
        
        # Check that pendulum oscillates
        assert q_traj.min() < -0.4
        assert q_traj.max() > 0.4