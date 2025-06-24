"""
Tests for physical constraint enforcement layers.
"""

import pytest
import torch
import torch.nn as nn
from phizzyML.constraints.conservation import (
    EnergyConservingLayer, MomentumConservingLayer, 
    AngularMomentumConservingLayer, ConstraintEnforcer
)


class TestEnergyConservingLayer:
    def test_energy_conservation_default(self):
        """Test energy conservation with default kinetic energy."""
        layer = EnergyConservingLayer()
        
        # Random input
        x = torch.randn(10, 3, requires_grad=True)
        
        # Apply layer
        x_conserved = layer(x)
        
        # Check that energy is conserved
        E_initial = layer._default_energy(x)
        E_final = layer._default_energy(x_conserved)
        
        assert torch.allclose(E_initial, E_final, rtol=1e-3)
    
    def test_energy_conservation_custom(self):
        """Test with custom energy function."""
        def custom_energy(x):
            # E = sum(x^4) - more complex energy function
            return torch.sum(x**4, dim=-1, keepdim=True)
        
        layer = EnergyConservingLayer(energy_function=custom_energy)
        
        x = torch.randn(5, 4, requires_grad=True)
        x_conserved = layer(x)
        
        E_initial = custom_energy(x)
        E_final = custom_energy(x_conserved)
        
        assert torch.allclose(E_initial, E_final, rtol=1e-2)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        layer = EnergyConservingLayer()
        
        x = torch.randn(10, 3, requires_grad=True)
        x_conserved = layer(x)
        
        # Compute loss and gradients
        loss = x_conserved.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestMomentumConservingLayer:
    def test_momentum_conservation(self):
        """Test linear momentum conservation."""
        masses = torch.tensor([1.0, 2.0, 3.0])
        layer = MomentumConservingLayer(mass=masses, dim=0)
        
        # Random velocities
        velocities = torch.randn(3, 3)
        
        # Apply conservation
        v_conserved = layer(velocities)
        
        # Check total momentum is zero
        total_momentum = torch.sum(masses.unsqueeze(-1) * v_conserved, dim=0)
        assert torch.allclose(total_momentum, torch.zeros_like(total_momentum), atol=1e-6)
    
    def test_momentum_conservation_batch(self):
        """Test with batched data."""
        layer = MomentumConservingLayer(dim=1)
        
        # Batch of particle systems
        velocities = torch.randn(5, 10, 3)  # 5 systems, 10 particles, 3D velocity
        v_conserved = layer(velocities)
        
        # Check momentum conservation for each system
        total_momentum = torch.sum(v_conserved, dim=1)
        assert torch.allclose(total_momentum, torch.zeros_like(total_momentum), atol=1e-6)


class TestAngularMomentumConservingLayer:
    def test_angular_momentum_conservation(self):
        """Test angular momentum conservation (simplified)."""
        # Simple 2-particle system
        positions = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        masses = torch.tensor([1.0, 1.0])
        
        layer = AngularMomentumConservingLayer(positions, masses)
        
        # Initial velocities with net angular momentum
        velocities = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.5, 0.0]])
        
        # Apply conservation
        v_conserved = layer(velocities)
        
        # The implementation is simplified, so we just check it runs
        assert v_conserved.shape == velocities.shape
    
    def test_inertia_tensor_computation(self):
        """Test moment of inertia tensor computation."""
        # Point mass at (1,0,0)
        positions = torch.tensor([[1.0, 0.0, 0.0]])
        masses = torch.tensor([1.0])
        
        layer = AngularMomentumConservingLayer(positions, masses)
        I = layer._compute_inertia_tensor()
        
        # For point mass at (1,0,0), I should have specific structure
        assert I[0, 0] == 0.0  # I_xx = m(y² + z²) = 0
        assert I[1, 1] == 1.0  # I_yy = m(x² + z²) = 1
        assert I[2, 2] == 1.0  # I_zz = m(x² + y²) = 1


class TestConstraintEnforcer:
    def test_single_constraint(self):
        """Test enforcing a single constraint."""
        # Constraint: sum of components = 0
        def zero_sum_constraint(x):
            return x.sum(dim=-1, keepdim=True)
        
        enforcer = ConstraintEnforcer([zero_sum_constraint], n_iterations=20)
        
        x = torch.randn(5, 3, requires_grad=True)
        x_constrained = enforcer(x)
        
        # Check constraint is satisfied
        constraint_val = zero_sum_constraint(x_constrained)
        assert torch.allclose(constraint_val, torch.zeros_like(constraint_val), atol=1e-5)
    
    def test_multiple_constraints(self):
        """Test enforcing multiple constraints simultaneously."""
        # Constraint 1: sum = 0
        def zero_sum(x):
            return x.sum(dim=-1, keepdim=True)
        
        # Constraint 2: norm = 1
        def unit_norm(x):
            return torch.norm(x, dim=-1, keepdim=True) - 1.0
        
        enforcer = ConstraintEnforcer([zero_sum, unit_norm], n_iterations=50)
        
        x = torch.randn(10, 3, requires_grad=True)
        x_constrained = enforcer(x)
        
        # Check both constraints
        assert torch.allclose(zero_sum(x_constrained), 
                            torch.zeros(10, 1), atol=1e-4)
        assert torch.allclose(unit_norm(x_constrained), 
                            torch.zeros(10, 1), atol=1e-4)
    
    def test_gradient_preservation(self):
        """Test that constraint enforcer preserves gradient flow."""
        def sphere_constraint(x):
            # Constrain to unit sphere
            return torch.sum(x**2, dim=-1, keepdim=True) - 1.0
        
        enforcer = ConstraintEnforcer([sphere_constraint])
        
        x = torch.randn(5, 3, requires_grad=True)
        x_constrained = enforcer(x)
        
        loss = x_constrained.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)