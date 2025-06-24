"""
Tests for uncertainty quantification and propagation.
"""

import pytest
import torch
import numpy as np
from phizzyML.uncertainty.propagation import (
    UncertainTensor, error_propagation, monte_carlo_propagation,
    UncertaintyAwareLayer
)


class TestUncertainTensor:
    def test_creation(self):
        """Test UncertainTensor creation."""
        value = torch.tensor([1.0, 2.0, 3.0])
        uncertainty = torch.tensor([0.1, 0.2, 0.3])
        
        ut = UncertainTensor(value, uncertainty)
        
        assert torch.allclose(ut.value, value)
        assert torch.allclose(ut.uncertainty, uncertainty)
        assert torch.allclose(ut.systematic, torch.zeros_like(value))
    
    def test_addition(self):
        """Test uncertainty propagation in addition."""
        ut1 = UncertainTensor(torch.tensor(5.0), torch.tensor(0.3))
        ut2 = UncertainTensor(torch.tensor(3.0), torch.tensor(0.4))
        
        result = ut1 + ut2
        
        assert result.value == pytest.approx(8.0)
        # σ_total = sqrt(0.3² + 0.4²) = 0.5
        assert result.uncertainty == pytest.approx(0.5, rel=1e-3)
    
    def test_multiplication(self):
        """Test uncertainty propagation in multiplication."""
        ut1 = UncertainTensor(torch.tensor(10.0), torch.tensor(0.5))
        ut2 = UncertainTensor(torch.tensor(2.0), torch.tensor(0.1))
        
        result = ut1 * ut2
        
        assert result.value == pytest.approx(20.0)
        # Relative uncertainties: 0.5/10 = 0.05, 0.1/2 = 0.05
        # Combined relative: sqrt(0.05² + 0.05²) ≈ 0.0707
        # Absolute: 20 * 0.0707 ≈ 1.414
        assert result.uncertainty == pytest.approx(1.414, rel=1e-2)
    
    def test_division(self):
        """Test uncertainty propagation in division."""
        ut1 = UncertainTensor(torch.tensor(10.0), torch.tensor(0.5))
        ut2 = UncertainTensor(torch.tensor(2.0), torch.tensor(0.1))
        
        result = ut1 / ut2
        
        assert result.value == pytest.approx(5.0)
        # Same relative uncertainty calculation as multiplication
        assert result.uncertainty == pytest.approx(0.354, rel=1e-2)
    
    def test_power(self):
        """Test uncertainty propagation in power operations."""
        ut = UncertainTensor(torch.tensor(2.0), torch.tensor(0.1))
        
        result = ut ** 2
        
        assert result.value == pytest.approx(4.0)
        # σ_z/z = 2 * σ_x/x = 2 * 0.1/2 = 0.1
        # σ_z = 4 * 0.1 = 0.4
        assert result.uncertainty == pytest.approx(0.4, rel=1e-3)
    
    def test_trigonometric(self):
        """Test uncertainty propagation in trig functions."""
        ut = UncertainTensor(torch.tensor(0.0), torch.tensor(0.1))
        
        # sin(0) = 0, d(sin)/dx at 0 = cos(0) = 1
        sin_result = ut.sin()
        assert sin_result.value == pytest.approx(0.0)
        assert sin_result.uncertainty == pytest.approx(0.1)
        
        # cos(0) = 1, d(cos)/dx at 0 = -sin(0) = 0
        cos_result = ut.cos()
        assert cos_result.value == pytest.approx(1.0)
        assert cos_result.uncertainty == pytest.approx(0.0, abs=1e-6)
    
    def test_exponential_logarithm(self):
        """Test uncertainty propagation in exp and log."""
        # Exponential
        ut = UncertainTensor(torch.tensor(1.0), torch.tensor(0.1))
        exp_result = ut.exp()
        
        assert exp_result.value == pytest.approx(np.e)
        assert exp_result.uncertainty == pytest.approx(np.e * 0.1)
        
        # Logarithm
        ut2 = UncertainTensor(torch.tensor(np.e), torch.tensor(0.1))
        log_result = ut2.log()
        
        assert log_result.value == pytest.approx(1.0)
        assert log_result.uncertainty == pytest.approx(0.1 / np.e)
    
    def test_systematic_uncertainty(self):
        """Test handling of systematic uncertainties."""
        value = torch.tensor(10.0)
        stat_unc = torch.tensor(0.5)
        sys_unc = torch.tensor(0.3)
        
        ut = UncertainTensor(value, stat_unc, sys_unc)
        
        # Total uncertainty: sqrt(0.5² + 0.3²) ≈ 0.583
        assert ut.total_uncertainty == pytest.approx(0.583, rel=1e-3)
    
    def test_to_samples(self):
        """Test Monte Carlo sample generation."""
        ut = UncertainTensor(torch.tensor(5.0), torch.tensor(0.5))
        samples = ut.to_samples(n_samples=10000)
        
        assert samples.shape == (10000,)
        assert samples.mean() == pytest.approx(5.0, abs=0.1)
        assert samples.std() == pytest.approx(0.5, abs=0.1)


class TestErrorPropagation:
    def test_linear_function(self):
        """Test error propagation through linear function."""
        def linear_func(x, y):
            return 2 * x + 3 * y
        
        x = UncertainTensor(torch.tensor(1.0), torch.tensor(0.1))
        y = UncertainTensor(torch.tensor(2.0), torch.tensor(0.2))
        
        result = error_propagation(linear_func, x, y)
        
        assert result.value == pytest.approx(8.0)
        # σ_f = sqrt((2*0.1)² + (3*0.2)²) = sqrt(0.04 + 0.36) ≈ 0.632
        assert result.uncertainty == pytest.approx(0.632, rel=1e-2)
    
    def test_nonlinear_function(self):
        """Test error propagation through nonlinear function."""
        def nonlinear_func(x):
            return x ** 2 + torch.sin(x)
        
        x = UncertainTensor(torch.tensor(1.0), torch.tensor(0.1))
        result = error_propagation(nonlinear_func, x)
        
        # f(1) = 1 + sin(1) ≈ 1.841
        assert result.value == pytest.approx(1.841, rel=1e-3)
        
        # df/dx at x=1 = 2*1 + cos(1) ≈ 2.540
        # σ_f ≈ 2.540 * 0.1 ≈ 0.254
        assert result.uncertainty == pytest.approx(0.254, rel=1e-2)


class TestMonteCarloPropagatino:
    def test_monte_carlo_linear(self):
        """Test Monte Carlo propagation for linear function."""
        def linear_func(x, y):
            return x + y
        
        x = UncertainTensor(torch.tensor(1.0), torch.tensor(0.1))
        y = UncertainTensor(torch.tensor(2.0), torch.tensor(0.2))
        
        result = monte_carlo_propagation(linear_func, x, y, n_samples=10000)
        
        assert result.value == pytest.approx(3.0, abs=0.05)
        # σ_total = sqrt(0.1² + 0.2²) ≈ 0.224
        assert result.uncertainty == pytest.approx(0.224, abs=0.05)
    
    def test_monte_carlo_nonlinear(self):
        """Test Monte Carlo for highly nonlinear function."""
        def complex_func(x):
            return torch.exp(-x**2) * torch.sin(5*x)
        
        x = UncertainTensor(torch.tensor(0.5), torch.tensor(0.1))
        
        result = monte_carlo_propagation(complex_func, x, n_samples=10000)
        
        # Just check it runs and gives reasonable results
        assert result.value.shape == torch.Size([])
        assert result.uncertainty > 0


class TestUncertaintyAwareLayer:
    def test_linear_layer_propagation(self):
        """Test uncertainty propagation through linear layer."""
        layer = torch.nn.Linear(3, 2)
        layer.weight.data = torch.tensor([[1.0, 2.0, 3.0], 
                                         [0.5, 1.0, 1.5]])
        layer.bias.data = torch.tensor([0.1, 0.2])
        
        uncertain_layer = UncertaintyAwareLayer(layer)
        
        x = UncertainTensor(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([0.1, 0.2, 0.3])
        )
        
        result = uncertain_layer(x)
        
        # Check output values
        expected_value = layer(x.value)
        assert torch.allclose(result.value, expected_value)
        
        # Check uncertainty propagation
        # For linear layer, uncertainties combine based on weights
        assert result.uncertainty.shape == (2,)
        assert torch.all(result.uncertainty > 0)