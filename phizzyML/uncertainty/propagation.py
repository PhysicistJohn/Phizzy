"""
Uncertainty quantification and propagation for experimental physics data.
"""

import torch
import torch.nn as nn
from typing import Union, Callable, Optional, Tuple
import numpy as np


class UncertainTensor:
    """
    Tensor with associated uncertainties that propagate through operations.
    Supports both systematic and statistical uncertainties.
    """
    
    def __init__(self, value: torch.Tensor, uncertainty: torch.Tensor, 
                 systematic: Optional[torch.Tensor] = None):
        """
        Initialize uncertain tensor.
        
        Args:
            value: Central values
            uncertainty: Statistical uncertainties (standard deviations)
            systematic: Systematic uncertainties (optional)
        """
        self.value = value
        self.uncertainty = uncertainty
        self.systematic = systematic if systematic is not None else torch.zeros_like(value)
        
        # Ensure shapes match
        assert value.shape == uncertainty.shape, "Value and uncertainty must have same shape"
        assert value.shape == self.systematic.shape, "Value and systematic must have same shape"
    
    @property
    def relative_uncertainty(self) -> torch.Tensor:
        """Compute relative uncertainty."""
        return self.uncertainty / (torch.abs(self.value) + 1e-10)
    
    @property
    def total_uncertainty(self) -> torch.Tensor:
        """Combine statistical and systematic uncertainties."""
        return torch.sqrt(self.uncertainty**2 + self.systematic**2)
    
    def __add__(self, other: Union['UncertainTensor', torch.Tensor, float]) -> 'UncertainTensor':
        """Addition with uncertainty propagation."""
        if isinstance(other, UncertainTensor):
            # Error propagation for addition: σ_z = sqrt(σ_x² + σ_y²)
            new_value = self.value + other.value
            new_uncertainty = torch.sqrt(self.uncertainty**2 + other.uncertainty**2)
            new_systematic = torch.sqrt(self.systematic**2 + other.systematic**2)
            return UncertainTensor(new_value, new_uncertainty, new_systematic)
        else:
            # Adding a constant doesn't change uncertainty
            return UncertainTensor(self.value + other, self.uncertainty, self.systematic)
    
    def __sub__(self, other: Union['UncertainTensor', torch.Tensor, float]) -> 'UncertainTensor':
        """Subtraction with uncertainty propagation."""
        if isinstance(other, UncertainTensor):
            new_value = self.value - other.value
            new_uncertainty = torch.sqrt(self.uncertainty**2 + other.uncertainty**2)
            new_systematic = torch.sqrt(self.systematic**2 + other.systematic**2)
            return UncertainTensor(new_value, new_uncertainty, new_systematic)
        else:
            return UncertainTensor(self.value - other, self.uncertainty, self.systematic)
    
    def __mul__(self, other: Union['UncertainTensor', torch.Tensor, float]) -> 'UncertainTensor':
        """Multiplication with uncertainty propagation."""
        if isinstance(other, UncertainTensor):
            # For z = x*y: σ_z/z = sqrt((σ_x/x)² + (σ_y/y)²)
            new_value = self.value * other.value
            rel_unc_self = self.uncertainty / (torch.abs(self.value) + 1e-10)
            rel_unc_other = other.uncertainty / (torch.abs(other.value) + 1e-10)
            rel_unc_total = torch.sqrt(rel_unc_self**2 + rel_unc_other**2)
            new_uncertainty = torch.abs(new_value) * rel_unc_total
            
            # Same for systematic
            rel_sys_self = self.systematic / (torch.abs(self.value) + 1e-10)
            rel_sys_other = other.systematic / (torch.abs(other.value) + 1e-10)
            rel_sys_total = torch.sqrt(rel_sys_self**2 + rel_sys_other**2)
            new_systematic = torch.abs(new_value) * rel_sys_total
            
            return UncertainTensor(new_value, new_uncertainty, new_systematic)
        else:
            # Multiplying by constant scales uncertainty
            other_tensor = torch.tensor(other) if not isinstance(other, torch.Tensor) else other
            return UncertainTensor(
                self.value * other,
                torch.abs(other_tensor) * self.uncertainty,
                torch.abs(other_tensor) * self.systematic
            )
    
    def __truediv__(self, other: Union['UncertainTensor', torch.Tensor, float]) -> 'UncertainTensor':
        """Division with uncertainty propagation."""
        if isinstance(other, UncertainTensor):
            new_value = self.value / other.value
            rel_unc_self = self.uncertainty / (torch.abs(self.value) + 1e-10)
            rel_unc_other = other.uncertainty / (torch.abs(other.value) + 1e-10)
            rel_unc_total = torch.sqrt(rel_unc_self**2 + rel_unc_other**2)
            new_uncertainty = torch.abs(new_value) * rel_unc_total
            
            rel_sys_self = self.systematic / (torch.abs(self.value) + 1e-10)
            rel_sys_other = other.systematic / (torch.abs(other.value) + 1e-10)
            rel_sys_total = torch.sqrt(rel_sys_self**2 + rel_sys_other**2)
            new_systematic = torch.abs(new_value) * rel_sys_total
            
            return UncertainTensor(new_value, new_uncertainty, new_systematic)
        else:
            other_tensor = torch.tensor(other) if not isinstance(other, torch.Tensor) else other
            return UncertainTensor(
                self.value / other,
                self.uncertainty / torch.abs(other_tensor),
                self.systematic / torch.abs(other_tensor)
            )
    
    def __pow__(self, power: float) -> 'UncertainTensor':
        """Power with uncertainty propagation."""
        # For z = x^n: σ_z/z = |n| * σ_x/x
        new_value = self.value ** power
        rel_uncertainty = self.uncertainty / (torch.abs(self.value) + 1e-10)
        new_uncertainty = torch.abs(new_value * power * rel_uncertainty)
        
        rel_systematic = self.systematic / (torch.abs(self.value) + 1e-10)
        new_systematic = torch.abs(new_value * power * rel_systematic)
        
        return UncertainTensor(new_value, new_uncertainty, new_systematic)
    
    def sin(self) -> 'UncertainTensor':
        """Sine with uncertainty propagation."""
        new_value = torch.sin(self.value)
        # d(sin(x))/dx = cos(x)
        derivative = torch.cos(self.value)
        new_uncertainty = torch.abs(derivative) * self.uncertainty
        new_systematic = torch.abs(derivative) * self.systematic
        return UncertainTensor(new_value, new_uncertainty, new_systematic)
    
    def cos(self) -> 'UncertainTensor':
        """Cosine with uncertainty propagation."""
        new_value = torch.cos(self.value)
        # d(cos(x))/dx = -sin(x)
        derivative = -torch.sin(self.value)
        new_uncertainty = torch.abs(derivative) * self.uncertainty
        new_systematic = torch.abs(derivative) * self.systematic
        return UncertainTensor(new_value, new_uncertainty, new_systematic)
    
    def exp(self) -> 'UncertainTensor':
        """Exponential with uncertainty propagation."""
        new_value = torch.exp(self.value)
        # d(exp(x))/dx = exp(x)
        new_uncertainty = new_value * self.uncertainty
        new_systematic = new_value * self.systematic
        return UncertainTensor(new_value, new_uncertainty, new_systematic)
    
    def log(self) -> 'UncertainTensor':
        """Natural logarithm with uncertainty propagation."""
        new_value = torch.log(self.value)
        # d(ln(x))/dx = 1/x
        derivative = 1.0 / self.value
        new_uncertainty = torch.abs(derivative) * self.uncertainty
        new_systematic = torch.abs(derivative) * self.systematic
        return UncertainTensor(new_value, new_uncertainty, new_systematic)
    
    def to_samples(self, n_samples: int = 1000) -> torch.Tensor:
        """Generate Monte Carlo samples from the distribution."""
        # Generate samples from normal distribution
        samples = self.value.unsqueeze(0) + self.total_uncertainty.unsqueeze(0) * torch.randn(
            n_samples, *self.value.shape, device=self.value.device
        )
        return samples
    
    def __repr__(self) -> str:
        return f"UncertainTensor(value={self.value}, uncertainty={self.uncertainty})"


def error_propagation(func: Callable, *args: UncertainTensor, **kwargs) -> UncertainTensor:
    """
    General error propagation using automatic differentiation.
    
    Args:
        func: Function to evaluate
        *args: UncertainTensor arguments
        **kwargs: Additional keyword arguments
    
    Returns:
        UncertainTensor with propagated uncertainties
    """
    # Extract values and create tensors with gradients
    values = []
    uncertainties = []
    
    for arg in args:
        if isinstance(arg, UncertainTensor):
            val = arg.value.requires_grad_(True)
            values.append(val)
            uncertainties.append(arg.uncertainty)
        else:
            values.append(arg)
            uncertainties.append(None)
    
    # Compute function value
    result = func(*values, **kwargs)
    
    # Compute Jacobian for uncertainty propagation
    total_uncertainty_squared = torch.zeros_like(result)
    
    for val, unc in zip(values, uncertainties):
        if unc is not None and val.requires_grad:
            # Compute gradient
            grad = torch.autograd.grad(result.sum(), val, create_graph=True)[0]
            # Add contribution to total uncertainty
            total_uncertainty_squared += (grad * unc) ** 2
    
    total_uncertainty = torch.sqrt(total_uncertainty_squared)
    
    return UncertainTensor(result.detach(), total_uncertainty.detach())


def monte_carlo_propagation(func: Callable, *args: UncertainTensor, 
                          n_samples: int = 10000, **kwargs) -> UncertainTensor:
    """
    Error propagation using Monte Carlo sampling.
    Useful for highly nonlinear functions.
    
    Args:
        func: Function to evaluate
        *args: UncertainTensor arguments
        n_samples: Number of Monte Carlo samples
        **kwargs: Additional keyword arguments
    
    Returns:
        UncertainTensor with propagated uncertainties
    """
    # Generate samples for each uncertain input
    samples_list = []
    for arg in args:
        if isinstance(arg, UncertainTensor):
            samples = arg.to_samples(n_samples)
            samples_list.append(samples)
        else:
            samples_list.append(arg)
    
    # Evaluate function on samples
    results = []
    for i in range(n_samples):
        sample_args = []
        for j, arg in enumerate(args):
            if isinstance(arg, UncertainTensor):
                sample_args.append(samples_list[j][i])
            else:
                sample_args.append(arg)
        
        result = func(*sample_args, **kwargs)
        results.append(result)
    
    # Stack results
    results = torch.stack(results)
    
    # Compute statistics
    mean_value = torch.mean(results, dim=0)
    std_value = torch.std(results, dim=0)
    
    return UncertainTensor(mean_value, std_value)


class UncertaintyAwareLayer(nn.Module):
    """
    Neural network layer that propagates uncertainties.
    """
    
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
    
    def forward(self, x: UncertainTensor) -> UncertainTensor:
        """Forward pass with uncertainty propagation."""
        # Use automatic differentiation for linear layers
        if isinstance(self.layer, nn.Linear):
            # For linear layer: y = Wx + b
            # σ_y = |W| σ_x
            output = self.layer(x.value)
            weight_abs = torch.abs(self.layer.weight)
            uncertainty = torch.matmul(x.uncertainty, weight_abs.t())
            systematic = torch.matmul(x.systematic, weight_abs.t())
            return UncertainTensor(output, uncertainty, systematic)
        else:
            # General case: use error propagation
            return error_propagation(self.layer, x)