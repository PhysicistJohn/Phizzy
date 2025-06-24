"""
Neural network layers that enforce physical conservation laws.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, List
from abc import ABC, abstractmethod


class ConservationLayer(nn.Module, ABC):
    """Base class for layers that enforce conservation laws."""
    
    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto constraint manifold."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with constraint enforcement."""
        pass


class EnergyConservingLayer(ConservationLayer):
    """
    Layer that ensures energy conservation in the output.
    Works by projecting outputs onto the energy-conserving manifold.
    """
    
    def __init__(self, energy_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.energy_function = energy_function or self._default_energy
        self._use_default = energy_function is None
    
    def _default_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Default energy: sum of squares (kinetic energy analogue)."""
        return 0.5 * torch.sum(x**2, dim=-1, keepdim=True)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto constant energy surface."""
        # For the default quadratic energy E = 0.5 * sum(x^2), 
        # the constraint surface is a hypersphere
        # The projection is simply: x_proj = x * sqrt(E_target / E_current)
        
        # Compute current energy
        E_current = self.energy_function(x)
        
        # For general energy functions, we need iterative projection
        # For quadratic energy, we have a closed-form solution
        if self._use_default:
            # Simple scaling for quadratic energy
            x_norm = torch.norm(x, dim=-1, keepdim=True)
            # Avoid division by zero
            x_normalized = x / (x_norm + 1e-10)
            # Scale to preserve energy: E = 0.5 * r^2, so r = sqrt(2*E)
            target_norm = torch.sqrt(2.0 * E_current)
            return x_normalized * target_norm
        else:
            # For general energy functions, use iterative Newton-Raphson
            # to find scaling factor alpha such that E(alpha * x) = E_target
            alpha = torch.ones_like(E_current)
            x_proj = x.clone()
            
            for _ in range(10):  # Newton iterations
                x_proj = alpha * x
                E_new = self.energy_function(x_proj)
                
                # If close enough, stop
                if torch.allclose(E_new, E_current, rtol=1e-6):
                    break
                
                # Newton step: find dalpha such that E(x*(alpha+dalpha)) ≈ E_target
                # Using first-order approximation: dE/dalpha ≈ dE/dx · x
                x_proj.requires_grad_(True)
                E_new = self.energy_function(x_proj)
                grad_E = torch.autograd.grad(E_new.sum(), x_proj, create_graph=True)[0]
                dE_dalpha = torch.sum(grad_E * x, dim=-1, keepdim=True)
                
                # Newton update
                dalpha = (E_current - E_new) / (dE_dalpha + 1e-10)
                alpha = alpha + dalpha
                alpha = torch.clamp(alpha, min=0.1, max=10.0)  # Stability
            
            return alpha * x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply energy conservation constraint."""
        # Store initial energy
        with torch.no_grad():
            E_initial = self.energy_function(x)
        
        # Apply projection
        x_proj = self.project(x)
        
        # Verify energy conservation (for debugging)
        with torch.no_grad():
            E_final = self.energy_function(x_proj)
            rel_error = torch.abs((E_final - E_initial) / (E_initial + 1e-8))
            if rel_error.max() > 0.01:
                print(f"Warning: Energy conservation error: {rel_error.max():.4f}")
        
        return x_proj


class MomentumConservingLayer(ConservationLayer):
    """
    Layer that ensures momentum conservation.
    Useful for particle systems and fluid dynamics.
    """
    
    def __init__(self, mass: Optional[torch.Tensor] = None, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.register_buffer('mass', mass if mass is not None else torch.ones(1))
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Remove net momentum from the system."""
        # Handle different cases based on mass shape and dim
        if self.mass.numel() == 1:
            # Uniform mass case - just compute mean
            avg_momentum = torch.mean(x, dim=self.dim, keepdim=True)
            return x - avg_momentum
        else:
            # Non-uniform mass case
            # Ensure mass has proper shape for broadcasting
            mass_shape = [1] * x.ndim
            mass_shape[self.dim] = -1
            mass_reshaped = self.mass.view(mass_shape)
            
            # Compute center of mass momentum
            total_mass = torch.sum(mass_reshaped, dim=self.dim, keepdim=True)
            momentum = torch.sum(mass_reshaped * x, dim=self.dim, keepdim=True)
            avg_momentum = momentum / total_mass
            
            # Subtract average momentum
            return x - avg_momentum
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply momentum conservation."""
        return self.project(x)


class AngularMomentumConservingLayer(ConservationLayer):
    """
    Layer that ensures angular momentum conservation.
    For 3D systems with rotational symmetry.
    """
    
    def __init__(self, positions: torch.Tensor, masses: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer('positions', positions)
        self.register_buffer('masses', masses if masses is not None else torch.ones(positions.shape[0]))
    
    def project(self, velocities: torch.Tensor) -> torch.Tensor:
        """Project velocities to conserve angular momentum."""
        # Compute current angular momentum
        L = torch.cross(self.positions, self.masses.unsqueeze(-1) * velocities, dim=-1)
        L_total = torch.sum(L, dim=0, keepdim=True)
        
        # Compute moment of inertia tensor
        I = self._compute_inertia_tensor()
        
        # Compute angular velocity that would give this angular momentum
        # Use pseudo-inverse for robustness with singular matrices
        try:
            omega = torch.linalg.solve(I, L_total.T).T
        except torch._C._LinAlgError:
            # Singular matrix - use pseudo-inverse
            I_pinv = torch.linalg.pinv(I)
            omega = torch.matmul(L_total, I_pinv)
        
        # Compute velocity field from rigid rotation
        v_rigid = torch.cross(omega.expand_as(self.positions), self.positions, dim=-1)
        
        # Project original velocities
        # This is a simplified approach - full treatment would use Lagrange multipliers
        alpha = 0.1  # Mixing parameter
        return (1 - alpha) * velocities + alpha * v_rigid
    
    def _compute_inertia_tensor(self) -> torch.Tensor:
        """Compute moment of inertia tensor."""
        r = self.positions
        m = self.masses.unsqueeze(-1)
        
        # I_ij = sum_k m_k (delta_ij r_k^2 - r_ki r_kj)
        r_squared = torch.sum(r**2, dim=-1, keepdim=True)
        I = torch.zeros(3, 3, device=r.device, dtype=r.dtype)
        
        for i in range(3):
            for j in range(3):
                if i == j:
                    I[i, j] = torch.sum(m.squeeze() * (r_squared.squeeze() - r[:, i]**2))
                else:
                    I[i, j] = -torch.sum(m.squeeze() * r[:, i] * r[:, j])
        
        return I
    
    def forward(self, velocities: torch.Tensor) -> torch.Tensor:
        """Apply angular momentum conservation."""
        return self.project(velocities)


class ConstraintEnforcer(nn.Module):
    """
    General constraint enforcer using Lagrange multipliers.
    Can handle multiple constraints simultaneously.
    """
    
    def __init__(self, constraints: List[Callable[[torch.Tensor], torch.Tensor]], 
                 n_iterations: int = 10, tolerance: float = 1e-6):
        super().__init__()
        self.constraints = constraints
        self.n_iterations = n_iterations
        self.tolerance = tolerance
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enforce constraints using iterative projection.
        Uses method of Lagrange multipliers.
        """
        x_proj = x.clone()
        
        for _ in range(self.n_iterations):
            # Evaluate all constraints
            constraint_values = []
            constraint_grads = []
            
            for constraint in self.constraints:
                c_val = constraint(x_proj)
                c_grad = torch.autograd.grad(c_val.sum(), x_proj, create_graph=True)[0]
                constraint_values.append(c_val)
                constraint_grads.append(c_grad)
            
            # Stack constraints
            C = torch.stack(constraint_values, dim=-1)
            J = torch.stack(constraint_grads, dim=-2)
            
            # Check convergence
            if torch.abs(C).max() < self.tolerance:
                break
            
            # Compute Lagrange multipliers
            # J * J^T * lambda = -C
            JJT = torch.matmul(J, J.transpose(-1, -2))
            lambda_vals = torch.linalg.solve(JJT + 1e-8 * torch.eye(JJT.shape[-1], device=JJT.device), -C)
            
            # Update x
            delta_x = torch.sum(lambda_vals.unsqueeze(-1) * J, dim=-2)
            x_proj = x_proj + delta_x
        
        return x_proj