"""
Physics-aware optimizers that leverage physical principles for better convergence.
"""

import torch
from torch.optim import Optimizer
from typing import Callable, Optional, Dict, Any, Tuple
import math


class EnergyMinimizer(Optimizer):
    """
    Optimizer that uses energy minimization principles.
    Suitable for finding equilibrium states in physical systems.
    """
    
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.9,
                 damping: float = 0.1, energy_func: Optional[Callable] = None):
        """
        Initialize energy minimizer.
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate (analogous to time step)
            momentum: Momentum coefficient
            damping: Damping coefficient for stability
            energy_func: Optional energy function to minimize
        """
        defaults = dict(lr=lr, momentum=momentum, damping=damping)
        super().__init__(params, defaults)
        self.energy_func = energy_func
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform optimization step using physics-inspired dynamics.
        Uses equation: m*a = -grad(E) - γ*v
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            momentum = group['momentum']
            damping = group['damping']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Get or initialize velocity (momentum buffer)
                param_state = self.state[p]
                if 'velocity' not in param_state:
                    param_state['velocity'] = torch.zeros_like(p.data)
                
                velocity = param_state['velocity']
                
                # Update velocity: v = momentum * v - lr * grad - damping * v
                velocity.mul_(momentum * (1 - damping)).add_(grad, alpha=-lr)
                
                # Update position: x = x + dt * v
                p.data.add_(velocity, alpha=lr)
        
        return loss


class VariationalOptimizer(Optimizer):
    """
    Optimizer based on variational principles.
    Minimizes action functional for dynamical systems.
    """
    
    def __init__(self, params, lr: float = 1e-3, 
                 lagrangian: Optional[Callable] = None,
                 constraints: Optional[list] = None):
        """
        Initialize variational optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate
            lagrangian: Lagrangian function L(q, q_dot)
            constraints: List of constraint functions
        """
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.lagrangian = lagrangian
        self.constraints = constraints or []
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform variational optimization step.
        Uses Euler-Lagrange equations.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Get or initialize state
                param_state = self.state[p]
                if 'velocity' not in param_state:
                    param_state['velocity'] = torch.zeros_like(p.data)
                    param_state['prev_position'] = p.data.clone()
                
                velocity = param_state['velocity']
                prev_pos = param_state['prev_position']
                
                # Compute velocity from positions
                velocity = (p.data - prev_pos) / lr
                
                # Apply Euler-Lagrange equation
                # d/dt(∂L/∂q̇) - ∂L/∂q = 0
                if self.lagrangian is not None:
                    # This is simplified - full implementation would compute
                    # partial derivatives of Lagrangian
                    effective_force = -grad
                else:
                    effective_force = -grad
                
                # Update velocity
                velocity.add_(effective_force, alpha=lr)
                
                # Apply constraints using Lagrange multipliers
                if self.constraints:
                    for constraint in self.constraints:
                        # Simplified constraint handling
                        c_val = constraint(p.data)
                        if c_val.abs().max() > 1e-6:
                            # Project onto constraint surface
                            c_grad = torch.autograd.grad(c_val.sum(), p)[0]
                            velocity -= (velocity * c_grad).sum() / (c_grad * c_grad).sum() * c_grad
                
                # Update position
                param_state['prev_position'] = p.data.clone()
                p.data.add_(velocity, alpha=lr)
                param_state['velocity'] = velocity
        
        return loss


class LangevinOptimizer(Optimizer):
    """
    Langevin dynamics optimizer for sampling from energy landscapes.
    Includes thermal noise for exploration.
    """
    
    def __init__(self, params, lr: float = 1e-3, temperature: float = 1.0,
                 friction: float = 1.0, noise_scale: Optional[float] = None):
        """
        Initialize Langevin dynamics optimizer.
        
        Args:
            params: Model parameters
            lr: Learning rate (time step)
            temperature: Temperature parameter (kT)
            friction: Friction coefficient
            noise_scale: Scale of random noise (computed from temperature if None)
        """
        if noise_scale is None:
            noise_scale = math.sqrt(2 * temperature * friction * lr)
        
        defaults = dict(lr=lr, temperature=temperature, 
                       friction=friction, noise_scale=noise_scale)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform Langevin dynamics step.
        dx = -∇U(x)dt + √(2kT/γ)dW
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            friction = group['friction']
            noise_scale = group['noise_scale']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Deterministic part: gradient descent
                p.data.add_(grad, alpha=-lr / friction)
                
                # Stochastic part: thermal noise
                noise = torch.randn_like(p.data) * noise_scale
                p.data.add_(noise)
        
        return loss


class HamiltonianMonteCarlo(Optimizer):
    """
    Hamiltonian Monte Carlo optimizer for efficient sampling.
    Uses Hamiltonian dynamics for proposals.
    """
    
    def __init__(self, params, step_size: float = 0.01, n_steps: int = 10,
                 mass_matrix: Optional[torch.Tensor] = None,
                 target_accept_rate: float = 0.8):
        """
        Initialize HMC optimizer.
        
        Args:
            params: Model parameters
            step_size: Integration step size
            n_steps: Number of leapfrog steps
            mass_matrix: Mass matrix for kinetic energy
            target_accept_rate: Target acceptance rate for adaptive step size
        """
        defaults = dict(step_size=step_size, n_steps=n_steps,
                       target_accept_rate=target_accept_rate)
        super().__init__(params, defaults)
        self.mass_matrix = mass_matrix
        self.accept_count = 0
        self.total_count = 0
    
    def _hamiltonian(self, position: torch.Tensor, momentum: torch.Tensor,
                     potential: torch.Tensor) -> torch.Tensor:
        """Compute total Hamiltonian H = K + U."""
        if self.mass_matrix is not None:
            kinetic = 0.5 * torch.sum(momentum * torch.matmul(
                torch.inverse(self.mass_matrix), momentum))
        else:
            kinetic = 0.5 * torch.sum(momentum ** 2)
        return kinetic + potential
    
    def _leapfrog_step(self, position: torch.Tensor, momentum: torch.Tensor,
                      grad: torch.Tensor, step_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single leapfrog integration step."""
        # Half step for momentum
        momentum = momentum - 0.5 * step_size * grad
        # Full step for position
        position = position + step_size * momentum
        # Half step for momentum (grad will be recomputed)
        return position, momentum
    
    def step(self, closure: Callable) -> Optional[float]:
        """
        Perform HMC step with Metropolis acceptance.
        """
        # Get current potential (negative log probability)
        current_U = closure()
        
        for group in self.param_groups:
            step_size = group['step_size']
            n_steps = group['n_steps']
            
            # Collect all parameters
            params_list = []
            momentum_list = []
            grad_list = []
            
            for p in group['params']:
                if p.grad is None:
                    continue
                params_list.append(p)
                # Sample momentum
                momentum = torch.randn_like(p.data)
                momentum_list.append(momentum)
                grad_list.append(p.grad.data.clone())
            
            # Store initial state
            initial_params = [p.data.clone() for p in params_list]
            initial_momentum = [m.clone() for m in momentum_list]
            
            # Compute initial Hamiltonian
            initial_H = self._hamiltonian(
                torch.cat([p.flatten() for p in initial_params]),
                torch.cat([m.flatten() for m in initial_momentum]),
                current_U
            )
            
            # Leapfrog integration
            for _ in range(n_steps):
                # Update positions and momenta
                for i, (p, m, g) in enumerate(zip(params_list, momentum_list, grad_list)):
                    new_p, new_m = self._leapfrog_step(p.data, m, g, step_size)
                    p.data = new_p
                    momentum_list[i] = new_m
                
                # Recompute gradients
                proposed_U = closure()
                for i, p in enumerate(params_list):
                    grad_list[i] = p.grad.data.clone()
            
            # Final half step for momentum
            for i, (m, g) in enumerate(zip(momentum_list, grad_list)):
                momentum_list[i] = m - 0.5 * step_size * g
            
            # Compute final Hamiltonian
            final_H = self._hamiltonian(
                torch.cat([p.data.flatten() for p in params_list]),
                torch.cat([m.flatten() for m in momentum_list]),
                proposed_U
            )
            
            # Metropolis acceptance step
            delta_H = final_H - initial_H
            accept_prob = torch.exp(-delta_H).clamp(max=1.0)
            
            self.total_count += 1
            if torch.rand(1) < accept_prob:
                # Accept proposal
                self.accept_count += 1
                loss = proposed_U
            else:
                # Reject proposal, restore initial state
                for p, init_p in zip(params_list, initial_params):
                    p.data = init_p
                loss = current_U
            
            # Adaptive step size
            if self.total_count % 100 == 0:
                accept_rate = self.accept_count / self.total_count
                if accept_rate < group['target_accept_rate'] - 0.1:
                    group['step_size'] *= 0.9
                elif accept_rate > group['target_accept_rate'] + 0.1:
                    group['step_size'] *= 1.1
        
        return loss