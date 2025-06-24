"""
Symplectic integrators for Hamiltonian systems that preserve energy and phase space volume.
"""

import torch
from typing import Callable, Tuple, Optional
from abc import ABC, abstractmethod


class SymplecticIntegrator(ABC):
    """Base class for symplectic integrators."""
    
    def __init__(self, dt: float):
        self.dt = dt
    
    @abstractmethod
    def step(self, q: torch.Tensor, p: torch.Tensor, 
             hamiltonian: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one integration step."""
        pass
    
    def integrate(self, q0: torch.Tensor, p0: torch.Tensor,
                  hamiltonian: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                  n_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate for multiple steps."""
        q, p = q0.clone(), p0.clone()
        trajectory_q = [q.clone()]
        trajectory_p = [p.clone()]
        
        for _ in range(n_steps):
            q, p = self.step(q, p, hamiltonian)
            trajectory_q.append(q.clone())
            trajectory_p.append(p.clone())
        
        return torch.stack(trajectory_q), torch.stack(trajectory_p)


class SymplecticEuler(SymplecticIntegrator):
    """First-order symplectic Euler integrator."""
    
    def step(self, q: torch.Tensor, p: torch.Tensor,
             hamiltonian: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symplectic Euler step:
        p_{n+1} = p_n - dt * dH/dq(q_n, p_n)
        q_{n+1} = q_n + dt * dH/dp(q_n, p_{n+1})
        """
        # Enable gradient computation
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        
        # Compute Hamiltonian
        H = hamiltonian(q, p)
        
        # Compute gradients
        dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        
        # Update momentum first
        p_new = p - self.dt * dH_dq
        
        # Recompute Hamiltonian with new momentum
        H_new = hamiltonian(q, p_new)
        dH_dp = torch.autograd.grad(H_new.sum(), p_new, create_graph=True)[0]
        
        # Update position
        q_new = q + self.dt * dH_dp
        
        return q_new.detach(), p_new.detach()


class LeapfrogIntegrator(SymplecticIntegrator):
    """Second-order leapfrog (Störmer-Verlet) integrator."""
    
    def step(self, q: torch.Tensor, p: torch.Tensor,
             hamiltonian: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Leapfrog step:
        p_{n+1/2} = p_n - (dt/2) * dH/dq(q_n, p_n)
        q_{n+1} = q_n + dt * dH/dp(q_n, p_{n+1/2})
        p_{n+1} = p_{n+1/2} - (dt/2) * dH/dq(q_{n+1}, p_{n+1/2})
        """
        # Enable gradient computation
        q = q.requires_grad_(True)
        p = p.requires_grad_(True)
        
        # Half step for momentum
        H = hamiltonian(q, p)
        dH_dq = torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        p_half = p - 0.5 * self.dt * dH_dq
        
        # Full step for position
        H_half = hamiltonian(q, p_half)
        dH_dp = torch.autograd.grad(H_half.sum(), p_half, create_graph=True)[0]
        q_new = q + self.dt * dH_dp
        
        # Half step for momentum
        H_new = hamiltonian(q_new, p_half)
        dH_dq_new = torch.autograd.grad(H_new.sum(), q_new, create_graph=True)[0]
        p_new = p_half - 0.5 * self.dt * dH_dq_new
        
        return q_new.detach(), p_new.detach()


class VerletIntegrator(SymplecticIntegrator):
    """Velocity Verlet integrator for systems with separable Hamiltonians."""
    
    def __init__(self, dt: float, mass: Optional[torch.Tensor] = None):
        super().__init__(dt)
        self.mass = mass if mass is not None else 1.0
    
    def step(self, q: torch.Tensor, p: torch.Tensor,
             hamiltonian: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Velocity Verlet for H = T(p) + V(q):
        q_{n+1} = q_n + dt * p_n/m + (dt^2/2m) * F_n
        p_{n+1} = p_n + (dt/2) * (F_n + F_{n+1})
        where F = -dV/dq
        """
        # Enable gradient computation
        q = q.requires_grad_(True)
        
        # Compute force at current position
        H = hamiltonian(q, p)
        force = -torch.autograd.grad(H.sum(), q, create_graph=True)[0]
        
        # Update position
        velocity = p / self.mass
        q_new = q + self.dt * velocity + 0.5 * self.dt**2 * force / self.mass
        
        # Compute force at new position
        H_new = hamiltonian(q_new, p)
        force_new = -torch.autograd.grad(H_new.sum(), q_new, create_graph=True)[0]
        
        # Update momentum
        p_new = p + 0.5 * self.dt * (force + force_new)
        
        return q_new.detach(), p_new.detach()


def compute_energy_error(q_traj: torch.Tensor, p_traj: torch.Tensor,
                        hamiltonian: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """Compute relative energy error throughout trajectory."""
    energies = []
    for q, p in zip(q_traj, p_traj):
        H = hamiltonian(q, p)
        energies.append(H)
    
    energies = torch.stack(energies)
    initial_energy = energies[0]
    relative_errors = torch.abs((energies - initial_energy) / initial_energy)
    
    return relative_errors