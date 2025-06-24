"""
Example: Neural network with physical constraints.
Shows how to enforce energy and momentum conservation in neural networks.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from phizzyML.constraints.conservation import (
    EnergyConservingLayer, MomentumConservingLayer
)


class PhysicsConstrainedNetwork(nn.Module):
    """Neural network that respects physical conservation laws."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Standard neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Physics constraint layers
        self.energy_constraint = EnergyConservingLayer()
        self.momentum_constraint = MomentumConservingLayer()
        
        self.activation = nn.Tanh()
    
    def forward(self, x, enforce_constraints=True):
        # Standard forward pass
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        if enforce_constraints:
            # Apply physical constraints
            x = self.energy_constraint(x)
            x = self.momentum_constraint(x)
        
        return x


def simulate_particle_system():
    """Simulate a particle system with and without constraints."""
    
    # Create network
    n_particles = 10
    dim = 3  # 3D positions
    network = PhysicsConstrainedNetwork(
        input_dim=n_particles * dim,
        hidden_dim=64,
        output_dim=n_particles * dim  # Output velocities
    )
    
    # Generate random initial positions
    torch.manual_seed(42)
    positions = torch.randn(100, n_particles, dim)  # 100 time steps
    positions_flat = positions.view(100, -1)
    
    # Compute velocities with and without constraints
    with torch.no_grad():
        velocities_constrained = network(positions_flat, enforce_constraints=True)
        velocities_unconstrained = network(positions_flat, enforce_constraints=False)
    
    # Reshape velocities
    velocities_constrained = velocities_constrained.view(100, n_particles, dim)
    velocities_unconstrained = velocities_unconstrained.view(100, n_particles, dim)
    
    # Compute conservation metrics
    def compute_total_energy(velocities):
        """Kinetic energy only for simplicity."""
        return 0.5 * torch.sum(velocities**2, dim=(1, 2))
    
    def compute_total_momentum(velocities):
        """Total momentum of the system."""
        return torch.sum(velocities, dim=1)
    
    # Calculate metrics
    energy_constrained = compute_total_energy(velocities_constrained).numpy()
    energy_unconstrained = compute_total_energy(velocities_unconstrained).numpy()
    
    momentum_constrained = compute_total_momentum(velocities_constrained).numpy()
    momentum_unconstrained = compute_total_momentum(velocities_unconstrained).numpy()
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy conservation
    ax = axes[0, 0]
    ax.plot(energy_unconstrained, 'r-', label='Unconstrained', alpha=0.7)
    ax.plot(energy_constrained, 'b-', label='Constrained', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Conservation')
    ax.legend()
    ax.grid(True)
    
    # Energy variation
    ax = axes[0, 1]
    energy_var_unconstrained = np.std(energy_unconstrained) / np.mean(energy_unconstrained)
    energy_var_constrained = np.std(energy_constrained) / np.mean(energy_constrained)
    ax.bar(['Unconstrained', 'Constrained'], 
           [energy_var_unconstrained, energy_var_constrained],
           color=['red', 'blue'], alpha=0.7)
    ax.set_ylabel('Relative Energy Variation')
    ax.set_title('Energy Stability')
    ax.grid(True, axis='y')
    
    # Momentum conservation (x-component)
    ax = axes[1, 0]
    ax.plot(momentum_unconstrained[:, 0], 'r-', label='Unconstrained', alpha=0.7)
    ax.plot(momentum_constrained[:, 0], 'b-', label='Constrained', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Total Momentum (x)')
    ax.set_title('Momentum Conservation (x-component)')
    ax.legend()
    ax.grid(True)
    
    # Total momentum magnitude
    ax = axes[1, 1]
    momentum_mag_unconstrained = np.linalg.norm(momentum_unconstrained, axis=1)
    momentum_mag_constrained = np.linalg.norm(momentum_constrained, axis=1)
    ax.semilogy(momentum_mag_unconstrained, 'r-', label='Unconstrained', alpha=0.7)
    ax.semilogy(momentum_mag_constrained, 'b-', label='Constrained', alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('|Total Momentum|')
    ax.set_title('Total Momentum Magnitude')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('constrained_dynamics.png', dpi=150)
    plt.show()
    
    # Print statistics
    print("Conservation Statistics:")
    print(f"Energy variation (unconstrained): {energy_var_unconstrained:.4f}")
    print(f"Energy variation (constrained): {energy_var_constrained:.4f}")
    print(f"Mean momentum magnitude (unconstrained): {momentum_mag_unconstrained.mean():.4f}")
    print(f"Mean momentum magnitude (constrained): {momentum_mag_constrained.mean():.4f}")


def train_with_constraints():
    """Example of training a network with physical constraints."""
    
    # Create a simple prediction task
    n_samples = 1000
    n_particles = 5
    dim = 2
    
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_particles * dim)
    # Target: velocities that conserve energy and momentum
    y_true = torch.randn(n_samples, n_particles * dim)
    # Normalize to have constant energy
    energy = torch.sum(y_true**2, dim=1, keepdim=True)
    y_true = y_true / torch.sqrt(energy) * 2.0  # Constant energy = 2
    
    # Create network
    model = PhysicsConstrainedNetwork(
        input_dim=n_particles * dim,
        hidden_dim=32,
        output_dim=n_particles * dim
    )
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    losses = []
    for epoch in range(100):
        # Forward pass
        y_pred = model(X, enforce_constraints=True)
        loss = loss_fn(y_pred, y_true)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Plot training curve
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training with Physical Constraints')
    plt.grid(True)
    plt.savefig('training_with_constraints.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    print("Simulating particle system with conservation laws...")
    simulate_particle_system()
    
    print("\nTraining network with physical constraints...")
    train_with_constraints()