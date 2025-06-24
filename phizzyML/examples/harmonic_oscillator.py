"""
Example: Simulating a harmonic oscillator with symplectic integration.
Demonstrates energy conservation and phase space evolution.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from phizzyML.integrators.symplectic import LeapfrogIntegrator, compute_energy_error
from phizzyML.core.units import Unit, Quantity


def harmonic_oscillator_hamiltonian(q, p, k=1.0, m=1.0):
    """Hamiltonian for harmonic oscillator: H = p²/(2m) + kq²/2"""
    kinetic = torch.sum(p**2) / (2 * m)
    potential = k * torch.sum(q**2) / 2
    return kinetic + potential


def main():
    # Physical parameters with units
    mass = Quantity(1.0, Unit.kilogram)
    spring_constant = Quantity(1.0, Unit.newton / Unit.meter)
    
    # Initial conditions
    initial_position = Quantity(1.0, Unit.meter)
    initial_momentum = Quantity(0.0, Unit.kilogram * Unit.meter / Unit.second)
    
    # Time parameters
    dt = 0.01
    total_time = 50.0
    n_steps = int(total_time / dt)
    
    # Create integrator
    integrator = LeapfrogIntegrator(dt=dt)
    
    # Convert to dimensionless for simulation
    q0 = torch.tensor([initial_position.magnitude])
    p0 = torch.tensor([initial_momentum.magnitude])
    
    # Integrate
    print("Simulating harmonic oscillator...")
    q_traj, p_traj = integrator.integrate(
        q0, p0, harmonic_oscillator_hamiltonian, n_steps
    )
    
    # Convert trajectories to numpy for plotting
    q_traj = q_traj.numpy()
    p_traj = p_traj.numpy()
    time = np.linspace(0, total_time, n_steps + 1)
    
    # Compute energy over time
    energies = []
    for q, p in zip(q_traj, p_traj):
        H = harmonic_oscillator_hamiltonian(
            torch.tensor(q), torch.tensor(p)
        ).item()
        energies.append(H)
    energies = np.array(energies)
    
    # Analytical solution for comparison
    omega = np.sqrt(spring_constant.magnitude / mass.magnitude)
    q_analytical = initial_position.magnitude * np.cos(omega * time)
    p_analytical = -mass.magnitude * omega * initial_position.magnitude * np.sin(omega * time)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position vs time
    ax = axes[0, 0]
    ax.plot(time, q_traj[:, 0], 'b-', label='Numerical', linewidth=2)
    ax.plot(time, q_analytical, 'r--', label='Analytical', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position vs Time')
    ax.legend()
    ax.grid(True)
    
    # Phase space
    ax = axes[0, 1]
    ax.plot(q_traj[:, 0], p_traj[:, 0], 'b-', alpha=0.7)
    ax.scatter(q_traj[0, 0], p_traj[0, 0], c='g', s=100, marker='o', label='Start')
    ax.scatter(q_traj[-1, 0], p_traj[-1, 0], c='r', s=100, marker='s', label='End')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Momentum (kg⋅m/s)')
    ax.set_title('Phase Space Portrait')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Energy conservation
    ax = axes[1, 0]
    energy_errors = compute_energy_error(
        torch.tensor(q_traj), torch.tensor(p_traj), 
        harmonic_oscillator_hamiltonian
    ).numpy()
    ax.semilogy(time, energy_errors, 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Relative Energy Error')
    ax.set_title('Energy Conservation')
    ax.grid(True)
    
    # Energy components
    ax = axes[1, 1]
    kinetic = 0.5 * mass.magnitude * np.sum(p_traj**2, axis=1)
    potential = 0.5 * spring_constant.magnitude * np.sum(q_traj**2, axis=1)
    ax.plot(time, kinetic, 'r-', label='Kinetic', linewidth=2)
    ax.plot(time, potential, 'b-', label='Potential', linewidth=2)
    ax.plot(time, energies, 'k--', label='Total', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy (J)')
    ax.set_title('Energy Components')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('harmonic_oscillator_simulation.png', dpi=150)
    plt.show()
    
    # Print summary statistics
    print("\nSimulation Summary:")
    print(f"Total time: {total_time} s")
    print(f"Time step: {dt} s")
    print(f"Number of steps: {n_steps}")
    print(f"Initial energy: {energies[0]:.6f} J")
    print(f"Final energy: {energies[-1]:.6f} J")
    print(f"Maximum relative energy error: {energy_errors.max():.2e}")
    print(f"Average relative energy error: {energy_errors.mean():.2e}")


if __name__ == "__main__":
    main()