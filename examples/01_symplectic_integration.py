"""
Example: Symplectic Integration for Hamiltonian Systems
======================================================

This example demonstrates the use of symplectic integrators that preserve
the geometric structure of phase space and conserve energy over long times.
"""

import numpy as np
import matplotlib.pyplot as plt
from phizzy import symplectic_integrate

# Example 1: Simple Harmonic Oscillator
print("Example 1: Simple Harmonic Oscillator")
print("-" * 40)

def harmonic_oscillator(q, p):
    """Hamiltonian for harmonic oscillator: H = p²/2m + kq²/2"""
    m, k = 1.0, 1.0
    return 0.5 * p**2 / m + 0.5 * k * q**2

# Initial conditions
q0 = np.array([1.0])  # Initial position
p0 = np.array([0.0])  # Initial momentum

# Compare different integration methods
methods = ['leapfrog', 'yoshida4', 'forest-ruth', 'pefrl']
colors = ['blue', 'red', 'green', 'purple']

plt.figure(figsize=(12, 8))

for i, method in enumerate(methods):
    result = symplectic_integrate(
        harmonic_oscillator,
        q0, p0,
        t_span=(0, 100 * np.pi),  # 50 periods
        dt=0.1,
        method=method
    )
    
    print(f"\n{method.capitalize()} method:")
    print(f"  Energy error: {result.energy_error:.2e}")
    print(f"  Energy drift: {result.info['energy_drift']:.2e}")
    
    # Plot energy conservation
    plt.subplot(2, 2, 1)
    plt.plot(result.t, (result.energy - result.energy[0]) / result.energy[0], 
             label=method, color=colors[i], alpha=0.7)
    
    # Plot phase space
    plt.subplot(2, 2, 2)
    plt.plot(result.q[:, 0], result.p[:, 0], color=colors[i], alpha=0.5, 
             label=method if i == 0 else None)

# Energy conservation plot
plt.subplot(2, 2, 1)
plt.xlabel('Time')
plt.ylabel('Relative Energy Error')
plt.title('Energy Conservation Comparison')
plt.legend()
plt.grid(True)
plt.yscale('log')

# Phase space plot
plt.subplot(2, 2, 2)
plt.xlabel('Position (q)')
plt.ylabel('Momentum (p)')
plt.title('Phase Space Trajectory')
plt.axis('equal')
plt.grid(True)

# Example 2: Kepler Problem (Planetary Motion)
print("\n\nExample 2: Kepler Problem - Planetary Motion")
print("-" * 40)

def kepler_hamiltonian(q, p):
    """Hamiltonian for 2D Kepler problem: H = |p|²/2 - GM/|q|"""
    GM = 1.0  # Gravitational parameter
    r = np.sqrt(q[0]**2 + q[1]**2)
    kinetic = 0.5 * (p[0]**2 + p[1]**2)
    potential = -GM / r
    return kinetic + potential

# Elliptical orbit initial conditions
q0 = np.array([1.0, 0.0])    # Start at perihelion
p0 = np.array([0.0, 0.95])   # Velocity for elliptical orbit

result = symplectic_integrate(
    kepler_hamiltonian,
    q0, p0,
    t_span=(0, 20 * 2 * np.pi),
    dt=0.01,
    method='pefrl'  # Use highest-order method
)

print(f"Energy conservation: {result.energy_error:.2e}")

# Compute angular momentum
L = result.q[:, 0] * result.p[:, 1] - result.q[:, 1] * result.p[:, 0]
L_error = (np.max(L) - np.min(L)) / np.mean(L)
print(f"Angular momentum conservation: {L_error:.2e}")

# Plot orbit
plt.subplot(2, 2, 3)
plt.plot(result.q[:, 0], result.q[:, 1], 'b-', alpha=0.6)
plt.plot(0, 0, 'yo', markersize=10, label='Star')
plt.plot(result.q[0, 0], result.q[0, 1], 'go', markersize=8, label='Start')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Kepler Orbit')
plt.axis('equal')
plt.grid(True)
plt.legend()

# Plot conserved quantities
plt.subplot(2, 2, 4)
plt.plot(result.t, result.energy, label='Energy')
plt.plot(result.t, L, label='Angular Momentum')
plt.xlabel('Time')
plt.ylabel('Conserved Quantity')
plt.title('Conservation Laws')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('examples/symplectic_integration_demo.png', dpi=150)
plt.show()

# Example 3: Hénon-Heiles System (Chaotic Dynamics)
print("\n\nExample 3: Hénon-Heiles System")
print("-" * 40)

def henon_heiles(q, p):
    """Hénon-Heiles Hamiltonian - a chaotic system"""
    x, y = q
    px, py = p
    
    # Kinetic energy
    T = 0.5 * (px**2 + py**2)
    
    # Potential energy
    V = 0.5 * (x**2 + y**2) + x**2 * y - y**3 / 3
    
    return T + V

# Different initial conditions lead to different behaviors
initial_conditions = [
    (np.array([0.0, 0.1]), np.array([0.5, 0.0])),  # Regular
    (np.array([0.0, 0.1]), np.array([0.55, 0.0])), # Mixed
    (np.array([0.0, 0.1]), np.array([0.6, 0.0]))   # Chaotic
]

plt.figure(figsize=(12, 4))

for i, (q0, p0) in enumerate(initial_conditions):
    result = symplectic_integrate(
        henon_heiles,
        q0, p0,
        t_span=(0, 1000),
        dt=0.01,
        method='yoshida4'
    )
    
    E0 = result.energy[0]
    print(f"\nInitial energy E = {E0:.3f}:")
    print(f"  Energy error: {result.energy_error:.2e}")
    
    plt.subplot(1, 3, i+1)
    plt.plot(result.q[:, 0], result.q[:, 1], ',', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'E = {E0:.3f}')
    plt.axis('equal')

plt.suptitle('Hénon-Heiles System: Transition to Chaos')
plt.tight_layout()
plt.savefig('examples/henon_heiles_chaos.png', dpi=150)
plt.show()

print("\n" + "="*50)
print("Symplectic Integration Examples Complete!")
print("\nKey Takeaways:")
print("1. Higher-order methods (Yoshida4, PEFRL) conserve energy better")
print("2. Symplectic integrators preserve phase space structure")
print("3. They are essential for long-time integration of Hamiltonian systems")
print("4. Work well for both regular and chaotic dynamics")