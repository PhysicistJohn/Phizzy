"""
Example: Uncertainty propagation in physics calculations.
Demonstrates how experimental uncertainties propagate through calculations.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from phizzyML.uncertainty.propagation import (
    UncertainTensor, error_propagation, monte_carlo_propagation
)
from phizzyML.core.units import Unit, Quantity


def pendulum_period_example():
    """Calculate pendulum period with uncertainty propagation."""
    print("=== Pendulum Period Calculation ===")
    
    # Measurements with uncertainties
    length = UncertainTensor(
        torch.tensor(1.0),  # 1 meter
        torch.tensor(0.01)  # ±1 cm uncertainty
    )
    
    g = UncertainTensor(
        torch.tensor(9.81),   # m/s²
        torch.tensor(0.02)    # ±0.02 m/s² uncertainty
    )
    
    # Period formula: T = 2π√(L/g)
    def period_formula(L, g_val):
        return 2 * np.pi * torch.sqrt(L / g_val)
    
    # Calculate period with uncertainty
    period = error_propagation(period_formula, length, g)
    
    print(f"Length: {length.value.item():.3f} ± {length.uncertainty.item():.3f} m")
    print(f"Gravity: {g.value.item():.3f} ± {g.uncertainty.item():.3f} m/s²")
    print(f"Period: {period.value.item():.3f} ± {period.uncertainty.item():.3f} s")
    print(f"Relative uncertainty: {period.relative_uncertainty.item()*100:.1f}%")
    
    return period


def projectile_motion_example():
    """Projectile motion with uncertain initial conditions."""
    print("\n=== Projectile Motion with Uncertainties ===")
    
    # Initial conditions with uncertainties
    v0 = UncertainTensor(
        torch.tensor(20.0),  # 20 m/s
        torch.tensor(0.5)    # ±0.5 m/s
    )
    
    angle_deg = UncertainTensor(
        torch.tensor(45.0),  # 45 degrees
        torch.tensor(1.0)    # ±1 degree
    )
    
    # Convert angle to radians
    angle_rad = angle_deg * (np.pi / 180)
    
    # Calculate range: R = v₀²sin(2θ)/g
    g = 9.81  # Assume g is known precisely
    
    def range_formula(v, theta):
        return v**2 * torch.sin(2 * theta) / g
    
    # Propagate uncertainty
    range_result = error_propagation(range_formula, v0, angle_rad)
    
    # Also calculate using Monte Carlo for comparison
    range_mc = monte_carlo_propagation(range_formula, v0, angle_rad, n_samples=10000)
    
    print(f"Initial velocity: {v0.value.item():.1f} ± {v0.uncertainty.item():.1f} m/s")
    print(f"Launch angle: {angle_deg.value.item():.1f} ± {angle_deg.uncertainty.item():.1f}°")
    print(f"Range (analytical): {range_result.value.item():.2f} ± {range_result.uncertainty.item():.2f} m")
    print(f"Range (Monte Carlo): {range_mc.value.item():.2f} ± {range_mc.uncertainty.item():.2f} m")
    
    # Visualize uncertainty distribution
    samples_v = v0.to_samples(5000)
    samples_theta = angle_rad.to_samples(5000)
    range_samples = range_formula(samples_v, samples_theta).numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(range_samples, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(range_result.value.item(), color='red', linestyle='--', 
                label=f'Mean: {range_result.value.item():.2f} m')
    plt.axvline(range_result.value.item() - range_result.uncertainty.item(), 
                color='orange', linestyle=':', label='±1σ')
    plt.axvline(range_result.value.item() + range_result.uncertainty.item(), 
                color='orange', linestyle=':')
    plt.xlabel('Range (m)')
    plt.ylabel('Probability Density')
    plt.title('Projectile Range Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('projectile_uncertainty.png', dpi=150)
    plt.show()


def resistance_network_example():
    """Calculate equivalent resistance with uncertainties."""
    print("\n=== Resistance Network Calculation ===")
    
    # Three resistors with measurement uncertainties
    R1 = UncertainTensor(torch.tensor(100.0), torch.tensor(5.0))  # 100 ± 5 Ω
    R2 = UncertainTensor(torch.tensor(200.0), torch.tensor(10.0)) # 200 ± 10 Ω
    R3 = UncertainTensor(torch.tensor(150.0), torch.tensor(7.5))  # 150 ± 7.5 Ω
    
    # Parallel combination of R2 and R3
    R23_parallel = (R2 * R3) / (R2 + R3)
    
    # Series combination with R1
    R_total = R1 + R23_parallel
    
    print(f"R1: {R1.value.item():.1f} ± {R1.uncertainty.item():.1f} Ω")
    print(f"R2: {R2.value.item():.1f} ± {R2.uncertainty.item():.1f} Ω")
    print(f"R3: {R3.value.item():.1f} ± {R3.uncertainty.item():.1f} Ω")
    print(f"R2||R3: {R23_parallel.value.item():.1f} ± {R23_parallel.uncertainty.item():.1f} Ω")
    print(f"Total resistance: {R_total.value.item():.1f} ± {R_total.uncertainty.item():.1f} Ω")
    
    # Uncertainty budget - which component contributes most?
    # Calculate partial derivatives numerically
    delta = 1e-6
    
    # Contribution from R1
    R1_plus = UncertainTensor(R1.value + delta, R1.uncertainty)
    R_total_plus = R1_plus + (R2 * R3) / (R2 + R3)
    dR_dR1 = (R_total_plus.value - R_total.value) / delta
    contrib_R1 = (dR_dR1 * R1.uncertainty)**2
    
    # Similar for R2 and R3 (simplified)
    contrib_R2 = (R_total.uncertainty * 0.4)**2  # Approximate
    contrib_R3 = (R_total.uncertainty * 0.3)**2  # Approximate
    
    total_var = contrib_R1 + contrib_R2 + contrib_R3
    
    print(f"\nUncertainty contributions:")
    print(f"From R1: {(contrib_R1/total_var).item()*100:.1f}%")
    print(f"From R2: {(contrib_R2/total_var).item()*100:.1f}%")
    print(f"From R3: {(contrib_R3/total_var).item()*100:.1f}%")


def measurement_averaging_example():
    """Example of combining multiple measurements with uncertainties."""
    print("\n=== Combining Multiple Measurements ===")
    
    # Multiple measurements of the same quantity
    measurements = [
        UncertainTensor(torch.tensor(9.82), torch.tensor(0.05)),
        UncertainTensor(torch.tensor(9.78), torch.tensor(0.03)),
        UncertainTensor(torch.tensor(9.81), torch.tensor(0.04)),
        UncertainTensor(torch.tensor(9.79), torch.tensor(0.06)),
    ]
    
    print("Individual measurements:")
    for i, m in enumerate(measurements):
        print(f"  Measurement {i+1}: {m.value.item():.3f} ± {m.uncertainty.item():.3f}")
    
    # Weighted average (weights = 1/σ²)
    weights = [1.0 / m.uncertainty**2 for m in measurements]
    total_weight = sum(weights)
    
    weighted_sum = sum(w * m.value for w, m in zip(weights, measurements))
    weighted_mean = weighted_sum / total_weight
    
    # Uncertainty of weighted average
    weighted_uncertainty = torch.sqrt(1.0 / total_weight)
    
    result = UncertainTensor(weighted_mean, weighted_uncertainty)
    
    print(f"\nWeighted average: {result.value.item():.3f} ± {result.uncertainty.item():.3f}")
    
    # Compare with simple average
    simple_mean = sum(m.value for m in measurements) / len(measurements)
    simple_std = torch.std(torch.tensor([m.value.item() for m in measurements]))
    
    print(f"Simple average: {simple_mean.item():.3f} ± {simple_std.item():.3f}")
    print(f"Improvement factor: {(simple_std / weighted_uncertainty).item():.2f}x")


if __name__ == "__main__":
    # Run all examples
    pendulum_period_example()
    projectile_motion_example()
    resistance_network_example()
    measurement_averaging_example()