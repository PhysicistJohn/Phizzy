"""
Example: Symbolic Uncertainty Propagation
========================================

This example shows how to propagate measurement uncertainties through
mathematical expressions while maintaining symbolic representations.
"""

import numpy as np
import matplotlib.pyplot as plt
from phizzy import propagate_uncertainties_symbolic
from phizzy.uncertainties.symbolic import UncertainValue

print("Symbolic Uncertainty Propagation Examples")
print("=" * 50)

# Example 1: Ohm's Law with Uncertainties
print("\nExample 1: Ohm's Law - Power Calculation")
print("-" * 40)

# Measured values with uncertainties
variables = {
    'V': (12.0, 0.1),   # Voltage: 12.0 ± 0.1 V
    'I': (2.5, 0.05)    # Current: 2.5 ± 0.05 A
}

# Calculate power P = V * I
power_result = propagate_uncertainties_symbolic('V * I', variables)

print(f"Power = {power_result.numeric_result}")
print(f"\nSymbolic uncertainty expression:")
print(f"  σ_P = {power_result.symbolic_uncertainty}")
print(f"\nSensitivity coefficients:")
print(f"  ∂P/∂V = {power_result.sensitivity_coefficients['V']:.3f}")
print(f"  ∂P/∂I = {power_result.sensitivity_coefficients['I']:.3f}")
print(f"\nRelative contributions:")
for var, contrib in power_result.relative_contributions.items():
    print(f"  {var}: {contrib*100:.1f}%")

# Example 2: Correlated Variables
print("\n\nExample 2: Correlated Measurements")
print("-" * 40)

# Two resistors in series, measured with same instrument (correlated errors)
variables = {
    'R1': (100.0, 2.0),  # 100 ± 2 Ω
    'R2': (150.0, 3.0)   # 150 ± 3 Ω
}

# Positive correlation due to systematic errors in measurement instrument
correlations = {('R1', 'R2'): 0.6}

# Total resistance R_total = R1 + R2
result_corr = propagate_uncertainties_symbolic(
    'R1 + R2', variables, correlations=correlations
)
result_uncorr = propagate_uncertainties_symbolic(
    'R1 + R2', variables, correlations={}
)

print(f"Total resistance (correlated): {result_corr.numeric_result}")
print(f"Total resistance (uncorrelated): {result_uncorr.numeric_result}")
print(f"\nCorrelation increases uncertainty by {(result_corr.numeric_result.uncertainty / result_uncorr.numeric_result.uncertainty - 1)*100:.1f}%")

# Example 3: Complex Expression - Lens Equation
print("\n\nExample 3: Thin Lens Equation")
print("-" * 40)

# 1/f = 1/u + 1/v, solve for f
variables = {
    'u': (30.0, 0.5),   # Object distance: 30 ± 0.5 cm
    'v': (60.0, 1.0)    # Image distance: 60 ± 1.0 cm
}

# Focal length f = uv/(u+v)
focal_result = propagate_uncertainties_symbolic(
    'u * v / (u + v)', variables
)

print(f"Focal length: {focal_result.numeric_result} cm")
print(f"\nWhich variable contributes more to uncertainty?")
for var, contrib in focal_result.relative_contributions.items():
    print(f"  {var}: {contrib*100:.1f}%")

# Example 4: Comparison of Propagation Methods
print("\n\nExample 4: Comparing Propagation Methods")
print("-" * 40)

# Non-linear expression where Taylor might not be accurate
variables = {
    'x': (2.0, 0.5),    # Large relative uncertainty
    'y': (3.0, 0.3)
}

expression = 'x**2 * sin(y)'

# Taylor method (first-order)
taylor_result = propagate_uncertainties_symbolic(
    expression, variables, method='taylor'
)

# Monte Carlo method
mc_result = propagate_uncertainties_symbolic(
    expression, variables, method='monte_carlo', monte_carlo_samples=100000
)

# Bootstrap method
bootstrap_result = propagate_uncertainties_symbolic(
    expression, variables, method='bootstrap', monte_carlo_samples=50000
)

print(f"Expression: {expression}")
print(f"\nResults:")
print(f"  Taylor method: {taylor_result.numeric_result}")
print(f"  Monte Carlo: {mc_result.numeric_result}")
print(f"  Bootstrap: {bootstrap_result.numeric_result}")

print(f"\nMonte Carlo distribution statistics:")
print(f"  Skewness: {mc_result.info['skewness']:.3f}")
print(f"  Kurtosis: {mc_result.info['kurtosis']:.3f}")

# Example 5: Physics Application - Pendulum Period
print("\n\nExample 5: Pendulum Period with Multiple Uncertainties")
print("-" * 40)

# Period T = 2π√(L/g) with uncertainties in length, gravity, and angle
variables = {
    'L': (1.00, 0.01),      # Length: 1.00 ± 0.01 m
    'g': (9.81, 0.02),      # Local gravity: 9.81 ± 0.02 m/s²
    'theta': (0.1, 0.01)    # Initial angle: 0.1 ± 0.01 rad (small angle)
}

# Include small-angle correction: T = 2π√(L/g) * (1 + θ²/16)
period_result = propagate_uncertainties_symbolic(
    '2 * pi * sqrt(L / g) * (1 + theta**2 / 16)',
    variables
)

print(f"Pendulum period: {period_result.numeric_result} s")
print(f"\nContributions to uncertainty:")
for var, contrib in period_result.relative_contributions.items():
    print(f"  {var}: {contrib*100:.1f}%")

# Visualize Monte Carlo distribution
mc_vis = propagate_uncertainties_symbolic(
    '2 * pi * sqrt(L / g) * (1 + theta**2 / 16)',
    variables,
    method='monte_carlo',
    monte_carlo_samples=50000
)

# Generate samples for visualization
n_samples = 10000
L_samples = np.random.normal(1.00, 0.01, n_samples)
g_samples = np.random.normal(9.81, 0.02, n_samples)
theta_samples = np.random.normal(0.1, 0.01, n_samples)
T_samples = 2 * np.pi * np.sqrt(L_samples / g_samples) * (1 + theta_samples**2 / 16)

plt.figure(figsize=(10, 6))
plt.hist(T_samples, bins=50, density=True, alpha=0.7, label='Monte Carlo samples')
plt.axvline(period_result.numeric_result.value, color='red', linestyle='--', 
            label=f'Mean: {period_result.numeric_result.value:.4f}')
plt.axvline(period_result.numeric_result.value - period_result.numeric_result.uncertainty, 
            color='orange', linestyle=':', label='±1σ')
plt.axvline(period_result.numeric_result.value + period_result.numeric_result.uncertainty, 
            color='orange', linestyle=':')
plt.xlabel('Period (s)')
plt.ylabel('Probability Density')
plt.title('Distribution of Pendulum Period with Uncertainties')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('examples/uncertainty_propagation_distribution.png', dpi=150)
plt.show()

# Example 6: Engineering Application - Beam Deflection
print("\n\nExample 6: Cantilever Beam Deflection")
print("-" * 40)

# Deflection δ = FL³/(3EI) for cantilever beam
variables = {
    'F': (100.0, 2.0),      # Force: 100 ± 2 N
    'L': (2.0, 0.01),       # Length: 2.0 ± 0.01 m
    'E': (200e9, 5e9),      # Young's modulus: 200 ± 5 GPa
    'I': (8.33e-6, 0.1e-6)  # Second moment: 8.33 ± 0.1 ×10⁻⁶ m⁴
}

deflection_result = propagate_uncertainties_symbolic(
    'F * L**3 / (3 * E * I)',
    variables
)

print(f"Beam deflection: {deflection_result.numeric_result.value*1000:.2f} ± {deflection_result.numeric_result.uncertainty*1000:.2f} mm")
print(f"\nMost significant uncertainty sources:")
sorted_contrib = sorted(deflection_result.relative_contributions.items(), 
                       key=lambda x: x[1], reverse=True)
for var, contrib in sorted_contrib:
    print(f"  {var}: {contrib*100:.1f}%")

print("\n" + "="*50)
print("Uncertainty Propagation Examples Complete!")
print("\nKey Takeaways:")
print("1. Symbolic propagation preserves mathematical relationships")
print("2. Correlations between variables can significantly affect results")
print("3. Monte Carlo methods capture non-linear effects better than Taylor")
print("4. Sensitivity analysis identifies which measurements to improve")
print("5. Different methods agree well for small uncertainties")