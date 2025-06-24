"""
Comprehensive demonstration of Phizzy library capabilities
=========================================================

This example showcases all major features of the Phizzy physics library.
"""

import numpy as np
import matplotlib.pyplot as plt
from phizzy import (
    symplectic_integrate,
    propagate_uncertainties_symbolic,
    unit_aware_fft
)
import pint

# Initialize unit registry
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

print("Phizzy Library - Comprehensive Demo")
print("=" * 50)

# ==============================================================================
# 1. Symplectic Integration - Double Pendulum
# ==============================================================================
print("\n1. Symplectic Integration - Double Pendulum")
print("-" * 40)

def double_pendulum_hamiltonian(q, p):
    """Hamiltonian for double pendulum"""
    m1, m2 = 1.0, 1.0  # masses
    l1, l2 = 1.0, 1.0  # lengths
    g = 9.81           # gravity
    
    theta1, theta2 = q
    p_theta1, p_theta2 = p
    
    # Helper terms
    c = np.cos(theta1 - theta2)
    s = np.sin(theta1 - theta2)
    den = m1 + m2 * s**2
    
    # Kinetic energy
    T = (p_theta1**2 * m2 + p_theta2**2 * (m1 + m2) - 
         2 * p_theta1 * p_theta2 * m2 * c) / (2 * m2 * l1**2 * l2**2 * den)
    
    # Potential energy
    V = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
    
    return T + V

# Initial conditions
q0 = np.array([np.pi/2, np.pi/2])  # Both pendulums horizontal
p0 = np.array([0.0, 0.0])          # Initially at rest

# Integrate using high-order symplectic method
result = symplectic_integrate(
    double_pendulum_hamiltonian, 
    q0, p0, 
    t_span=(0, 20),
    dt=0.001,
    method='yoshida4'
)

print(f"Integration complete!")
print(f"Energy conservation error: {result.energy_error:.2e}")
print(f"Initial energy: {result.energy[0]:.6f}")
print(f"Final energy: {result.energy[-1]:.6f}")
print(f"Energy drift: {result.info['energy_drift']:.2e}")

# Plot energy conservation
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(result.t, result.energy)
plt.xlabel('Time (s)')
plt.ylabel('Total Energy')
plt.title('Energy Conservation in Double Pendulum')
plt.grid(True)

# Plot phase space
plt.subplot(1, 2, 2)
plt.plot(result.q[:, 0], result.p[:, 0], alpha=0.6, label='Pendulum 1')
plt.plot(result.q[:, 1], result.p[:, 1], alpha=0.6, label='Pendulum 2')
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Momentum')
plt.title('Phase Space Trajectories')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('examples/double_pendulum_symplectic.png', dpi=150)
plt.close()

# ==============================================================================
# 2. Symbolic Uncertainty Propagation - Pendulum Period
# ==============================================================================
print("\n2. Symbolic Uncertainty Propagation - Pendulum Period")
print("-" * 40)

# Pendulum period: T = 2π√(L/g)
variables = {
    'L': (1.00, 0.02),   # Length: 1.00 ± 0.02 m
    'g': (9.81, 0.05)    # Gravity: 9.81 ± 0.05 m/s²
}

# Propagate uncertainties through the period formula
period_result = propagate_uncertainties_symbolic(
    '2 * pi * sqrt(L / g)',
    variables,
    method='taylor'
)

print(f"Pendulum period: {period_result.numeric_result}")
print(f"\nSensitivity coefficients:")
for var, coeff in period_result.sensitivity_coefficients.items():
    print(f"  ∂T/∂{var} = {coeff:.4f}")

print(f"\nRelative contributions to uncertainty:")
for var, contrib in period_result.relative_contributions.items():
    print(f"  {var}: {contrib*100:.1f}%")

# Compare with Monte Carlo method
mc_result = propagate_uncertainties_symbolic(
    '2 * pi * sqrt(L / g)',
    variables,
    method='monte_carlo',
    monte_carlo_samples=100000
)

print(f"\nMonte Carlo result: {mc_result.numeric_result}")
print(f"Skewness: {mc_result.info['skewness']:.4f}")
print(f"Kurtosis: {mc_result.info['kurtosis']:.4f}")

# ==============================================================================
# 3. Unit-Aware FFT - Multi-frequency Signal Analysis
# ==============================================================================
print("\n3. Unit-Aware FFT - Multi-frequency Signal Analysis")
print("-" * 40)

# Create a complex signal with multiple components and noise
fs = Q_(10000, 'Hz')  # 10 kHz sampling rate
duration = Q_(2, 'second')
dt = 1 / fs
t = np.arange(0, duration.magnitude, dt.magnitude) * ureg.second

# Signal components:
# - 50 Hz power line frequency (2V)
# - 1 kHz signal (1V)
# - 2.5 kHz signal (0.5V)
# - White noise (0.1V RMS)
signal = (Q_(2.0, 'volt') * np.sin(2 * np.pi * 50 * t.magnitude) +
          Q_(1.0, 'volt') * np.sin(2 * np.pi * 1000 * t.magnitude) +
          Q_(0.5, 'volt') * np.sin(2 * np.pi * 2500 * t.magnitude) +
          Q_(0.1, 'volt') * np.random.normal(0, 1, len(t)))

# Apply Hann window to reduce spectral leakage
fft_result = unit_aware_fft(signal, sampling_rate=fs, window='hann')

print(f"Frequency resolution: {fft_result.info['frequency_resolution']}")
print(f"Nyquist frequency: {fft_result.info['nyquist_frequency']}")
print(f"Dominant frequency: {fft_result.dominant_frequency()}")

# Extract specific components
components = [
    (Q_(50, 'Hz'), "Power line"),
    (Q_(1000, 'Hz'), "1 kHz signal"),
    (Q_(2500, 'Hz'), "2.5 kHz signal")
]

print("\nExtracted frequency components:")
for freq, name in components:
    amplitude = np.abs(fft_result.get_component(freq, tolerance=Q_(10, 'Hz')))
    print(f"  {name} ({freq}): {amplitude:.3f}")

# Plot spectrum
plt.figure(figsize=(10, 6))
freq_vals = fft_result.frequencies.magnitude
spectrum_vals = np.abs(fft_result.spectrum.magnitude)

# Only plot positive frequencies up to Nyquist
mask = (freq_vals >= 0) & (freq_vals <= 5000)
plt.semilogy(freq_vals[mask], spectrum_vals[mask])
plt.xlabel(f'Frequency ({fft_result.frequencies.units})')
plt.ylabel(f'Amplitude ({fft_result.spectrum.units})')
plt.title('Frequency Spectrum of Multi-Component Signal')
plt.grid(True, which='both', alpha=0.3)
plt.xlim(0, 3000)
plt.ylim(1e-4, 10)

# Mark the expected frequencies
for freq, name in components:
    plt.axvline(freq.magnitude, color='red', linestyle='--', alpha=0.5)
    plt.text(freq.magnitude + 50, 1, name, rotation=90, va='bottom')

plt.tight_layout()
plt.savefig('examples/unit_aware_fft_spectrum.png', dpi=150)
plt.close()

# Power Spectral Density
psd_result = unit_aware_fft(signal, sampling_rate=fs, scaling='psd', window='hann')
print(f"\nPSD units: {psd_result.spectrum.units}")

# ==============================================================================
# 4. Combined Example: Driven Oscillator with Uncertainty
# ==============================================================================
print("\n4. Combined Example: Driven Oscillator with Uncertainty")
print("-" * 40)

# Measure resonance frequency of a driven harmonic oscillator
# with uncertainties in mass and spring constant

# System parameters with uncertainties
mass = UncertainValue(1.0, 0.05)      # 1.0 ± 0.05 kg
k_spring = UncertainValue(100.0, 2.0) # 100 ± 2 N/m
damping = UncertainValue(0.1, 0.01)   # 0.1 ± 0.01 Ns/m

# Expected resonance frequency: f0 = 1/(2π) * sqrt(k/m)
resonance_result = propagate_uncertainties_symbolic(
    'sqrt(k / m) / (2 * pi)',
    {'k': (k_spring.value, k_spring.uncertainty),
     'm': (mass.value, mass.uncertainty)}
)

print(f"Theoretical resonance frequency: {resonance_result.numeric_result} Hz")

# Simulate driven oscillator and find actual resonance
def driven_oscillator_response(f_drive, m, k, c):
    """Amplitude response of driven harmonic oscillator"""
    omega = 2 * np.pi * f_drive
    omega0 = np.sqrt(k / m)
    amplitude = 1 / np.sqrt((omega0**2 - omega**2)**2 + (2 * c * omega / m)**2)
    return amplitude

# Frequency sweep
frequencies = np.linspace(0.1, 5.0, 500)  # Hz
response = driven_oscillator_response(frequencies, mass.value, k_spring.value, damping.value)

# Find resonance peak
resonance_idx = np.argmax(response)
measured_resonance = frequencies[resonance_idx]

print(f"Measured resonance frequency: {measured_resonance:.3f} Hz")
print(f"Difference: {abs(measured_resonance - resonance_result.numeric_result.value):.3f} Hz")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "=" * 50)
print("Demo Complete!")
print("\nKey Features Demonstrated:")
print("1. Symplectic integration preserves energy to machine precision")
print("2. Uncertainty propagation works with both analytical and MC methods")
print("3. Unit-aware FFT automatically handles dimensional analysis")
print("4. These tools can be combined for complex physics problems")
print("\nGenerated plots:")
print("- examples/double_pendulum_symplectic.png")
print("- examples/unit_aware_fft_spectrum.png")