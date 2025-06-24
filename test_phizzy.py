#!/usr/bin/env python
"""
Quick test script to verify Phizzy library functionality
"""

import numpy as np
from phizzy import symplectic_integrate, propagate_uncertainties_symbolic, unit_aware_fft
import pint

print("Testing Phizzy Library")
print("=" * 50)

# Test 1: Symplectic Integration
print("\n1. Testing Symplectic Integration...")
try:
    def harmonic_oscillator(q, p):
        return 0.5 * (p**2 + q**2)
    
    result = symplectic_integrate(
        harmonic_oscillator,
        q0=[1.0], p0=[0.0],
        t_span=(0, 2*np.pi),
        dt=0.01
    )
    print(f"✓ Symplectic integration works!")
    print(f"  Energy conservation: {result.energy_error:.2e}")
except Exception as e:
    print(f"✗ Symplectic integration failed: {e}")

# Test 2: Uncertainty Propagation
print("\n2. Testing Uncertainty Propagation...")
try:
    variables = {'x': (10.0, 0.1), 'y': (5.0, 0.05)}
    result = propagate_uncertainties_symbolic('x * y', variables)
    print(f"✓ Uncertainty propagation works!")
    print(f"  Result: {result.numeric_result}")
except Exception as e:
    print(f"✗ Uncertainty propagation failed: {e}")

# Test 3: Unit-Aware FFT
print("\n3. Testing Unit-Aware FFT...")
try:
    ureg = pint.UnitRegistry()
    fs = ureg.Quantity(1000, 'Hz')
    t = np.linspace(0, 1, 1000) * ureg.second
    signal = 2.0 * ureg.volt * np.sin(2 * np.pi * 50 * t.magnitude)
    
    result = unit_aware_fft(signal, sampling_rate=fs)
    print(f"✓ Unit-aware FFT works!")
    print(f"  Dominant frequency: {result.dominant_frequency()}")
except Exception as e:
    print(f"✗ Unit-aware FFT failed: {e}")

# Test unimplemented functions
print("\n4. Testing stub functions...")
unimplemented = [
    ('field_visualization_3d', lambda: __import__('phizzy').field_visualization_3d()),
    ('monte_carlo_physics', lambda: __import__('phizzy').monte_carlo_physics()),
    ('tensor_contract_optimize', lambda: __import__('phizzy').tensor_contract_optimize()),
    ('phase_transition_detect', lambda: __import__('phizzy').phase_transition_detect()),
    ('coupled_pde_solve', lambda: __import__('phizzy').coupled_pde_solve()),
    ('lagrangian_to_equations', lambda: __import__('phizzy').lagrangian_to_equations()),
    ('experimental_fit_physics', lambda: __import__('phizzy').experimental_fit_physics()),
]

for name, func in unimplemented:
    try:
        func()
        print(f"✗ {name} should raise NotImplementedError")
    except NotImplementedError:
        print(f"✓ {name} correctly raises NotImplementedError")
    except Exception as e:
        print(f"✗ {name} raised unexpected error: {e}")

print("\n" + "=" * 50)
print("Testing complete!")
print("\nImplemented functions:")
print("- symplectic_integrate")
print("- propagate_uncertainties_symbolic") 
print("- unit_aware_fft")
print("\nPending implementation:")
print("- field_visualization_3d")
print("- monte_carlo_physics")
print("- tensor_contract_optimize")
print("- phase_transition_detect")
print("- coupled_pde_solve")
print("- lagrangian_to_equations")
print("- experimental_fit_physics")