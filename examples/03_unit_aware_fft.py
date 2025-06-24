"""
Example: Unit-Aware Fast Fourier Transform
=========================================

This example demonstrates FFT with automatic unit tracking and conversion,
showing various applications in signal processing and physics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pint
from phizzy import unit_aware_fft
from phizzy.transforms.unit_fft import inverse_unit_aware_fft

# Initialize unit registry
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

print("Unit-Aware FFT Examples")
print("=" * 50)

# Example 1: Basic Signal Analysis
print("\nExample 1: Analyzing a Voltage Signal")
print("-" * 40)

# Create a signal with physical units
fs = Q_(1000, 'Hz')  # 1 kHz sampling rate
duration = Q_(1, 'second')
t = np.arange(0, duration.magnitude, 1/fs.magnitude) * ureg.second

# Composite signal: 50 Hz power line + 200 Hz signal
signal = (Q_(5.0, 'volt') * np.sin(2 * np.pi * 50 * t.magnitude) +
          Q_(2.0, 'volt') * np.sin(2 * np.pi * 200 * t.magnitude + np.pi/4))

# Perform FFT
result = unit_aware_fft(signal, sampling_rate=fs)

print(f"Signal units: {signal.units}")
print(f"Frequency units: {result.frequencies.units}")
print(f"Spectrum units: {result.spectrum.units}")
print(f"Frequency resolution: {result.info['frequency_resolution']}")

# Find peaks
peaks = []
for f in [50, 200]:
    component = result.get_component(Q_(f, 'Hz'), tolerance=Q_(5, 'Hz'))
    amplitude = np.abs(component)
    phase = np.angle(component)
    peaks.append((f, amplitude, phase))
    print(f"\n{f} Hz component:")
    print(f"  Amplitude: {amplitude:.3f}")
    print(f"  Phase: {phase:.3f} rad ({np.degrees(phase):.1f}°)")

# Example 2: Different Scaling Modes
print("\n\nExample 2: FFT Scaling Modes")
print("-" * 40)

# Generate test signal with known power
fs = Q_(10, 'kHz')  # 10 kHz sampling
t = np.arange(0, 1, 1/fs.to('Hz').magnitude) * ureg.second
amplitude = Q_(3.0, 'ampere')
freq = Q_(1000, 'Hz')
signal = amplitude * np.sin(2 * np.pi * freq * t.magnitude)

# Different scaling modes
scalings = ['amplitude', 'power', 'psd']
results = {}

for scaling in scalings:
    result = unit_aware_fft(signal, sampling_rate=fs, scaling=scaling)
    results[scaling] = result
    
    # Get value at 1 kHz
    value = result.get_component(freq)
    print(f"\n{scaling.upper()} scaling:")
    print(f"  Units: {result.spectrum.units}")
    print(f"  Value at {freq}: {np.abs(value):.3f}")

# For sine wave: RMS = A/√2, Power = A²/2
print(f"\nExpected values:")
print(f"  Amplitude: {amplitude}")
print(f"  RMS Power: {(amplitude**2 / 2).to('A**2')}")

# Example 3: Window Functions
print("\n\nExample 3: Window Functions for Spectral Leakage")
print("-" * 40)

# Create signal with non-integer number of periods (causes leakage)
fs = Q_(1000, 'Hz')
t = np.arange(0, 1, 1/fs.magnitude) * ureg.second
# 100.7 Hz - not exactly aligned with FFT bins
signal = Q_(1.0, 'meter') * np.sin(2 * np.pi * 100.7 * t.magnitude)

windows = ['none', 'hann', 'hamming', 'blackman']
plt.figure(figsize=(12, 8))

for i, window in enumerate(windows):
    if window == 'none':
        result = unit_aware_fft(signal, sampling_rate=fs, window=None)
    else:
        result = unit_aware_fft(signal, sampling_rate=fs, window=window)
    
    # Plot spectrum around the peak
    freq_mask = (result.frequencies.magnitude > 80) & (result.frequencies.magnitude < 120)
    
    plt.subplot(2, 2, i+1)
    plt.semilogy(result.frequencies[freq_mask].magnitude, 
                 np.abs(result.spectrum[freq_mask]).magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(f'Amplitude ({result.spectrum.units})')
    plt.title(f'Window: {window if window else "None (Rectangular)"}')
    plt.grid(True, alpha=0.3)
    plt.ylim(1e-4, 1)

plt.suptitle('Effect of Window Functions on Spectral Leakage')
plt.tight_layout()
plt.savefig('examples/fft_window_comparison.png', dpi=150)
plt.show()

# Example 4: Mechanical Vibration Analysis
print("\n\nExample 4: Mechanical Vibration Analysis")
print("-" * 40)

# Simulate accelerometer data from rotating machinery
fs = Q_(5000, 'Hz')  # 5 kHz sampling
t = np.arange(0, 2, 1/fs.magnitude) * ureg.second

# Machine running at 1800 RPM = 30 Hz
fundamental = Q_(30, 'Hz')

# Vibration components:
# - Fundamental (30 Hz)
# - 2nd harmonic (60 Hz) - misalignment
# - 3rd harmonic (90 Hz) - looseness
# - High frequency (500 Hz) - bearing defect
# - Noise

acceleration = (Q_(1.0, 'm/s**2') * np.sin(2 * np.pi * 30 * t.magnitude) +
                Q_(0.3, 'm/s**2') * np.sin(2 * np.pi * 60 * t.magnitude) +
                Q_(0.2, 'm/s**2') * np.sin(2 * np.pi * 90 * t.magnitude) +
                Q_(0.1, 'm/s**2') * np.sin(2 * np.pi * 500 * t.magnitude) +
                Q_(0.05, 'm/s**2') * np.random.normal(0, 1, len(t)))

# Compute PSD for vibration analysis
psd_result = unit_aware_fft(acceleration, sampling_rate=fs, 
                           scaling='psd', window='hann')

print(f"Acceleration units: {acceleration.units}")
print(f"PSD units: {psd_result.spectrum.units}")

# Analyze harmonics
harmonics = [1, 2, 3]
print("\nHarmonic analysis:")
for n in harmonics:
    freq = n * fundamental
    try:
        value = psd_result.get_component(freq, tolerance=Q_(2, 'Hz'))
        print(f"  {n}× fundamental ({freq}): {np.sqrt(np.abs(value)):.3f}")
    except ValueError:
        print(f"  {n}× fundamental ({freq}): Not found")

# Example 5: FFT and Inverse FFT
print("\n\nExample 5: FFT and Inverse FFT Reconstruction")
print("-" * 40)

# Original signal
fs = Q_(1000, 'Hz')
t_orig = np.arange(0, 0.5, 1/fs.magnitude) * ureg.second
signal_orig = (Q_(2.0, 'volt') * np.sin(2 * np.pi * 100 * t_orig.magnitude) +
               Q_(1.0, 'volt') * np.sin(2 * np.pi * 250 * t_orig.magnitude))

# Forward FFT
fft_result = unit_aware_fft(signal_orig, sampling_rate=fs)

# Inverse FFT
t_recon, signal_recon = inverse_unit_aware_fft(fft_result)

# Check reconstruction error
error = np.max(np.abs(signal_orig.magnitude - signal_recon.magnitude))
print(f"Reconstruction error: {error:.2e} {signal_orig.units}")
print(f"Original signal units: {signal_orig.units}")
print(f"Reconstructed signal units: {signal_recon.units}")

# Plot comparison
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_orig.magnitude[:100], signal_orig.magnitude[:100], 'b-', 
         label='Original', linewidth=2)
plt.plot(t_recon.magnitude[:100], signal_recon.magnitude[:100], 'r--', 
         label='Reconstructed', linewidth=2)
plt.xlabel(f'Time ({t_orig.units})')
plt.ylabel(f'Signal ({signal_orig.units})')
plt.title('Signal Reconstruction via Inverse FFT')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.semilogy(t_orig.magnitude[:100], 
            np.abs(signal_orig.magnitude[:100] - signal_recon.magnitude[:100]))
plt.xlabel(f'Time ({t_orig.units})')
plt.ylabel('Absolute Error')
plt.title('Reconstruction Error')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('examples/fft_reconstruction.png', dpi=150)
plt.show()

# Example 6: Cross-Unit Analysis
print("\n\nExample 6: Cross-Unit Analysis")
print("-" * 40)

# Optical signal in different units
wavelength = Q_(632.8, 'nanometer')  # He-Ne laser
c = Q_(3e8, 'm/s')  # Speed of light
optical_freq = (c / wavelength).to('THz')

print(f"Optical signal:")
print(f"  Wavelength: {wavelength}")
print(f"  Frequency: {optical_freq}")

# Electric field oscillation (simplified)
fs = Q_(10, 'PHz')  # Petahertz sampling (theoretical)
t = np.arange(0, 100, 1/fs.to('Hz').magnitude) * ureg.second
E_field = Q_(1e6, 'V/m') * np.sin(2 * np.pi * optical_freq.to('Hz') * t.magnitude)

# This would work with proper units!
print(f"\nElectric field units: {E_field.units}")
print(f"Sampling rate: {fs}")

print("\n" + "="*50)
print("Unit-Aware FFT Examples Complete!")
print("\nKey Takeaways:")
print("1. Automatic unit tracking prevents errors")
print("2. Different scaling modes for different applications")
print("3. Window functions reduce spectral leakage")
print("4. Works seamlessly with any unit system")
print("5. Preserves dimensional consistency throughout analysis")