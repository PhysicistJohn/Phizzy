import numpy as np
import pint
from typing import Union, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from scipy import fft as sp_fft


# Initialize unit registry
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


@dataclass
class FFTResult:
    """Result of unit-aware FFT"""
    frequencies: pint.Quantity
    spectrum: pint.Quantity
    phase: pint.Quantity
    power_spectrum: pint.Quantity
    info: Dict[str, Any]
    
    def dominant_frequency(self) -> pint.Quantity:
        """Get the dominant frequency from the spectrum"""
        idx = np.argmax(np.abs(self.spectrum[1:len(self.spectrum)//2])) + 1
        return self.frequencies[idx]
    
    def get_component(self, frequency: pint.Quantity, tolerance: pint.Quantity = None) -> pint.Quantity:
        """Extract component at specific frequency"""
        if tolerance is None:
            tolerance = (self.frequencies[1] - self.frequencies[0]) / 2
        
        freq_val = frequency.to(self.frequencies.units).magnitude
        freq_array = self.frequencies.magnitude
        tol_val = tolerance.to(self.frequencies.units).magnitude
        
        mask = np.abs(freq_array - freq_val) < tol_val
        if not np.any(mask):
            raise ValueError(f"No frequency found near {frequency}")
        
        return self.spectrum[mask][0]


def unit_aware_fft(
    signal: Union[pint.Quantity, np.ndarray],
    time: Union[pint.Quantity, np.ndarray] = None,
    sampling_rate: Union[pint.Quantity, float] = None,
    dt: Union[pint.Quantity, float] = None,
    window: Optional[str] = None,
    detrend: str = 'constant',
    scaling: str = 'amplitude',
    axis: int = -1,
    return_onesided: bool = True
) -> FFTResult:
    """
    Fast Fourier Transform with proper unit handling and automatic unit conversion.
    
    Parameters
    ----------
    signal : pint.Quantity or array
        Input signal with units (e.g., volts, meters, etc.)
    time : pint.Quantity or array, optional
        Time array with units. Required if sampling_rate and dt not provided
    sampling_rate : pint.Quantity or float, optional
        Sampling frequency with units (e.g., Hz, kHz)
    dt : pint.Quantity or float, optional
        Time step between samples
    window : str, optional
        Window function: 'hann', 'hamming', 'blackman', 'bartlett', 'kaiser'
    detrend : str
        Detrending method: 'constant', 'linear', 'none'
    scaling : str
        Scaling mode: 'amplitude', 'power', 'psd' (power spectral density)
    axis : int
        Axis along which to compute FFT
    return_onesided : bool
        Return one-sided spectrum for real signals
        
    Returns
    -------
    FFTResult
        Object with frequency array and spectrum, both with proper units
    """
    
    # Handle units
    if not isinstance(signal, pint.Quantity):
        signal = Q_(signal, 'dimensionless')
    
    # Determine sampling information
    if time is not None:
        if not isinstance(time, pint.Quantity):
            time = Q_(time, 'second')
        dt = time[1] - time[0]
        sampling_rate = 1 / dt
        n_samples = len(time)
    elif sampling_rate is not None:
        if not isinstance(sampling_rate, pint.Quantity):
            sampling_rate = Q_(sampling_rate, 'hertz')
        dt = 1 / sampling_rate
        n_samples = signal.shape[axis]
    elif dt is not None:
        if not isinstance(dt, pint.Quantity):
            dt = Q_(dt, 'second')
        sampling_rate = 1 / dt
        n_samples = signal.shape[axis]
    else:
        raise ValueError("Must provide either time array, sampling_rate, or dt")
    
    # Ensure we have consistent units
    dt = dt.to('second')
    sampling_rate = sampling_rate.to('hertz')
    
    # Apply detrending
    sig_values = signal.magnitude
    if detrend == 'constant':
        sig_values = sig_values - np.mean(sig_values, axis=axis, keepdims=True)
    elif detrend == 'linear':
        from scipy import signal as sp_signal
        sig_values = sp_signal.detrend(sig_values, axis=axis, type='linear')
    
    # Apply window function
    if window is not None:
        from scipy import signal as sp_signal
        if window == 'hann':
            w = sp_signal.hann(n_samples)
        elif window == 'hamming':
            w = sp_signal.hamming(n_samples)
        elif window == 'blackman':
            w = sp_signal.blackman(n_samples)
        elif window == 'bartlett':
            w = sp_signal.bartlett(n_samples)
        elif window == 'kaiser':
            w = sp_signal.kaiser(n_samples, beta=8.6)
        else:
            raise ValueError(f"Unknown window: {window}")
        
        # Apply window
        if axis == -1:
            sig_values = sig_values * w
        else:
            w_shape = [1] * sig_values.ndim
            w_shape[axis] = n_samples
            sig_values = sig_values * w.reshape(w_shape)
        
        # Window correction factors
        amplitude_correction = np.sum(w) / n_samples
        power_correction = np.sqrt(np.sum(w**2) / n_samples)
    else:
        amplitude_correction = 1.0
        power_correction = 1.0
    
    # Compute FFT
    spectrum_values = sp_fft.fft(sig_values, axis=axis)
    freqs = sp_fft.fftfreq(n_samples, dt.magnitude)
    
    # Handle one-sided spectrum for real signals
    if return_onesided and np.isrealobj(sig_values):
        # Keep only positive frequencies
        if axis == -1:
            spectrum_values = spectrum_values[:n_samples//2 + 1]
            freqs = freqs[:n_samples//2 + 1]
        else:
            slices = [slice(None)] * spectrum_values.ndim
            slices[axis] = slice(None, n_samples//2 + 1)
            spectrum_values = spectrum_values[tuple(slices)]
            freqs = freqs[:n_samples//2 + 1]
        
        # Double the amplitude for positive frequencies (except DC and Nyquist)
        spectrum_values[1:-1] *= 2
    
    # Apply scaling
    if scaling == 'amplitude':
        spectrum_values = spectrum_values / (n_samples * amplitude_correction)
    elif scaling == 'power':
        spectrum_values = np.abs(spectrum_values)**2 / (n_samples**2 * power_correction**2)
        if return_onesided:
            spectrum_values[1:-1] *= 2  # Account for negative frequencies
    elif scaling == 'psd':
        # Power spectral density
        spectrum_values = np.abs(spectrum_values)**2 / (sampling_rate.magnitude * n_samples * power_correction**2)
        if return_onesided:
            spectrum_values[1:-1] *= 2
    
    # Determine units for spectrum
    freq_units = 1 / dt.units
    
    if scaling == 'amplitude':
        spectrum_units = signal.units
    elif scaling == 'power':
        spectrum_units = signal.units ** 2
    elif scaling == 'psd':
        spectrum_units = signal.units ** 2 * dt.units
    
    # Create result quantities
    frequencies = Q_(freqs, freq_units)
    spectrum = Q_(spectrum_values, spectrum_units)
    
    # Compute phase
    phase_values = np.angle(spectrum_values) if scaling == 'amplitude' else np.zeros_like(spectrum_values)
    phase = Q_(phase_values, 'radian')
    
    # Compute power spectrum
    if scaling == 'amplitude':
        power_values = np.abs(spectrum_values)**2
        power_spectrum = Q_(power_values, signal.units ** 2)
    else:
        power_spectrum = spectrum
    
    return FFTResult(
        frequencies=frequencies,
        spectrum=spectrum,
        phase=phase,
        power_spectrum=power_spectrum,
        info={
            'sampling_rate': sampling_rate,
            'dt': dt,
            'n_samples': n_samples,
            'window': window,
            'detrend': detrend,
            'scaling': scaling,
            'onesided': return_onesided,
            'amplitude_correction': amplitude_correction,
            'power_correction': power_correction,
            'nyquist_frequency': sampling_rate / 2,
            'frequency_resolution': Q_(freqs[1] - freqs[0], freq_units) if len(freqs) > 1 else None
        }
    )


def inverse_unit_aware_fft(
    fft_result: FFTResult,
    n_samples: Optional[int] = None
) -> Tuple[pint.Quantity, pint.Quantity]:
    """
    Inverse FFT with proper unit handling.
    
    Parameters
    ----------
    fft_result : FFTResult
        Result from unit_aware_fft
    n_samples : int, optional
        Number of samples for output signal
        
    Returns
    -------
    time : pint.Quantity
        Time array with proper units
    signal : pint.Quantity
        Reconstructed signal with proper units
    """
    
    # Get spectrum values
    spectrum_values = fft_result.spectrum.magnitude
    
    # Handle scaling
    scaling = fft_result.info['scaling']
    n_fft = fft_result.info['n_samples']
    
    if scaling == 'amplitude':
        # Reverse amplitude scaling
        spectrum_values = spectrum_values * n_fft * fft_result.info['amplitude_correction']
    elif scaling in ['power', 'psd']:
        # Can't directly invert power spectrum - need phase information
        raise ValueError(f"Cannot invert {scaling} spectrum without phase information")
    
    # Handle one-sided spectrum
    if fft_result.info['onesided']:
        # Reconstruct full spectrum
        if len(spectrum_values.shape) == 1:
            # Create negative frequencies
            full_spectrum = np.zeros(n_fft, dtype=complex)
            full_spectrum[:len(spectrum_values)] = spectrum_values
            # Mirror for negative frequencies (except DC and Nyquist)
            full_spectrum[n_fft-len(spectrum_values)+2:] = np.conj(spectrum_values[1:-1][::-1])
        else:
            raise NotImplementedError("Multi-dimensional one-sided spectrum not supported")
        spectrum_values = full_spectrum
    
    # Compute inverse FFT
    signal_values = sp_fft.ifft(spectrum_values)
    
    # Handle real signals
    if fft_result.info['onesided']:
        signal_values = np.real(signal_values)
    
    # Create time array
    dt = fft_result.info['dt']
    if n_samples is None:
        n_samples = len(signal_values)
    time_values = np.arange(n_samples) * dt.magnitude
    time = Q_(time_values, dt.units)
    
    # Determine signal units
    if scaling == 'amplitude':
        signal_units = fft_result.spectrum.units
    else:
        # This shouldn't happen due to error above
        signal_units = 'dimensionless'
    
    signal = Q_(signal_values[:n_samples], signal_units)
    
    return time, signal