import pytest
import numpy as np
import pint
from phizzy.transforms import unit_aware_fft
from phizzy.transforms.unit_fft import inverse_unit_aware_fft


class TestUnitAwareFFT:
    
    def setup_method(self):
        """Set up unit registry for tests"""
        self.ureg = pint.UnitRegistry()
        self.Q_ = self.ureg.Quantity
    
    def test_simple_sine_wave(self):
        """Test FFT of simple sine wave with units"""
        # Create 1 kHz sine wave sampled at 10 kHz
        fs = self.Q_(10000, 'Hz')
        duration = self.Q_(1, 'second')
        t = np.arange(0, duration.magnitude, 1/fs.magnitude) * self.ureg.second
        
        frequency = self.Q_(1000, 'Hz')
        amplitude = self.Q_(2.5, 'volt')
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Compute FFT
        result = unit_aware_fft(signal, sampling_rate=fs)
        
        # Check units
        assert result.frequencies.units == self.ureg.Hz
        assert result.spectrum.units == self.ureg.volt
        
        # Check dominant frequency
        dominant_freq = result.dominant_frequency()
        assert abs(dominant_freq.magnitude - 1000) < 10  # Within 10 Hz
        
        # Check amplitude at 1 kHz
        component = result.get_component(frequency, tolerance=self.Q_(10, 'Hz'))
        assert abs(np.abs(component).magnitude - amplitude.magnitude) < 0.1
    
    def test_multiple_frequencies(self):
        """Test FFT with multiple frequency components"""
        # Create signal with 3 components
        dt = self.Q_(0.001, 'second')  # 1 ms sampling
        t = np.arange(0, 1, dt.magnitude) * self.ureg.second
        
        # 50 Hz at 1V, 120 Hz at 0.5V, 200 Hz at 0.3V
        signal = (self.Q_(1.0, 'V') * np.sin(2 * np.pi * 50 * t.magnitude) +
                 self.Q_(0.5, 'V') * np.sin(2 * np.pi * 120 * t.magnitude) +
                 self.Q_(0.3, 'V') * np.sin(2 * np.pi * 200 * t.magnitude))
        
        result = unit_aware_fft(signal, dt=dt)
        
        # Check each frequency component
        freqs_to_check = [(50, 1.0), (120, 0.5), (200, 0.3)]
        for freq, expected_amp in freqs_to_check:
            component = result.get_component(self.Q_(freq, 'Hz'), tolerance=self.Q_(5, 'Hz'))
            assert abs(np.abs(component).magnitude - expected_amp) < 0.05
    
    def test_power_spectrum(self):
        """Test power spectrum calculation"""
        # Simple signal
        fs = self.Q_(1000, 'Hz')
        t = np.arange(0, 1, 1/fs.magnitude) * self.ureg.second
        signal = self.Q_(2.0, 'ampere') * np.sin(2 * np.pi * 100 * t.magnitude)
        
        # Power scaling
        result = unit_aware_fft(signal, sampling_rate=fs, scaling='power')
        
        # Check units
        assert result.spectrum.units == self.ureg.ampere ** 2
        
        # Check power at 100 Hz (should be A²/2 for sine wave)
        component = result.get_component(self.Q_(100, 'Hz'))
        expected_power = (2.0**2) / 2  # RMS power
        assert abs(component.magnitude - expected_power) < 0.1
    
    def test_psd_scaling(self):
        """Test power spectral density calculation"""
        # White noise signal
        fs = self.Q_(1000, 'Hz')
        duration = 10  # seconds
        t = np.arange(0, duration, 1/fs.magnitude) * self.ureg.second
        
        # Generate white noise with known variance
        noise_power = 1.0  # V²
        signal = self.Q_(np.random.normal(0, np.sqrt(noise_power), len(t)), 'volt')
        
        result = unit_aware_fft(signal, sampling_rate=fs, scaling='psd')
        
        # Check units (V²/Hz)
        assert result.spectrum.units == self.ureg.volt**2 * self.ureg.second
        
        # For white noise, PSD should be approximately flat
        # Average PSD should equal variance / (fs/2) for one-sided spectrum
        psd_avg = np.mean(np.abs(result.spectrum[1:-1]).magnitude)
        expected_psd = 2 * noise_power / fs.magnitude  # Factor of 2 for one-sided
        assert abs(psd_avg - expected_psd) / expected_psd < 0.2  # Within 20%
    
    def test_window_functions(self):
        """Test different window functions"""
        # Signal with spectral leakage
        fs = self.Q_(1000, 'Hz')
        t = np.arange(0, 1, 1/fs.magnitude) * self.ureg.second
        # Non-integer number of periods
        signal = self.Q_(1.0, 'volt') * np.sin(2 * np.pi * 100.5 * t.magnitude)
        
        windows = ['hann', 'hamming', 'blackman', 'bartlett', 'kaiser']
        
        for window in windows:
            result = unit_aware_fft(signal, sampling_rate=fs, window=window)
            
            # Window should reduce spectral leakage
            # Peak should still be near 100.5 Hz
            peak_freq = result.dominant_frequency()
            assert 95 < peak_freq.magnitude < 106
    
    def test_detrending(self):
        """Test detrending options"""
        fs = self.Q_(100, 'Hz')
        t = np.arange(0, 10, 1/fs.magnitude) * self.ureg.second
        
        # Signal with DC offset and linear trend
        signal = (self.Q_(5.0, 'volt') +  # DC offset
                 self.Q_(0.1, 'volt/second') * t +  # Linear trend
                 self.Q_(1.0, 'volt') * np.sin(2 * np.pi * 10 * t.magnitude))
        
        # Test constant detrending
        result_const = unit_aware_fft(signal, sampling_rate=fs, detrend='constant')
        dc_component = np.abs(result_const.spectrum[0])
        assert dc_component.magnitude < 1.0  # DC largely removed
        
        # Test linear detrending
        result_linear = unit_aware_fft(signal, sampling_rate=fs, detrend='linear')
        dc_component_linear = np.abs(result_linear.spectrum[0])
        assert dc_component_linear.magnitude < dc_component.magnitude
    
    def test_two_sided_spectrum(self):
        """Test two-sided spectrum output"""
        fs = self.Q_(1000, 'Hz')
        t = np.arange(0, 1, 1/fs.magnitude) * self.ureg.second
        signal = self.Q_(1.0, 'meter') * np.sin(2 * np.pi * 100 * t.magnitude)
        
        result = unit_aware_fft(signal, sampling_rate=fs, return_onesided=False)
        
        # Should have both positive and negative frequencies
        assert len(result.frequencies) == len(t)
        assert np.any(result.frequencies.magnitude < 0)
        
        # Check symmetry for real signal
        positive_idx = np.where(result.frequencies.magnitude > 0)[0]
        negative_idx = np.where(result.frequencies.magnitude < 0)[0]
        
        # Amplitudes should be conjugate symmetric
        for p_idx in positive_idx[:10]:  # Check first 10
            freq = result.frequencies[p_idx].magnitude
            n_idx = np.argmin(np.abs(result.frequencies.magnitude + freq))
            assert np.abs(result.spectrum[p_idx] - np.conj(result.spectrum[n_idx])).magnitude < 1e-10
    
    def test_complex_signal(self):
        """Test FFT of complex signal"""
        fs = self.Q_(1000, 'Hz')
        t = np.arange(0, 1, 1/fs.magnitude) * self.ureg.second
        
        # Complex exponential (single frequency at +100 Hz)
        signal = self.Q_(1.0 + 0j, 'volt') * np.exp(2j * np.pi * 100 * t.magnitude)
        
        result = unit_aware_fft(signal, sampling_rate=fs, return_onesided=False)
        
        # Should have peak only at +100 Hz
        pos_100_idx = np.argmin(np.abs(result.frequencies.magnitude - 100))
        neg_100_idx = np.argmin(np.abs(result.frequencies.magnitude + 100))
        
        assert np.abs(result.spectrum[pos_100_idx]).magnitude > 0.9
        assert np.abs(result.spectrum[neg_100_idx]).magnitude < 0.1
    
    def test_inverse_fft(self):
        """Test inverse FFT reconstruction"""
        # Original signal
        fs = self.Q_(1000, 'Hz')
        t_original = np.arange(0, 1, 1/fs.magnitude) * self.ureg.second
        signal_original = self.Q_(2.0, 'volt') * np.sin(2 * np.pi * 50 * t_original.magnitude)
        
        # Forward FFT
        result = unit_aware_fft(signal_original, sampling_rate=fs)
        
        # Inverse FFT
        t_reconstructed, signal_reconstructed = inverse_unit_aware_fft(result)
        
        # Check reconstruction
        assert np.allclose(signal_original.magnitude, signal_reconstructed.magnitude, rtol=1e-10)
        assert signal_reconstructed.units == signal_original.units
        assert t_reconstructed.units == t_original.units
    
    def test_different_input_formats(self):
        """Test different ways to specify input"""
        signal_values = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        
        # With time array
        t = self.Q_(np.linspace(0, 1, 1000), 'second')
        signal = self.Q_(signal_values, 'volt')
        result1 = unit_aware_fft(signal, time=t)
        
        # With sampling rate
        fs = self.Q_(1000, 'Hz')
        result2 = unit_aware_fft(signal, sampling_rate=fs)
        
        # With dt
        dt = self.Q_(0.001, 'second')
        result3 = unit_aware_fft(signal, dt=dt)
        
        # All should give same result
        assert np.allclose(result1.spectrum.magnitude, result2.spectrum.magnitude)
        assert np.allclose(result1.spectrum.magnitude, result3.spectrum.magnitude)
    
    def test_unit_conversion(self):
        """Test automatic unit handling"""
        # Signal in millivolts, sampling in kHz
        fs = self.Q_(10, 'kHz')
        t = np.arange(0, 0.1, 1/(fs.to('Hz').magnitude)) * self.ureg.second
        signal = self.Q_(500, 'millivolt') * np.sin(2 * np.pi * 1000 * t.magnitude)
        
        result = unit_aware_fft(signal, sampling_rate=fs)
        
        # Should handle unit conversions automatically
        assert result.frequencies.units == self.ureg.Hz
        assert result.spectrum.units == self.ureg.millivolt
        
        # Check frequency is correct despite kHz input
        dominant = result.dominant_frequency()
        assert abs(dominant.to('Hz').magnitude - 1000) < 10
    
    def test_multidimensional_fft(self):
        """Test FFT along specific axis"""
        # 2D signal: multiple channels
        n_channels = 3
        fs = self.Q_(1000, 'Hz')
        t = np.arange(0, 1, 1/fs.magnitude)
        
        # Different frequency for each channel
        freqs = [50, 100, 200]
        signal_values = np.array([np.sin(2 * np.pi * f * t) for f in freqs])
        signal = self.Q_(signal_values, 'volt')
        
        result = unit_aware_fft(signal, sampling_rate=fs, axis=1)
        
        # Check each channel
        for i, expected_freq in enumerate(freqs):
            spectrum_channel = result.spectrum[i]
            peak_idx = np.argmax(np.abs(spectrum_channel[1:len(spectrum_channel)//2])) + 1
            peak_freq = result.frequencies[peak_idx]
            assert abs(peak_freq.magnitude - expected_freq) < 5
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        signal = self.Q_(np.random.randn(100), 'volt')
        
        # No timing information
        with pytest.raises(ValueError):
            unit_aware_fft(signal)
        
        # Invalid window
        with pytest.raises(ValueError):
            unit_aware_fft(signal, dt=0.01, window='invalid_window')
        
        # Can't inverse power spectrum
        result = unit_aware_fft(signal, dt=0.01, scaling='power')
        with pytest.raises(ValueError):
            inverse_unit_aware_fft(result)