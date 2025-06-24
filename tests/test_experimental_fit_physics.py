import pytest
import numpy as np
from phizzy.fitting import experimental_fit_physics


class TestExperimentalFitPhysics:
    
    def test_linear_fit(self):
        """Test fitting linear model"""
        
        # Generate synthetic linear data
        x = np.linspace(0, 10, 50)
        true_slope = 2.5
        true_intercept = 1.0
        y_true = true_slope * x + true_intercept
        y_noise = 0.1 * np.random.randn(50)
        y = y_true + y_noise
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='linear',
            y_errors=np.ones_like(y) * 0.1
        )
        
        assert 'a' in result.parameters  # slope
        assert 'b' in result.parameters  # intercept
        
        # Check that fitted parameters are reasonably close to true values
        assert abs(result.parameters['a'] - true_slope) < 0.5
        assert abs(result.parameters['b'] - true_intercept) < 0.5
        
        # Check that parameter errors exist
        assert 'a' in result.parameter_errors
        assert 'b' in result.parameter_errors
        
        # Check goodness of fit
        assert result.chi_squared > 0
        assert result.p_value >= 0
        assert result.p_value <= 1
        assert 'r_squared' in result.goodness_of_fit
    
    def test_gaussian_fit(self):
        """Test fitting Gaussian model"""
        
        # Generate synthetic Gaussian data
        x = np.linspace(-3, 3, 100)
        true_amplitude = 10.0
        true_mean = 0.5
        true_sigma = 1.2
        
        y_true = true_amplitude * np.exp(-(x - true_mean)**2 / (2 * true_sigma**2))
        y = y_true + 0.1 * np.random.randn(100)
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='gaussian',
            initial_guess={'a': 8.0, 'mu': 0.0, 'sigma': 1.0}
        )
        
        assert 'a' in result.parameters    # amplitude
        assert 'mu' in result.parameters   # mean
        assert 'sigma' in result.parameters  # width
        
        # Check fitted parameters are reasonable
        assert abs(result.parameters['a'] - true_amplitude) < 2.0
        assert abs(result.parameters['mu'] - true_mean) < 0.5
        assert abs(result.parameters['sigma'] - true_sigma) < 0.5
        
        # Check that fitted curve exists
        assert len(result.fitted_curve) == len(x)
    
    def test_exponential_decay(self):
        """Test fitting exponential decay"""
        
        x = np.linspace(0, 5, 60)
        true_amplitude = 5.0
        true_decay = 1.5
        true_offset = 0.2
        
        y_true = true_amplitude * np.exp(-true_decay * x) + true_offset
        y = y_true + 0.05 * np.random.randn(60)
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='exponential',
            parameter_bounds={'b': (0, 10), 'c': (-1, 2)}
        )
        
        assert 'a' in result.parameters  # amplitude
        assert 'b' in result.parameters  # decay constant
        assert 'c' in result.parameters  # offset
        
        # Decay constant should be positive and reasonable
        assert result.parameters['b'] > 0
        assert abs(result.parameters['b'] - true_decay) < 1.0
    
    def test_power_law_fit(self):
        """Test fitting power law"""
        
        x = np.linspace(1, 10, 40)  # Avoid x=0 for power law
        true_prefactor = 2.0
        true_exponent = 1.5
        
        y_true = true_prefactor * x**true_exponent
        y = y_true * (1 + 0.05 * np.random.randn(40))  # Multiplicative noise
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='power_law',
            initial_guess={'a': 1.0, 'b': 1.0}
        )
        
        assert 'a' in result.parameters  # prefactor
        assert 'b' in result.parameters  # exponent
        
        # Check exponent is reasonable
        assert abs(result.parameters['b'] - true_exponent) < 0.5
    
    def test_custom_function(self):
        """Test fitting with custom function"""
        
        def custom_model(x, A, tau, phi):
            """Damped oscillation"""
            return A * np.exp(-x/tau) * np.cos(x + phi)
        
        x = np.linspace(0, 10, 80)
        y_true = custom_model(x, 3.0, 2.0, 0.5)
        y = y_true + 0.1 * np.random.randn(80)
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model=custom_model,
            initial_guess={'A': 2.0, 'tau': 1.5, 'phi': 0.0}
        )
        
        assert 'A' in result.parameters
        assert 'tau' in result.parameters
        assert 'phi' in result.parameters
        
        # Should converge to reasonable values
        assert result.parameters['A'] > 0
        assert result.parameters['tau'] > 0
    
    def test_poisson_distribution(self):
        """Test fitting with Poisson error distribution"""
        
        x = np.linspace(0, 5, 30)
        y_true = 100 * np.exp(-2*x) + 5  # Exponential decay with background
        y = np.random.poisson(y_true)  # Poisson noise
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='exponential',
            distribution='poisson'
        )
        
        assert result.parameters is not None
        assert len(result.fitted_curve) == len(x)
        
        # Poisson fitting should handle count data appropriately
        assert all(result.fitted_curve >= 0)  # Non-negative predictions
    
    def test_robust_fitting(self):
        """Test robust fitting with outliers"""
        
        x = np.linspace(0, 10, 50)
        y_true = 2 * x + 1
        y = y_true + 0.1 * np.random.randn(50)
        
        # Add some outliers
        y[10] += 5
        y[25] -= 4
        y[40] += 3
        
        # Standard fit
        result_standard = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='linear',
            robust_fitting=False
        )
        
        # Robust fit
        result_robust = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='linear',
            robust_fitting=True
        )
        
        # Robust fit should be less affected by outliers
        # (though this is a statistical test and might occasionally fail)
        assert abs(result_robust.parameters['a'] - 2.0) <= abs(result_standard.parameters['a'] - 2.0)
    
    def test_outlier_detection(self):
        """Test automatic outlier detection"""
        
        x = np.linspace(0, 10, 40)
        y_true = x**2
        y = y_true + 0.5 * np.random.randn(40)
        
        # Add clear outliers
        y[15] += 10
        y[30] -= 15
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='power_law',
            outlier_detection=True,
            initial_guess={'a': 1.0, 'b': 2.0}
        )
        
        # Should still fit reasonably well despite outliers
        assert abs(result.parameters['b'] - 2.0) < 0.5
    
    def test_systematic_errors(self):
        """Test handling of systematic uncertainties"""
        
        x = np.linspace(0, 5, 30)
        y = 3 * x + 1 + 0.1 * np.random.randn(30)
        
        systematic_errors = {
            'calibration': 0.02,  # 2% systematic error
            'temperature': 0.01   # 1% systematic error
        }
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='linear',
            systematic_errors=systematic_errors
        )
        
        # Should have systematic uncertainty estimates
        assert len(result.systematic_uncertainties) > 0
        assert any('calibration' in key for key in result.systematic_uncertainties.keys())
    
    def test_bootstrap_errors(self):
        """Test bootstrap error estimation"""
        
        x = np.linspace(0, 4, 25)
        y = 2 * x + 0.1 * np.random.randn(25)
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='linear',
            bootstrap_samples=50  # Reduced for testing speed
        )
        
        # Bootstrap should provide error estimates
        assert 'a' in result.parameter_errors
        assert 'b' in result.parameter_errors
        assert result.parameter_errors['a'] > 0
        assert result.parameter_errors['b'] > 0
    
    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        
        x = np.linspace(0, 3, 20)
        y = 1.5 * x + 0.5 + 0.1 * np.random.randn(20)
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='linear'
        )
        
        # Test confidence interval method
        ci_68 = result.confidence_interval('a', confidence=0.68)
        ci_95 = result.confidence_interval('a', confidence=0.95)
        
        assert len(ci_68) == 2
        assert len(ci_95) == 2
        assert ci_68[0] < ci_68[1]  # Lower < Upper
        assert ci_95[0] < ci_95[1]
        
        # 95% interval should be wider than 68%
        assert (ci_95[1] - ci_95[0]) > (ci_68[1] - ci_68[0])
    
    def test_information_criteria(self):
        """Test AIC and BIC calculation"""
        
        x = np.linspace(0, 5, 40)
        y = 2 * x + 1 + 0.15 * np.random.randn(40)
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='linear'
        )
        
        # Should have information criteria
        assert hasattr(result, 'aic')
        assert hasattr(result, 'bic')
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
    
    def test_physics_models(self):
        """Test some physics-specific models"""
        
        # Test Fermi-Dirac distribution
        x = np.linspace(-2, 2, 60)  # Energy relative to chemical potential
        y_true = 1.0 / (1 + np.exp(x / 0.3))  # kT = 0.3
        y = y_true + 0.02 * np.random.randn(60)
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='fermi_dirac',
            initial_guess={'a': 1.0, 'mu': 0.0, 'kT': 0.5}
        )
        
        assert 'a' in result.parameters
        assert 'mu' in result.parameters
        assert 'kT' in result.parameters
        
        # Temperature should be positive
        assert result.parameters['kT'] > 0
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        
        x = np.linspace(0, 5, 20)
        y = 2 * x + 1
        
        # Test unknown model
        with pytest.raises(ValueError):
            experimental_fit_physics(
                x_data=x,
                y_data=y,
                model='unknown_model'
            )
        
        # Test mismatched array sizes
        with pytest.raises((ValueError, IndexError)):
            experimental_fit_physics(
                x_data=x,
                y_data=y[:-5],  # Shorter array
                model='linear'
            )
    
    def test_failed_fit_handling(self):
        """Test handling of failed fits"""
        
        # Create data that's difficult to fit
        x = np.linspace(0, 1, 10)
        y = np.random.randn(10) * 1000  # Very noisy data
        
        # Try to fit with unrealistic bounds
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='gaussian',
            parameter_bounds={'sigma': (1e-10, 1e-10)}  # Impossible bounds
        )
        
        # Should return a result even if fitting fails
        assert hasattr(result, 'parameters')
        assert hasattr(result, 'goodness_of_fit')
        
        # Failed fits should have some indication
        if 'error' in result.goodness_of_fit:
            assert isinstance(result.goodness_of_fit['error'], str)
    
    def test_background_subtraction(self):
        """Test background subtraction functionality"""
        
        x = np.linspace(0, 10, 50)
        signal = 5 * np.exp(-(x-5)**2 / 2)  # Gaussian signal
        background = 0.1 * x + 0.5  # Linear background
        y = signal + background + 0.1 * np.random.randn(50)
        
        result = experimental_fit_physics(
            x_data=x,
            y_data=y,
            model='gaussian',
            background_model='linear'
        )
        
        # Should fit the signal after background subtraction
        assert 'mu' in result.parameters
        # Peak should be near x=5
        assert abs(result.parameters['mu'] - 5.0) < 1.0