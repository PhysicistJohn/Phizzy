import pytest
import numpy as np
import sympy as sp
from phizzy.uncertainties import propagate_uncertainties_symbolic
from phizzy.uncertainties.symbolic import UncertainValue


class TestPropagateUncertaintiesSymbolic:
    
    def test_simple_addition(self):
        """Test uncertainty propagation for addition"""
        variables = {
            'x': (10.0, 0.1),
            'y': (20.0, 0.2)
        }
        
        result = propagate_uncertainties_symbolic('x + y', variables)
        
        # For uncorrelated addition: σ = sqrt(σ_x² + σ_y²)
        expected_uncertainty = np.sqrt(0.1**2 + 0.2**2)
        
        assert abs(result.numeric_result.value - 30.0) < 1e-10
        assert abs(result.numeric_result.uncertainty - expected_uncertainty) < 1e-10
        
    def test_multiplication(self):
        """Test uncertainty propagation for multiplication"""
        variables = {
            'x': UncertainValue(2.0, 0.1),
            'y': UncertainValue(3.0, 0.15)
        }
        
        result = propagate_uncertainties_symbolic('x * y', variables)
        
        # For multiplication: σ/f = sqrt((σ_x/x)² + (σ_y/y)²)
        relative_uncertainty = np.sqrt((0.1/2.0)**2 + (0.15/3.0)**2)
        expected_uncertainty = 6.0 * relative_uncertainty
        
        assert abs(result.numeric_result.value - 6.0) < 1e-10
        assert abs(result.numeric_result.uncertainty - expected_uncertainty) < 1e-10
        
    def test_correlated_variables(self):
        """Test with correlated variables"""
        variables = {
            'x': (1.0, 0.1),
            'y': (2.0, 0.2)
        }
        correlations = {('x', 'y'): 0.5}
        
        result = propagate_uncertainties_symbolic(
            'x + y', variables, correlations=correlations
        )
        
        # With correlation: σ² = σ_x² + σ_y² + 2ρσ_xσ_y
        expected_var = 0.1**2 + 0.2**2 + 2 * 0.5 * 0.1 * 0.2
        expected_uncertainty = np.sqrt(expected_var)
        
        assert abs(result.numeric_result.uncertainty - expected_uncertainty) < 1e-10
        
    def test_complex_expression(self):
        """Test complex mathematical expression"""
        variables = {
            'x': (1.0, 0.05),
            'y': (2.0, 0.1),
            'z': (3.0, 0.15)
        }
        
        expression = 'sin(x) * exp(y) / sqrt(z)'
        result = propagate_uncertainties_symbolic(expression, variables)
        
        # Check central value
        import math
        expected_value = math.sin(1.0) * math.exp(2.0) / math.sqrt(3.0)
        assert abs(result.numeric_result.value - expected_value) < 1e-10
        
        # Check that uncertainty is reasonable
        assert 0 < result.numeric_result.uncertainty < 1.0
        
    def test_sensitivity_coefficients(self):
        """Test sensitivity coefficient calculation"""
        variables = {
            'x': (10.0, 1.0),
            'y': (5.0, 0.5)
        }
        
        result = propagate_uncertainties_symbolic('x / y', variables)
        
        # Sensitivity coefficients should be partial derivatives
        # ∂(x/y)/∂x = 1/y = 1/5 = 0.2
        # ∂(x/y)/∂y = -x/y² = -10/25 = -0.4
        
        assert abs(result.sensitivity_coefficients['x'] - 0.2) < 1e-10
        assert abs(result.sensitivity_coefficients['y'] - (-0.4)) < 1e-10
        
    def test_relative_contributions(self):
        """Test relative contribution calculation"""
        variables = {
            'x': (1.0, 0.1),  # 10% relative uncertainty
            'y': (10.0, 0.1)  # 1% relative uncertainty
        }
        
        result = propagate_uncertainties_symbolic('x * y', variables)
        
        # For multiplication, relative uncertainties add in quadrature
        # x contributes more to the total uncertainty
        assert result.relative_contributions['x'] > result.relative_contributions['y']
        assert abs(sum(result.relative_contributions.values()) - 1.0) < 1e-10
        
    def test_monte_carlo_method(self):
        """Test Monte Carlo propagation method"""
        variables = {
            'x': (1.0, 0.1),
            'y': (2.0, 0.2)
        }
        
        result = propagate_uncertainties_symbolic(
            'x**2 + y', variables, method='monte_carlo', monte_carlo_samples=50000
        )
        
        # Compare with Taylor propagation
        taylor_result = propagate_uncertainties_symbolic('x**2 + y', variables)
        
        # Results should be close
        assert abs(result.numeric_result.value - taylor_result.numeric_result.value) < 0.01
        assert abs(result.numeric_result.uncertainty - taylor_result.numeric_result.uncertainty) < 0.01
        
        # Check that Monte Carlo provides additional statistics
        assert 'skewness' in result.info
        assert 'kurtosis' in result.info
        
    def test_bootstrap_method(self):
        """Test bootstrap propagation method"""
        variables = {
            'x': (5.0, 0.5),
            'y': (3.0, 0.3)
        }
        
        result = propagate_uncertainties_symbolic(
            'sqrt(x**2 + y**2)', variables, method='bootstrap', monte_carlo_samples=10000
        )
        
        # Should have bootstrap-specific info
        assert 'bootstrap_ci' in result.info
        assert 'bootstrap_std' in result.info
        
        # Result should be reasonable
        expected_value = np.sqrt(5**2 + 3**2)
        assert abs(result.numeric_result.value - expected_value) < 0.1
        
    def test_symbolic_output(self):
        """Test symbolic uncertainty expression"""
        x, y = sp.symbols('x y')
        expression = x**2 + sp.sin(y)
        
        variables = {
            'x': (1.0, 0.1),
            'y': (np.pi/2, 0.05)
        }
        
        result = propagate_uncertainties_symbolic(expression, variables)
        
        # Should have symbolic uncertainty expression
        assert isinstance(result.symbolic_uncertainty, sp.Expr)
        assert result.expression == expression
        
    def test_simplification(self):
        """Test expression simplification"""
        variables = {
            'x': (1.0, 0.1),
            'y': (1.0, 0.1)
        }
        
        # Expression that can be simplified
        result = propagate_uncertainties_symbolic(
            'x*y/x + x - x', variables, simplify=True
        )
        
        # Should simplify to just y
        assert result.numeric_result.value == 1.0
        
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Unknown variable
        with pytest.raises(Exception):
            propagate_uncertainties_symbolic('x + z', {'x': (1, 0.1)})
        
        # Invalid correlation
        with pytest.raises(ValueError):
            propagate_uncertainties_symbolic(
                'x + y', 
                {'x': (1, 0.1), 'y': (2, 0.2)},
                correlations={('x', 'y'): 1.5}  # > 1
            )
        
        # Unknown method
        with pytest.raises(ValueError):
            propagate_uncertainties_symbolic(
                'x', {'x': (1, 0.1)}, method='invalid'
            )
            
    def test_physics_example(self):
        """Test with a real physics example: pendulum period"""
        # T = 2π√(L/g)
        variables = {
            'L': (1.0, 0.01),  # Length: 1.0 ± 0.01 m
            'g': (9.81, 0.01)  # Gravity: 9.81 ± 0.01 m/s²
        }
        
        expression = '2 * pi * sqrt(L / g)'
        result = propagate_uncertainties_symbolic(expression, variables)
        
        # Check period value
        expected_period = 2 * np.pi * np.sqrt(1.0 / 9.81)
        assert abs(result.numeric_result.value - expected_period) < 1e-6
        
        # Check relative contributions (length uncertainty dominates for small uncertainties)
        # Since both have ~1% relative uncertainty, contributions should be similar
        assert 0.4 < result.relative_contributions['L'] < 0.6
        assert 0.4 < result.relative_contributions['g'] < 0.6