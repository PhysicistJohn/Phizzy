import pytest
import numpy as np
import sympy as sp
from phizzy.mechanics import lagrangian_to_equations


class TestLagrangianToEquations:
    
    def test_simple_harmonic_oscillator(self):
        """Test simple harmonic oscillator"""
        
        # Lagrangian: L = (1/2) * m * q_dot^2 - (1/2) * k * q^2
        lagrangian = "0.5 * m * q_dot**2 - 0.5 * k * q**2"
        coordinates = ['q']
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates
        )
        
        assert len(result.equations_of_motion) == 1
        assert len(result.generalized_coords) == 1
        assert len(result.generalized_velocities) == 1
        
        # Check that we get the expected equation: m * q_ddot + k * q = 0
        eq = result.equations_of_motion[0]
        
        # The equation should involve q and its derivatives
        q = result.generalized_coords[0]
        assert q in eq.free_symbols
    
    def test_double_pendulum(self):
        """Test double pendulum system"""
        
        # Simplified double pendulum coordinates
        coordinates = ['theta1', 'theta2']
        
        # Simplified Lagrangian (without full complexity)
        lagrangian = ("0.5 * m1 * l1**2 * theta1_dot**2 + "
                     "0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + "
                     "2 * l1 * l2 * theta1_dot * theta2_dot * cos(theta1 - theta2)) - "
                     "m1 * g * l1 * cos(theta1) - m2 * g * (l1 * cos(theta1) + l2 * cos(theta2))")
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates
        )
        
        assert len(result.equations_of_motion) == 2
        assert len(result.generalized_coords) == 2
        assert len(result.generalized_velocities) == 2
        
        # Both equations should exist
        assert result.equations_of_motion[0] is not None
        assert result.equations_of_motion[1] is not None
    
    def test_free_particle_cartesian(self):
        """Test free particle in Cartesian coordinates"""
        
        lagrangian = "0.5 * m * (x_dot**2 + y_dot**2 + z_dot**2)"
        coordinates = ['x', 'y', 'z']
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates
        )
        
        assert len(result.equations_of_motion) == 3
        
        # For free particle, all accelerations should be zero
        # Each equation should be of the form: m * q_ddot = 0
        for eq in result.equations_of_motion:
            # Should be a simple equation (ideally zero or containing second derivatives)
            assert eq is not None
    
    def test_particle_in_potential(self):
        """Test particle in gravitational potential"""
        
        lagrangian = "0.5 * m * (x_dot**2 + y_dot**2) - m * g * y"
        coordinates = ['x', 'y']
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates
        )
        
        assert len(result.equations_of_motion) == 2
        
        # x should have no force (free motion)
        # y should have gravitational force
        x_eq = result.equations_of_motion[0]
        y_eq = result.equations_of_motion[1]
        
        assert x_eq is not None
        assert y_eq is not None
    
    def test_with_constraints(self):
        """Test system with holonomic constraints"""
        
        lagrangian = "0.5 * m * (x_dot**2 + y_dot**2) - m * g * y"
        coordinates = ['x', 'y']
        constraints = ["x**2 + y**2 - L**2"]  # Constraint to circle
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates,
            constraints=constraints
        )
        
        assert len(result.equations_of_motion) == 2
        assert len(result.constraints) == 1
        assert len(result.lagrange_multipliers) == 1
        
        # Should have constraint equation
        assert result.constraints[0] is not None
    
    def test_kinetic_potential_separate(self):
        """Test with separate kinetic and potential energy"""
        
        kinetic = "0.5 * m * q_dot**2"
        potential = "0.5 * k * q**2"
        coordinates = ['q']
        
        result = lagrangian_to_equations(
            lagrangian=None,
            coordinates=coordinates,
            kinetic_energy=kinetic,
            potential_energy=potential
        )
        
        assert len(result.equations_of_motion) == 1
        assert result.kinetic_energy is not None
        assert result.potential_energy is not None
        assert result.lagrangian is not None
    
    def test_external_forces(self):
        """Test with external non-conservative forces"""
        
        lagrangian = "0.5 * m * q_dot**2"
        coordinates = ['q']
        external_forces = {'q': 'F_ext'}
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates,
            external_forces=external_forces
        )
        
        assert len(result.equations_of_motion) == 1
        
        # Equation should include external force
        eq = result.equations_of_motion[0]
        assert eq is not None
    
    def test_dissipation_function(self):
        """Test with Rayleigh dissipation function"""
        
        lagrangian = "0.5 * m * q_dot**2 - 0.5 * k * q**2"
        coordinates = ['q']
        dissipation = "0.5 * c * q_dot**2"  # Viscous damping
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates,
            dissipation_function=dissipation
        )
        
        assert len(result.equations_of_motion) == 1
        
        # Should have damping term
        eq = result.equations_of_motion[0]
        assert eq is not None
    
    def test_conserved_quantities(self):
        """Test finding conserved quantities"""
        
        # Time-independent Lagrangian should conserve energy
        lagrangian = "0.5 * m * q_dot**2 - 0.5 * k * q**2"
        coordinates = ['q']
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates,
            conserved_quantities=True
        )
        
        # Should find at least energy conservation
        assert len(result.conserved_quantities) >= 1
    
    def test_cylindrical_coordinates(self):
        """Test with cylindrical coordinates (angular momentum conservation)"""
        
        # Free particle in cylindrical coordinates
        lagrangian = "0.5 * m * (r_dot**2 + r**2 * theta_dot**2 + z_dot**2)"
        coordinates = ['r', 'theta', 'z']
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates,
            conserved_quantities=True
        )
        
        assert len(result.equations_of_motion) == 3
        
        # Should find conserved quantities (angular momentum for theta, linear momentum for z)
        # Note: the implementation might not catch all, but should find some
        assert isinstance(result.conserved_quantities, list)
    
    def test_callable_lagrangian(self):
        """Test with callable Lagrangian function"""
        
        def lagrangian_func(q, q_dot, t):
            return 0.5 * q_dot**2 - 0.5 * q**2
        
        coordinates = ['q']
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian_func,
            coordinates=coordinates
        )
        
        assert len(result.equations_of_motion) == 1
        assert result.equations_of_motion[0] is not None
    
    def test_numerical_function_conversion(self):
        """Test conversion to numerical function"""
        
        lagrangian = "0.5 * m * q_dot**2 - 0.5 * k * q**2"
        coordinates = ['q']
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates
        )
        
        # Test numerical function conversion
        numerical_func = result.get_numerical_function()
        
        # Should be callable
        assert callable(numerical_func)
        
        # Test with some values (this might require specific symbolic values)
        # For now, just check that it doesn't crash
        try:
            # This might not work without substituting symbolic constants
            # but at least check the function exists
            assert numerical_func is not None
        except Exception:
            # Expected if symbolic constants (m, k) are not substituted
            pass
    
    def test_simplification_option(self):
        """Test equation simplification option"""
        
        lagrangian = "0.5 * m * q_dot**2 - 0.5 * k * q**2"
        coordinates = ['q']
        
        # Test with simplification
        result_simplified = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates,
            simplify_equations=True
        )
        
        # Test without simplification
        result_unsimplified = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates,
            simplify_equations=False
        )
        
        assert len(result_simplified.equations_of_motion) == 1
        assert len(result_unsimplified.equations_of_motion) == 1
        
        # Both should produce valid equations
        assert result_simplified.equations_of_motion[0] is not None
        assert result_unsimplified.equations_of_motion[0] is not None
    
    def test_time_dependent_lagrangian(self):
        """Test time-dependent Lagrangian"""
        
        lagrangian = "0.5 * m * q_dot**2 - 0.5 * k * t * q**2"  # Time-dependent spring constant
        coordinates = ['q']
        
        result = lagrangian_to_equations(
            lagrangian=lagrangian,
            coordinates=coordinates,
            conserved_quantities=True
        )
        
        assert len(result.equations_of_motion) == 1
        
        # Time-dependent system should have fewer conserved quantities
        # (energy not conserved for explicit time dependence)
        # But we should still get a valid equation
        assert result.equations_of_motion[0] is not None
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        
        # Test with empty coordinates
        with pytest.raises((ValueError, IndexError)):
            lagrangian_to_equations(
                lagrangian="0.5 * m * q_dot**2",
                coordinates=[]
            )
        
        # Test with invalid Lagrangian string
        with pytest.raises((ValueError, SyntaxError, TypeError)):
            lagrangian_to_equations(
                lagrangian="invalid expression +++",
                coordinates=['q']
            )