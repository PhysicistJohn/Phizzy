"""
Tests for dimensional analysis and units system.
"""

import pytest
import torch
import numpy as np
from phizzyML.core.units import Unit, Quantity, DimensionalError, Dimensions


class TestDimensions:
    def test_dimensions_equality(self):
        dim1 = Dimensions(length=1, mass=1, time=-2)
        dim2 = Dimensions(length=1, mass=1, time=-2)
        dim3 = Dimensions(length=2, mass=1, time=-2)
        
        assert dim1 == dim2
        assert dim1 != dim3
    
    def test_dimensions_multiplication(self):
        dim1 = Dimensions(length=1, time=-1)
        dim2 = Dimensions(length=1, time=-1)
        result = dim1 * dim2
        
        assert result == Dimensions(length=2, time=-2)
    
    def test_dimensions_division(self):
        dim1 = Dimensions(length=2, time=-2)
        dim2 = Dimensions(length=1, time=-1)
        result = dim1 / dim2
        
        assert result == Dimensions(length=1, time=-1)
    
    def test_dimensions_power(self):
        dim = Dimensions(length=1, time=-1)
        result = dim ** 2
        
        assert result == Dimensions(length=2, time=-2)
    
    def test_dimensionless(self):
        dim1 = Dimensions()
        dim2 = Dimensions(length=1, time=-1)
        dim3 = dim2 / dim2
        
        assert dim1.is_dimensionless()
        assert not dim2.is_dimensionless()
        assert dim3.is_dimensionless()


class TestUnit:
    def test_base_units(self):
        assert Unit.meter.dimensions == Dimensions(length=1)
        assert Unit.kilogram.dimensions == Dimensions(mass=1)
        assert Unit.second.dimensions == Dimensions(time=1)
    
    def test_derived_units(self):
        # Force = mass * acceleration
        force = Unit.kilogram * Unit.meter / Unit.second**2
        assert force.dimensions == Unit.newton.dimensions
        
        # Energy = force * distance
        energy = Unit.newton * Unit.meter
        assert energy.dimensions == Unit.joule.dimensions
    
    def test_unit_operations(self):
        velocity = Unit.meter / Unit.second
        assert velocity.dimensions == Dimensions(length=1, time=-1)
        
        acceleration = velocity / Unit.second
        assert acceleration.dimensions == Dimensions(length=1, time=-2)


class TestQuantity:
    def test_quantity_creation(self):
        mass = Quantity(10.0, Unit.kilogram)
        assert mass.value == 10.0
        assert mass.unit == Unit.kilogram
        
        # Array quantity
        positions = Quantity(np.array([1.0, 2.0, 3.0]), Unit.meter)
        assert positions.value.shape == (3,)
    
    def test_quantity_addition(self):
        q1 = Quantity(5.0, Unit.meter)
        q2 = Quantity(3.0, Unit.meter)
        result = q1 + q2
        
        assert result.value == 8.0
        assert result.unit.dimensions == Unit.meter.dimensions
    
    def test_quantity_addition_error(self):
        q1 = Quantity(5.0, Unit.meter)
        q2 = Quantity(3.0, Unit.second)
        
        with pytest.raises(DimensionalError):
            _ = q1 + q2
    
    def test_quantity_multiplication(self):
        mass = Quantity(2.0, Unit.kilogram)
        acceleration = Quantity(9.8, Unit.meter / Unit.second**2)
        force = mass * acceleration
        
        assert force.value == pytest.approx(19.6)
        assert force.unit.dimensions == Unit.newton.dimensions
    
    def test_quantity_division(self):
        distance = Quantity(100.0, Unit.meter)
        time = Quantity(10.0, Unit.second)
        velocity = distance / time
        
        assert velocity.value == 10.0
        assert velocity.unit.dimensions == Dimensions(length=1, time=-1)
    
    def test_quantity_power(self):
        length = Quantity(3.0, Unit.meter)
        area = length ** 2
        
        assert area.value == 9.0
        assert area.unit.dimensions == Dimensions(length=2)
    
    def test_unit_conversion(self):
        # Create a kilometer unit
        kilometer = Unit("kilometer", "km", Dimensions(length=1), scale=1000.0)
        
        distance_m = Quantity(5000.0, Unit.meter)
        distance_km = distance_m.to(kilometer)
        
        assert distance_km.value == pytest.approx(5.0)
        assert distance_km.unit.dimensions == Unit.meter.dimensions
    
    def test_physics_calculations(self):
        # Kinetic energy calculation: E = 1/2 * m * v^2
        mass = Quantity(2.0, Unit.kilogram)
        velocity = Quantity(10.0, Unit.meter / Unit.second)
        
        kinetic_energy = 0.5 * mass * velocity**2
        
        assert kinetic_energy.value == 100.0
        assert kinetic_energy.unit.dimensions == Unit.joule.dimensions
    
    def test_dimensional_checking(self):
        force = Quantity(10.0, Unit.newton)
        
        # Check if it has force dimensions
        force_dims = Dimensions(length=1, mass=1, time=-2)
        assert force.check_dimensions(force_dims)
        
        # Check against wrong dimensions
        energy_dims = Dimensions(length=2, mass=1, time=-2)
        assert not force.check_dimensions(energy_dims)