"""
Dimensional analysis and unit system for physics-aware computations.
"""

from typing import Dict, Union, Optional, Any
import numpy as np
from dataclasses import dataclass
import operator


class DimensionalError(Exception):
    """Raised when dimensional analysis fails."""
    pass


@dataclass(frozen=True)
class Dimensions:
    """Represents physical dimensions using the SI base units."""
    length: float = 0
    mass: float = 0
    time: float = 0
    current: float = 0
    temperature: float = 0
    amount: float = 0
    luminous_intensity: float = 0
    
    def __mul__(self, other: 'Dimensions') -> 'Dimensions':
        return Dimensions(
            length=self.length + other.length,
            mass=self.mass + other.mass,
            time=self.time + other.time,
            current=self.current + other.current,
            temperature=self.temperature + other.temperature,
            amount=self.amount + other.amount,
            luminous_intensity=self.luminous_intensity + other.luminous_intensity
        )
    
    def __truediv__(self, other: 'Dimensions') -> 'Dimensions':
        return Dimensions(
            length=self.length - other.length,
            mass=self.mass - other.mass,
            time=self.time - other.time,
            current=self.current - other.current,
            temperature=self.temperature - other.temperature,
            amount=self.amount - other.amount,
            luminous_intensity=self.luminous_intensity - other.luminous_intensity
        )
    
    def __pow__(self, power: float) -> 'Dimensions':
        return Dimensions(
            length=self.length * power,
            mass=self.mass * power,
            time=self.time * power,
            current=self.current * power,
            temperature=self.temperature * power,
            amount=self.amount * power,
            luminous_intensity=self.luminous_intensity * power
        )
    
    def __eq__(self, other: 'Dimensions') -> bool:
        return all([
            abs(self.length - other.length) < 1e-10,
            abs(self.mass - other.mass) < 1e-10,
            abs(self.time - other.time) < 1e-10,
            abs(self.current - other.current) < 1e-10,
            abs(self.temperature - other.temperature) < 1e-10,
            abs(self.amount - other.amount) < 1e-10,
            abs(self.luminous_intensity - other.luminous_intensity) < 1e-10
        ])
    
    def is_dimensionless(self) -> bool:
        return self == Dimensions()


class Unit:
    """Represents a physical unit with dimensional analysis."""
    
    def __init__(self, name: str, symbol: str, dimensions: Dimensions, scale: float = 1.0):
        self.name = name
        self.symbol = symbol
        self.dimensions = dimensions
        self.scale = scale
    
    def __mul__(self, other: Union['Unit', float]) -> 'Unit':
        if isinstance(other, (int, float)):
            return Unit(f"{self.name}", self.symbol, self.dimensions, self.scale * other)
        return Unit(
            f"{self.name}*{other.name}",
            f"{self.symbol}*{other.symbol}",
            self.dimensions * other.dimensions,
            self.scale * other.scale
        )
    
    def __truediv__(self, other: Union['Unit', float]) -> 'Unit':
        if isinstance(other, (int, float)):
            return Unit(f"{self.name}", self.symbol, self.dimensions, self.scale / other)
        return Unit(
            f"{self.name}/{other.name}",
            f"{self.symbol}/{other.symbol}",
            self.dimensions / other.dimensions,
            self.scale / other.scale
        )
    
    def __pow__(self, power: float) -> 'Unit':
        return Unit(
            f"{self.name}^{power}",
            f"{self.symbol}^{power}",
            self.dimensions ** power,
            self.scale ** power
        )
    
    def __repr__(self) -> str:
        return f"Unit({self.symbol})"


# Define base SI units
Unit.meter = Unit("meter", "m", Dimensions(length=1))
Unit.kilogram = Unit("kilogram", "kg", Dimensions(mass=1))
Unit.second = Unit("second", "s", Dimensions(time=1))
Unit.ampere = Unit("ampere", "A", Dimensions(current=1))
Unit.kelvin = Unit("kelvin", "K", Dimensions(temperature=1))
Unit.mole = Unit("mole", "mol", Dimensions(amount=1))
Unit.candela = Unit("candela", "cd", Dimensions(luminous_intensity=1))

# Define common derived units
Unit.newton = Unit("newton", "N", Dimensions(length=1, mass=1, time=-2))
Unit.joule = Unit("joule", "J", Dimensions(length=2, mass=1, time=-2))
Unit.watt = Unit("watt", "W", Dimensions(length=2, mass=1, time=-3))
Unit.pascal = Unit("pascal", "Pa", Dimensions(length=-1, mass=1, time=-2))
Unit.hertz = Unit("hertz", "Hz", Dimensions(time=-1))
Unit.coulomb = Unit("coulomb", "C", Dimensions(time=1, current=1))
Unit.volt = Unit("volt", "V", Dimensions(length=2, mass=1, time=-3, current=-1))

# Define dimensionless unit
Unit.dimensionless = Unit("dimensionless", "", Dimensions())


class Quantity:
    """A physical quantity with value and units."""
    
    def __init__(self, value: Union[float, np.ndarray], unit: Unit):
        self.value = np.asarray(value)
        self.unit = unit
    
    def to(self, new_unit: Unit) -> 'Quantity':
        """Convert quantity to a different unit."""
        if self.unit.dimensions != new_unit.dimensions:
            raise DimensionalError(
                f"Cannot convert from {self.unit.symbol} to {new_unit.symbol}: "
                f"incompatible dimensions"
            )
        new_value = self.value * (self.unit.scale / new_unit.scale)
        return Quantity(new_value, new_unit)
    
    def __add__(self, other: 'Quantity') -> 'Quantity':
        if not isinstance(other, Quantity):
            raise TypeError("Can only add Quantity to Quantity")
        if self.unit.dimensions != other.unit.dimensions:
            raise DimensionalError(
                f"Cannot add quantities with different dimensions: "
                f"{self.unit.symbol} and {other.unit.symbol}"
            )
        # Convert other to self's units
        other_converted = other.to(self.unit)
        return Quantity(self.value + other_converted.value, self.unit)
    
    def __sub__(self, other: 'Quantity') -> 'Quantity':
        if not isinstance(other, Quantity):
            raise TypeError("Can only subtract Quantity from Quantity")
        if self.unit.dimensions != other.unit.dimensions:
            raise DimensionalError(
                f"Cannot subtract quantities with different dimensions: "
                f"{self.unit.symbol} and {other.unit.symbol}"
            )
        other_converted = other.to(self.unit)
        return Quantity(self.value - other_converted.value, self.unit)
    
    def __mul__(self, other: Union['Quantity', float]) -> 'Quantity':
        if isinstance(other, (int, float, np.ndarray)):
            return Quantity(self.value * other, self.unit)
        if isinstance(other, Quantity):
            return Quantity(
                self.value * other.value,
                self.unit * other.unit
            )
        raise TypeError(f"Cannot multiply Quantity by {type(other)}")
    
    def __rmul__(self, other: Union[float, int]) -> 'Quantity':
        """Right multiplication for scalar * Quantity."""
        if isinstance(other, (int, float, np.ndarray)):
            return Quantity(other * self.value, self.unit)
        raise TypeError(f"Cannot multiply {type(other)} by Quantity")
    
    def __truediv__(self, other: Union['Quantity', float]) -> 'Quantity':
        if isinstance(other, (int, float, np.ndarray)):
            return Quantity(self.value / other, self.unit)
        if isinstance(other, Quantity):
            return Quantity(
                self.value / other.value,
                self.unit / other.unit
            )
        raise TypeError(f"Cannot divide Quantity by {type(other)}")
    
    def __pow__(self, power: float) -> 'Quantity':
        return Quantity(self.value ** power, self.unit ** power)
    
    def __repr__(self) -> str:
        return f"Quantity({self.value}, {self.unit.symbol})"
    
    @property
    def magnitude(self) -> np.ndarray:
        """Return the numerical value without units."""
        return self.value
    
    def check_dimensions(self, expected_dims: Dimensions) -> bool:
        """Check if quantity has expected dimensions."""
        return self.unit.dimensions == expected_dims