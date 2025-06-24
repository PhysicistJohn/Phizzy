"""
PhizzyML: A Python library for physics-informed machine learning
================================================================

A comprehensive library addressing key gaps in physics ML:
- Dimensional analysis and automatic unit checking
- Symplectic integrators for Hamiltonian systems
- Hard physical constraint enforcement
- Uncertainty quantification for experimental data
- Physics-aware optimizers
- Causality-respecting layers
- Geometry and mesh support for complex domains
- Experimental data interfaces with metadata support
"""

__version__ = "0.1.0"

# Core functionality
from .core.units import Unit, Quantity, DimensionalError

# Integrators
from .integrators.symplectic import SymplecticEuler, LeapfrogIntegrator, VerletIntegrator

# Constraints
from .constraints.conservation import ConservationLayer, EnergyConservingLayer, MomentumConservingLayer

# Uncertainty quantification
from .uncertainty.propagation import UncertainTensor, error_propagation, monte_carlo_propagation

# Physics-aware optimizers
from .optimizers.physics_aware import EnergyMinimizer, VariationalOptimizer, LangevinOptimizer

# Causality-aware layers
from .layers.causal import CausalConv1d, CausalAttention, CausalLSTM

# Geometry and mesh support
from .geometry.mesh import Mesh, MeshGenerator, AdaptiveMesh
from .geometry.operators import LaplacianOperator, GradientOperator, DivergenceOperator

# Experimental data interfaces
from .data.experimental import HDF5DataInterface, ROOTDataInterface, CSVDataInterface, ExperimentalDataset
from .data.preprocessing import DataNormalizer, OutlierDetector, PhysicsFeatureExtractor

__all__ = [
    # Core
    'Unit', 'Quantity', 'DimensionalError',
    
    # Integrators
    'SymplecticEuler', 'LeapfrogIntegrator', 'VerletIntegrator',
    
    # Constraints
    'ConservationLayer', 'EnergyConservingLayer', 'MomentumConservingLayer',
    
    # Uncertainty
    'UncertainTensor', 'error_propagation', 'monte_carlo_propagation',
    
    # Optimizers
    'EnergyMinimizer', 'VariationalOptimizer', 'LangevinOptimizer',
    
    # Layers
    'CausalConv1d', 'CausalAttention', 'CausalLSTM',
    
    # Geometry
    'Mesh', 'MeshGenerator', 'AdaptiveMesh',
    'LaplacianOperator', 'GradientOperator', 'DivergenceOperator',
    
    # Data
    'HDF5DataInterface', 'ROOTDataInterface', 'CSVDataInterface', 'ExperimentalDataset',
    'DataNormalizer', 'OutlierDetector', 'PhysicsFeatureExtractor'
]