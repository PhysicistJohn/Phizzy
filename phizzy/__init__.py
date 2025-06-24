"""
Phizzy - Advanced Physics Library for Python
===========================================

A comprehensive physics library providing essential functions missing from 
existing Python scientific computing libraries.

Main Features:
- Symplectic integrators for Hamiltonian systems
- Symbolic computation with uncertainty propagation
- Unit-aware Fourier transforms
- 3D field visualization
- Advanced Monte Carlo physics simulations
- Optimized tensor contractions
- Phase transition detection
- Coupled PDE solvers
- Lagrangian/Hamiltonian mechanics
- Physics-aware experimental fitting
"""

__version__ = "0.1.0"
__author__ = "Phizzy Development Team"

from .integrators import symplectic_integrate
from .uncertainties import propagate_uncertainties_symbolic
from .transforms import unit_aware_fft
from .visualization import field_visualization_3d
from .monte_carlo import monte_carlo_physics
from .tensors import tensor_contract_optimize
from .transitions import phase_transition_detect
from .pde import coupled_pde_solve
from .mechanics import lagrangian_to_equations
from .fitting import experimental_fit_physics

__all__ = [
    'symplectic_integrate',
    'propagate_uncertainties_symbolic',
    'unit_aware_fft',
    'field_visualization_3d',
    'monte_carlo_physics',
    'tensor_contract_optimize',
    'phase_transition_detect',
    'coupled_pde_solve',
    'lagrangian_to_equations',
    'experimental_fit_physics'
]