# Phizzy - Advanced Physics Library for Python

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive physics library providing 10 essential functions missing from the Python scientific computing ecosystem. Phizzy fills critical gaps in physics simulation, analysis, and visualization.

## 🚀 Features

### Core Functions

1. **Symplectic Integration** (`symplectic_integrate`)
   - Energy-preserving numerical integration for Hamiltonian systems
   - Multiple algorithms: Leapfrog, Yoshida4, Forest-Ruth, PEFRL
   - Automatic energy conservation monitoring

2. **Symbolic Uncertainty Propagation** (`propagate_uncertainties_symbolic`)
   - Combines symbolic mathematics with automatic error propagation
   - Handles correlated variables and complex expressions
   - Multiple methods: Taylor expansion, Monte Carlo

3. **Unit-Aware FFT** (`unit_aware_fft`)
   - Fast Fourier Transform with automatic unit tracking using Pint
   - Proper frequency unit conversion
   - Built-in windowing functions

4. **3D Field Visualization** (`field_visualization_3d`)
   - Intelligent visualization of vector and tensor fields
   - Supports electromagnetic fields, fluid dynamics
   - Multiple backends: matplotlib, plotly

5. **Monte Carlo Physics** (`monte_carlo_physics`)
   - Unified framework for statistical physics simulations
   - Built-in models: Ising, Potts, XY, Heisenberg
   - Advanced algorithms: Metropolis, Wolff, parallel tempering

6. **Optimized Tensor Contractions** (`tensor_contract_optimize`)
   - Automatic optimization of complex tensor operations
   - Multiple strategies: greedy, optimal, dynamic programming
   - Memory-efficient computation

7. **Phase Transition Detection** (`phase_transition_detect`)
   - ML-based automatic detection from simulation/experimental data
   - Identifies continuous and discontinuous transitions
   - Extracts critical exponents and order parameters

8. **Coupled PDE Solver** (`coupled_pde_solve`)
   - Solves systems of partial differential equations
   - Automatic mesh refinement
   - Multiple boundary condition types

9. **Lagrangian Mechanics** (`lagrangian_to_equations`)
   - Converts Lagrangian to equations of motion
   - Handles constraints with Lagrange multipliers
   - Identifies conserved quantities via Noether's theorem

10. **Physics-Aware Fitting** (`experimental_fit_physics`)
    - Experimental data fitting with proper error handling
    - Supports multiple error distributions: Gaussian, Poisson, custom
    - Systematic uncertainty propagation

## 📦 Installation

### From PyPI (when published)
```bash
pip install phizzy
```

### From Source
```bash
git clone https://github.com/PhysicistJohn/Phizzy.git
cd Phizzy
pip install -e .
```

### Dependencies
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- SymPy >= 1.9
- Matplotlib >= 3.4.0
- Pint (for unit handling)
- scikit-learn (for ML features)
- numba (optional, for performance)

## 🎯 Quick Start

### Example 1: Symplectic Integration
```python
import numpy as np
from phizzy.integrators import symplectic_integrate

# Define a pendulum Hamiltonian
def pendulum_hamiltonian(q, p, g=9.81, L=1.0):
    return p**2 / (2 * L**2) + g * L * (1 - np.cos(q))

# Integrate with energy conservation
result = symplectic_integrate(
    pendulum_hamiltonian, 
    q0=np.array([np.pi/4]),  # Initial angle
    p0=np.array([0.0]),      # Initial momentum
    t_span=(0, 10),
    dt=0.01,
    method='yoshida4'
)

print(f"Energy drift: {np.std(result.energy)/np.mean(result.energy):.2e}")
```

### Example 2: Uncertainty Propagation
```python
from phizzy.uncertainties import propagate_uncertainties_symbolic

# Experimental measurements with uncertainties
variables = {
    'R': (10.0, 0.1),    # Resistance: 10.0 ± 0.1 Ω
    'I': (2.0, 0.05),    # Current: 2.0 ± 0.05 A
}

# Calculate power with uncertainty
result = propagate_uncertainties_symbolic(
    expression="I**2 * R",
    variables=variables,
    correlations={('R', 'I'): 0.3}  # 30% correlation
)

print(f"Power = {result.numeric_result}")  # 40.0 ± 2.1 W
```

### Example 3: Phase Transition Detection
```python
from phizzy.transitions import phase_transition_detect

# Simulated magnetization data
temperature = np.linspace(0, 4, 200)
magnetization = np.tanh(2 * (2.269 - temperature))  # Ising critical temperature
magnetization += 0.05 * np.random.randn(200)  # Add noise

result = phase_transition_detect(
    magnetization,
    control_parameter=temperature,
    method='derivatives'
)

print(f"Critical temperature: {result.transition_points[0]:.3f}")
```

## 📚 Documentation

### Detailed Examples
See the [examples/](examples/) directory for comprehensive tutorials:
- `example_symplectic.py` - Hamiltonian mechanics examples
- `example_uncertainties.py` - Error propagation workflows
- `example_monte_carlo.py` - Statistical physics simulations
- `example_phase_transitions.py` - Critical phenomena analysis
- And more...

### API Reference
Each function has extensive docstrings. Use Python's help system:
```python
from phizzy import symplectic_integrate
help(symplectic_integrate)
```

## 🧪 Validation

All functions have been rigorously validated:
- ✅ Analytical solutions comparison
- ✅ Cross-validation with SciPy/NumPy
- ✅ Performance benchmarking
- ✅ Unit test coverage

Run validation suite:
```bash
python simple_validation.py  # Quick functionality check
python run_validation.py     # Comprehensive validation
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional physics models
- Performance optimizations
- Documentation improvements
- Bug fixes

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Phizzy was created to fill critical gaps in the Python physics ecosystem identified through comprehensive analysis of existing libraries.

## 📊 Citation

If you use Phizzy in your research, please cite:
```bibtex
@software{phizzy2024,
  title = {Phizzy: Advanced Physics Library for Python},
  author = {Elliott, John},
  year = {2024},
  url = {https://github.com/PhysicistJohn/Phizzy}
}
```

---
**Note**: Phizzy is under active development. APIs may change between versions.