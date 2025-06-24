# PhizzyML - Physics-Informed Machine Learning Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PhizzyML is a comprehensive Python library that bridges the gap between physics and machine learning. It addresses the top 10 critical gaps identified in existing physics ML libraries, providing researchers and engineers with powerful tools for physics-informed machine learning.

## 🌟 Key Features

### 1. **Dimensional Analysis & Unit Safety**
- Automatic unit tracking and conversion
- Compile-time dimensional correctness checking
- Seamless integration with PyTorch tensors

### 2. **Uncertainty Quantification**
- Automatic error propagation through calculations
- Support for correlated uncertainties
- Monte Carlo and analytical methods

### 3. **Symplectic Integration**
- Energy-preserving numerical integrators
- Multiple algorithms: Leapfrog, Yoshida, Forest-Ruth, PEFRL
- Ideal for Hamiltonian systems

### 4. **Physics-Constrained Neural Networks**
- Layers that enforce conservation laws (energy, momentum, angular momentum)
- Hard constraint enforcement during training
- Lagrange multiplier methods

### 5. **Causality-Aware Layers**
- Neural network components that respect temporal causality
- Prevents information flow from future to past
- Essential for time-series physics modeling

### 6. **Physics-Informed Optimizers**
- Energy minimization algorithms
- Variational methods
- Langevin dynamics for statistical physics

### 7. **Mesh Generation & Differential Operators**
- Flexible mesh generation for complex geometries
- Laplacian, gradient, and divergence operators
- Adaptive mesh refinement

### 8. **Experimental Data Integration**
- Direct loading from HDF5, ROOT, CSV formats
- Automatic uncertainty handling
- Metadata preservation

### 9. **Multi-Scale Physics**
- Seamless integration across different scales
- Automatic scale bridging
- Hierarchical modeling support

### 10. **Quantum-Classical Interface**
- Tools for hybrid quantum-classical simulations
- Wavefunction handling
- Measurement operators

## 📦 Installation

### From Source
```bash
git clone https://github.com/yourusername/phizzyML.git
cd phizzyML
pip install -e .
```

### Dependencies
```bash
pip install torch numpy scipy sympy matplotlib h5py scikit-learn pandas
```

## 🚀 Quick Start

### Example 1: Dimensional Analysis
```python
from phizzyML.core.units import Unit, Quantity

# Define quantities with units
force = Quantity(10.0, Unit.newton)
distance = Quantity(5.0, Unit.meter)

# Automatic unit checking
work = force * distance  # Result: 50.0 J
print(f"Work done: {work}")

# Unit errors caught automatically
# velocity = force + distance  # Raises DimensionalError!
```

### Example 2: Uncertainty Propagation
```python
from phizzyML.uncertainty import UncertainTensor

# Measurements with uncertainties
mass = UncertainTensor(value=2.0, uncertainty=0.05)  # 2.0 ± 0.05 kg
velocity = UncertainTensor(value=10.0, uncertainty=0.2)  # 10.0 ± 0.2 m/s

# Automatic error propagation
kinetic_energy = 0.5 * mass * velocity**2
print(f"KE: {kinetic_energy.value:.2f} ± {kinetic_energy.uncertainty:.2f}")
```

### Example 3: Physics-Constrained Neural Network
```python
import torch
import torch.nn as nn
from phizzyML.constraints import EnergyConservingLayer

class PhysicsNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(6, 64)
        self.output = nn.Linear(64, 6)
        self.energy_constraint = EnergyConservingLayer()
    
    def forward(self, x):
        h = torch.relu(self.hidden(x))
        out = self.output(h)
        # Enforce energy conservation
        return self.energy_constraint(out)

# Train network with guaranteed energy conservation
model = PhysicsNN()
```

### Example 4: Symplectic Integration
```python
from phizzyML.integrators import LeapfrogIntegrator
import torch

def pendulum_hamiltonian(q, p, g=9.81, L=1.0):
    """Simple pendulum Hamiltonian."""
    kinetic = p**2 / (2 * L**2)
    potential = g * L * (1 - torch.cos(q))
    return kinetic + potential

# Initialize integrator
integrator = LeapfrogIntegrator(dt=0.01)

# Initial conditions
q0 = torch.tensor([0.1])  # Small angle
p0 = torch.tensor([0.0])  # Zero initial momentum

# Integrate for 1000 steps
q_trajectory, p_trajectory = integrator.integrate(
    q0, p0, pendulum_hamiltonian, n_steps=1000
)

# Energy is conserved to machine precision!
```

## 📚 Documentation

### Core Modules

- **`phizzyML.core`** - Units, dimensional analysis, and core utilities
- **`phizzyML.uncertainty`** - Uncertainty propagation and error analysis
- **`phizzyML.integrators`** - Symplectic and geometric integrators
- **`phizzyML.constraints`** - Physics constraint layers
- **`phizzyML.layers`** - Causality-aware neural network layers
- **`phizzyML.optimizers`** - Physics-informed optimization algorithms
- **`phizzyML.geometry`** - Mesh generation and differential operators
- **`phizzyML.data`** - Experimental data interfaces

### Examples

See the `examples/` directory for comprehensive examples:
- `harmonic_oscillator.py` - Symplectic integration demo
- `uncertainty_propagation.py` - Error analysis workflows
- `constrained_dynamics.py` - Conservation law enforcement
- `complete_physics_ml_workflow.py` - End-to-end physics ML pipeline

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run validation suite:
```bash
python phizzyML_final_validation.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

Areas for contribution:
- Additional integrator algorithms
- More physics constraint types
- Extended quantum-classical interfaces
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

PhizzyML was created to address critical gaps in the physics machine learning ecosystem. We thank the physics and ML communities for their invaluable feedback and contributions.

## 📊 Citation

If you use PhizzyML in your research, please cite:

```bibtex
@software{phizzyml2024,
  title = {PhizzyML: Physics-Informed Machine Learning Library},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/phizzyML}
}
```

## 🚦 Status

- ✅ Core functionality implemented
- ✅ Comprehensive test coverage
- ✅ Examples and documentation
- ✅ Production ready

## 📞 Contact

For questions and support:
- GitHub Issues: [Create an issue](https://github.com/yourusername/phizzyML/issues)
- Email: your.email@example.com

---
*PhizzyML - Bridging Physics and Machine Learning*