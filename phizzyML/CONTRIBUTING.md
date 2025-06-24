# Contributing to PhizzyML

First off, thank you for considering contributing to PhizzyML! It's people like you that make PhizzyML such a great tool for the physics and machine learning community.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what you expected**
- **Include Python version, PyTorch version, and OS information**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List any alternative solutions you've considered**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the style guidelines
6. Issue that pull request!

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/phizzyML.git
cd phizzyML
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Run tests to ensure everything is working:
```bash
pytest tests/
```

## Style Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for code formatting
- Use type hints where appropriate
- Write docstrings for all public functions/classes

### Example Code Style:
```python
def calculate_energy(mass: float, velocity: torch.Tensor) -> torch.Tensor:
    """
    Calculate kinetic energy using classical mechanics.
    
    Args:
        mass: Mass in kilograms
        velocity: Velocity tensor in m/s
        
    Returns:
        Kinetic energy in Joules
    """
    return 0.5 * mass * torch.sum(velocity**2, dim=-1)
```

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Documentation

- Use clear, concise language
- Include code examples for new features
- Update the README.md if needed
- Add docstrings to all new functions and classes

## Testing

- Write tests for any new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>90%)
- Include both unit tests and integration tests

### Running Tests:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=phizzyml

# Run specific test file
pytest tests/test_units.py
```

## Areas for Contribution

We especially welcome contributions in these areas:

1. **New Physics Integrators**
   - Geometric integrators
   - Multi-scale methods
   - Adaptive timestep algorithms

2. **Additional Constraints**
   - Gauge invariance
   - Thermodynamic constraints
   - Quantum mechanical constraints

3. **Performance Optimizations**
   - GPU acceleration
   - Distributed computing support
   - Memory optimization

4. **Documentation**
   - Tutorials and examples
   - API documentation
   - Theory guides

5. **Quantum-Classical Interface**
   - Wavefunction representations
   - Measurement operators
   - Hybrid algorithms

## Questions?

Feel free to open an issue with your question or reach out to the maintainers directly.

Thank you for contributing to PhizzyML! 🎉