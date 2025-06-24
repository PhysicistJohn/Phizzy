# Contributing to PhizzyML

We welcome contributions to PhizzyML! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/phizzyML.git`
3. Create a new branch: `git checkout -b feature-name`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Testing
- Write tests for all new features
- Ensure all tests pass: `pytest`
- Aim for >80% code coverage: `pytest --cov=phizzyML`

### Documentation
- Document all public functions and classes
- Include docstrings with parameter descriptions
- Add examples where helpful

## Submitting Changes

1. Ensure all tests pass
2. Run linters: `flake8 phizzyML/`
3. Format code: `black phizzyML/`
4. Commit with clear messages
5. Push to your fork
6. Submit a pull request

## Areas for Contribution

- **New Features**: Implement missing physics ML functionality
- **Bug Fixes**: Fix issues and improve stability
- **Documentation**: Improve docs and add examples
- **Tests**: Increase test coverage
- **Performance**: Optimize existing code

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Collaborate openly

Thank you for contributing to PhizzyML!