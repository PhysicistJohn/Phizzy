"""Setup script for PhizzyML - Physics-Informed Machine Learning Library."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Core requirements
install_requires = [
    "torch>=1.9.0",
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "sympy>=1.9",
    "matplotlib>=3.4.0",
    "h5py>=3.0.0",
    "pandas>=1.3.0",
    "scikit-learn>=0.24.0",
]

# Optional requirements
extras_require = {
    'dev': [
        'pytest>=6.0',
        'pytest-cov',
        'black',
        'flake8',
        'mypy',
        'sphinx',
        'sphinx-rtd-theme',
    ],
    'viz': [
        'plotly>=5.0',
        'seaborn>=0.11',
    ],
    'hpc': [
        'numba>=0.54',
        'cupy>=9.0',  # For GPU acceleration
    ],
}

setup(
    name="phizzyml",
    version="1.0.0",
    author="John Elliott",
    author_email="your.email@example.com",
    description="Physics-Informed Machine Learning Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/phizzyML",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/phizzyML/issues",
        "Documentation": "https://phizzyml.readthedocs.io",
        "Source Code": "https://github.com/yourusername/phizzyML",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    keywords=[
        "physics",
        "machine learning",
        "deep learning",
        "scientific computing",
        "uncertainty quantification",
        "dimensional analysis",
        "symplectic integration",
        "conservation laws",
        "pytorch",
    ],
)