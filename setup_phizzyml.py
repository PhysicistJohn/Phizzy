from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phizzyML",
    version="0.1.0",
    author="PhizzyML Contributors",
    description="A comprehensive Python library for physics-informed machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/phizzyML",
    packages=find_packages(include=["phizzyML*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "scipy>=1.7.0",
        "sympy>=1.9",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "h5py>=3.0.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "root": [
            "uproot>=4.0.0",
        ],
        "quantum": [
            "pennylane>=0.25.0",
            "qiskit>=0.36.0",
        ],
        "all": [
            "uproot>=4.0.0",
            "pennylane>=0.25.0",
            "qiskit>=0.36.0",
        ],
    },
)