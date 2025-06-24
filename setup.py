from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phizzy",
    version="0.1.0",
    author="Phizzy Development Team",
    author_email="phizzy@example.com",
    description="Advanced Physics Library for Python - Essential functions missing from existing libraries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/phizzy",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "sympy>=1.9",
        "pint>=0.18",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
        "numba>=0.55.0",
        "opt-einsum>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
            "jax>=0.3.0",
            "jaxlib>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phizzy=phizzy.cli:main",
        ],
    },
)