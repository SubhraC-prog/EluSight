from setuptools import setup, find_packages

setup(
    name="elusight",
    version="0.1.0",
    description="EluSight: A comprehensive framework for explainability, uncertainty, and trust in AI systems",
    author="SubhraC-prog",
    author_email="author@elusight.dev",
    url="https://github.com/SubhraC-prog/EluSight",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "ruff>=0.1.0",
            "mypy>=0.990",
            "isort>=5.11",
            "pre-commit>=2.20",
        ],
        "all": [
            "optuna>=3.0",
            "pymoo>=0.6.0",
            "scikit-optimize>=0.9.0",
            "gpytorch>=1.9.0",
            "botorch>=0.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
