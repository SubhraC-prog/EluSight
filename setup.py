#!/usr/bin/env python3
"""
EluSight - Chromatographic Decision Intelligence Framework

Setup script for installing EluSight package.
"""

import os
import re
import sys
from setuptools import setup, find_packages

# ============================================
# Version extraction
# ============================================
def get_version():
    """Extract version from __init__.py"""
    version_file = os.path.join(os.path.dirname(__file__), 'elusight', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        content = f.read()
        version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
        if version_match:
            return version_match.group(1)
    return "1.0.0"

# ============================================
# Long description from README
# ============================================
def get_long_description():
    """Read long description from README.md"""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    except (IOError, OSError):
        return "EluSight - Chromatographic Decision Intelligence Framework"

# ============================================
# Requirements extraction
# ============================================
def get_requirements():
    """Extract requirements from requirements.txt"""
    requirements = []
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    try:
        with open(req_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments, empty lines, and optional dependencies
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Skip optional dependencies (commented out with #)
                    if not line.startswith('#'):
                        # Remove any inline comments
                        line = line.split('#')[0].strip()
                        if line:
                            requirements.append(line)
    except (IOError, OSError):
        # Fallback requirements
        requirements = [
            'numpy>=1.24.0',
            'pandas>=2.0.0',
            'scipy>=1.10.0',
            'pydantic>=2.0.0',
            'scikit-learn>=1.3.0',
            'shap>=0.42.0',
            'plotly>=5.17.0',
            'streamlit>=1.25.0',
            'fastapi>=0.100.0',
            'uvicorn>=0.23.0',
        ]
    
    return requirements

# ============================================
# Setup configuration
# ============================================
setup(
    # Basic package information
    name="elusight",
    version=get_version(),
    description="EluSight - Chromatographic Decision Intelligence Framework",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # Author information
    author="SubhraC-prog",
    author_email="subhrac@example.com",
    maintainer="SubhraC-prog",
    maintainer_email="subhrac@example.com",
    
    # URLs
    url="https://github.com/SubhraC-prog/EluSight",
    project_urls={
        "Documentation": "https://github.com/SubhraC-prog/EluSight/docs",
        "Source": "https://github.com/SubhraC-prog/EluSight",
        "Issue Tracker": "https://github.com/SubhraC-prog/EluSight/issues",
        "Changelog": "https://github.com/SubhraC-prog/EluSight/releases",
    },
    
    # Package configuration
    packages=find_packages(
        where=".",
        exclude=["tests", "tests.*", "docs", "docs.*", "examples", "examples.*"]
    ),
    include_package_data=True,
    package_data={
        'elusight': ['py.typed', '*.pyi'],
        'elusight.dashboard': ['assets/*', 'static/*'],
    },
    zip_safe=False,
    
    # Python version requirement
    python_requires=">=3.11",
    
    # Dependencies
    install_requires=get_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-xdist>=3.3.0',
            'pytest-timeout>=2.1.0',
            'pytest-mock>=3.11.0',
            'pytest-benchmark>=4.0.0',
            'black>=23.0.0',
            'ruff>=0.0.280',
            'isort>=5.12.0',
            'mypy>=1.4.0',
            'pre-commit>=3.3.0',
            'bandit>=1.7.5',
            'safety>=2.3.0',
        ],
        'gp': [
            'gpytorch>=1.9.0',
            'botorch>=0.8.0',
            'torch>=2.0.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=2.0.0',
            'sphinx-autodoc-typehints>=1.24.0',
        ],
        'reports': [
            'weasyprint>=60.0',
            'python-docx>=0.8.11',
            'markdown>=3.4.0',
            'jinja2>=3.1.0',
        ],
        'explainability': [
            'alibi>=0.9.0',
            'dowhy>=0.11.0',
        ],
        'integrations': [
            'optuna>=3.3.0',
            'pymoo>=0.6.1',
            'scikit-optimize>=0.9.0',
        ],
        'all': [
            'elusight[dev,gp,docs,reports,explainability,integrations]',
        ],
    },
    
    # Console scripts (entry points)
    entry_points={
        'console_scripts': [
            'elusight-api=elusight.api.routes:run_api',
            'elusight-dashboard=elusight.dashboard.app:run_dashboard',
            'elusight-verify=verify_installation:main',
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Typing :: Typed",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "chromatography",
        "hplc", "uplc", "gc", "lc-ms",
        "optimization",
        "machine-learning",
        "decision-intelligence",
        "aqbd",
        "pareto-front",
        "bayesian-optimization",
        "method-development",
        "analytical-chemistry",
    ],
    
    # Other metadata
    platforms=["any"],
    license="MIT",
    classifiers_extra={
        "License :: OSI Approved :: MIT License",
    },
)

# ============================================
# Installation verification message
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔬 EluSight - Chromatographic Decision Intelligence Framework")
    print("=" * 60)
    print(f"Version: {get_version()}")
    print(f"Python: {sys.version.split()[0]}")
    print("-" * 60)
    print("Installation complete!")
    print("\nQuick start:")
    print("  from elusight import Ingestor, TrustEngine")
    print("  methods = Ingestor.from_json('results.json')")
    print("\nRun dashboard:")
    print("  streamlit run elusight/dashboard/app.py")
    print("\nRun API server:")
    print("  elusight-api")
    print("=" * 60)