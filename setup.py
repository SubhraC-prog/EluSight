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
# Version extraction from root __init__.py
# ============================================
def get_version():
    """Extract version from root __init__.py"""
    init_file = os.path.join(os.path.dirname(__file__), '__init__.py')
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
            version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
            if version_match:
                return version_match.group(1)
    except FileNotFoundError:
        pass
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
                if line and not line.startswith('#') and not line.startswith('//'):
                    line = line.split('#')[0].strip()
                    if line:
                        requirements.append(line)
    except (IOError, OSError):
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
    
    # URLs
    url="https://github.com/SubhraC-prog/EluSight",
    project_urls={
        "Documentation": "https://github.com/SubhraC-prog/EluSight/docs",
        "Source": "https://github.com/SubhraC-prog/EluSight",
        "Issue Tracker": "https://github.com/SubhraC-prog/EluSight/issues",
    },
    
    # Package configuration - finds all subdirectories at root level
    packages=find_packages(
        where=".",
        exclude=["tests", "tests.*", "docs", "docs.*", "examples", "examples.*"]
    ),
    package_dir={"": "."},
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.11",
    
    # Dependencies
    install_requires=get_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'ruff>=0.0.280',
            'mypy>=1.4.0',
        ],
        'gp': [
            'gpytorch>=1.9.0',
            'botorch>=0.8.0',
            'torch>=2.0.0',
        ],
    },
    
    # Console scripts (entry points)
    entry_points={
        'console_scripts': [
            'elusight-api=api.routes:run_api',
            'elusight-dashboard=dashboard.app:run_dashboard',
            'elusight-verify=verify_installation:main',
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords
    keywords=[
        "chromatography", "hplc", "optimization", 
        "machine-learning", "decision-intelligence", "aqbd"
    ],
    
    license="MIT",
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
    print("  from api.routes import app")
    print("  from constraints.engine import ConstraintEngine")
    print("  from trust.engine import TrustEngine")
    print("  from reasoning.engine import ReasoningEngine")
    print("\nRun dashboard:")
    print("  streamlit run dashboard/app.py")
    print("\nRun API server:")
    print("  uvicorn api.routes:app --reload")
    print("=" * 60)
