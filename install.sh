#!/bin/bash

echo "========================================="
echo "EluSight Installation"
echo "========================================="

# Check Python version
python3 --version

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install EluSight
pip install -e .

# Run verification
python -c "from elusight import __version__; print(f'EluSight {__version__} installed successfully!')"

echo ""
echo "Quick start:"
echo "  source venv/bin/activate"
echo "  python -c \"from elusight import Ingestor; print('Ready!')\""
echo ""
echo "Run dashboard:"
echo "  streamlit run elusight/dashboard/app.py"