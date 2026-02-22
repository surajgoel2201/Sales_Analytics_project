#!/bin/bash

# Sales Analytics Project Setup Script
# This script sets up the environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "Sales Analytics Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/raw data/processed reports/figures notebooks

# Run a quick test
echo ""
echo "Testing installation..."
python -c "import yfinance; import pandas; import matplotlib; print('✓ All packages installed successfully')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the main script: python main.py"
echo "  3. Or open Jupyter: jupyter notebook"
echo ""
echo "The analysis will fetch live data and generate reports in the 'reports' directory."
echo ""
