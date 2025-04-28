#!/usr/bin/env python
"""
Development setup script for GreenOps Agent.

This script ensures that the greenops_agent package can be imported correctly
in development mode without requiring a full installation.

Usage:
    python setup_dev.py

This adds the necessary paths to the PYTHONPATH environment variable for your shell.
"""

import os
import sys
import subprocess
import platform

def main():
    """Set up the development environment."""
    # Get the absolute path to the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(script_dir, "src")

    # Check if src directory exists
    if not os.path.exists(src_path):
        print(f"Error: Source directory {src_path} does not exist")
        sys.exit(1)

    # Print environment info
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Install the package in development mode
    print("\n=== Installing in development mode ===")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])

    # Add instructions for setting PYTHONPATH
    print("\n=== Development Environment Setup ===")
    print("To ensure Python can find the greenops_agent package, you can:")
    print("\n1. Add the src directory to PYTHONPATH for the current session:")

    if platform.system() == "Windows":
        print(f'   set PYTHONPATH={src_path};%PYTHONPATH%')
    else:  # macOS or Linux
        print(f'   export PYTHONPATH={src_path}:$PYTHONPATH')

    print("\n2. Or add the following at the start of your Python scripts:")
    print('   import sys, os')
    print('   sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))')
    
    print("\n3. Or continue using the modified test script with the path already added")

    print("\nSetup complete!")

if __name__ == "__main__":
    main()