#!/usr/bin/env python
"""
Update requirements.txt to include all necessary dependencies for diagnostic plots.
This ensures that the required packages are installed for plot generation.
"""
import os
import subprocess
import sys

# List of packages needed for diagnostic plots
DIAGNOSTIC_PLOT_PACKAGES = [
    "matplotlib>=3.5.0",
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "scipy>=1.7.0",
    "seaborn>=0.11.0",
    "statsmodels>=0.13.0",
    "pillow>=9.0.0"  # For image manipulation
]

def update_requirements(requirements_path, output_path=None):
    """Update requirements file with diagnostic plot dependencies
    
    Args:
        requirements_path: Path to the original requirements.txt
        output_path: Path to save the updated requirements (if None, overwrites original)
    """
    if output_path is None:
        output_path = requirements_path
    
    # Read existing requirements
    existing_packages = set()
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name without version
                    package_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                    existing_packages.add(package_name)
    
    # Add diagnostic plot packages that aren't already included
    new_requirements = []
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            new_requirements = f.read().splitlines()
    
    added_packages = []
    for package in DIAGNOSTIC_PLOT_PACKAGES:
        package_name = package.split('>=')[0].split('==')[0].strip()
        if package_name not in existing_packages:
            new_requirements.append(package)
            added_packages.append(package)
    
    # Write updated requirements
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_requirements))
        # Add a newline at the end of the file if it doesn't already have one
        if new_requirements and not new_requirements[-1].endswith('\n'):
            f.write('\n')
    
    if added_packages:
        print(f"Added the following packages to {output_path}:")
        for package in added_packages:
            print(f"  - {package}")
    else:
        print(f"No new packages needed to be added to {output_path}")
    
    return added_packages

def install_requirements(requirements_path):
    """Install requirements using pip
    
    Args:
        requirements_path: Path to the requirements.txt file
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print(f"Successfully installed dependencies from {requirements_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")

def main():
    if len(sys.argv) > 1:
        requirements_path = sys.argv[1]
    else:
        requirements_path = 'requirements.txt'
    
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = None
    
    # Update requirements
    added_packages = update_requirements(requirements_path, output_path)
    
    # Ask if user wants to install the added packages
    if added_packages:
        response = input("Would you like to install the added packages now? (y/n) ")
        if response.lower() in ['y', 'yes']:
            install_requirements(output_path or requirements_path)

if __name__ == "__main__":
    main() 