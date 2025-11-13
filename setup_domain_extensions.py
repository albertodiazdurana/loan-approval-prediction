#!/usr/bin/env python3
"""
Domain-Specific Package Installer
Project: Loan Approval Prediction - Binary Classification

Installs packages specific to ML classification tasks with CSV data.
"""

import subprocess
import sys
from pathlib import Path

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        __import__(package_name.replace('-', '_'))
        return True
    except ImportError:
        return False

def install_package(package_name, display_name=None):
    """Install a single package with progress indication"""
    display = display_name or package_name
    
    if check_package_installed(package_name):
        print(f"OK: {display} already installed - skipping")
        return True
    
    print(f"Installing {display}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name, "--break-system-packages"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print(f"OK: {display} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install {display}: {e}")
        return False

def update_requirements_file(packages):
    """Update requirements.txt with new packages"""
    print_section("Updating requirements.txt")
    
    req_file = Path("requirements.txt")
    
    # Read existing requirements
    existing = set()
    if req_file.exists():
        with open(req_file, 'r') as f:
            existing = {line.strip() for line in f if line.strip() and not line.startswith('#')}
    
    # Add new packages
    new_packages = set(packages) - existing
    
    if new_packages:
        with open(req_file, 'a') as f:
            f.write("\n# Domain-specific packages for ML Classification\n")
            for pkg in sorted(new_packages):
                f.write(f"{pkg}\n")
        print(f"OK: Added {len(new_packages)} new packages to requirements.txt")
    else:
        print("OK: All packages already in requirements.txt")

def verify_installations():
    """Verify all domain packages are importable"""
    print_section("Verifying Installations")
    
    verification_tests = [
        ("sklearn", "scikit-learn"),
        ("seaborn", "seaborn"),
        ("imblearn", "imbalanced-learn")
    ]
    
    all_passed = True
    for module_name, display_name in verification_tests:
        try:
            __import__(module_name)
            print(f"OK: {display_name} verified")
        except ImportError:
            print(f"ERROR: {display_name} verification failed")
            all_passed = False
    
    return all_passed

def main():
    """Main installation routine"""
    print_section("Domain Extension Installer - Binary Classification")
    print("Project: Loan Approval Prediction")
    print("Domain: ML Classification with CSV data")
    
    # Define domain-specific packages
    domain_packages = [
        ("scikit-learn", "scikit-learn (ML models)"),
        ("seaborn", "seaborn (statistical visualization)"),
        ("imbalanced-learn", "imbalanced-learn (class imbalance handling)")
    ]
    
    print(f"\nPackages to install: {len(domain_packages)}")
    for pkg, desc in domain_packages:
        print(f"  - {desc}")
    
    print_section("Installing Packages")
    
    # Install packages
    results = []
    for package_name, display_name in domain_packages:
        success = install_package(package_name, display_name)
        results.append((display_name, success))
    
    # Update requirements.txt
    package_specs = [pkg for pkg, _ in domain_packages]
    update_requirements_file(package_specs)
    
    # Verify installations
    verification_passed = verify_installations()
    
    # Summary
    print_section("Installation Summary")
    successful = sum(1 for _, success in results if success)
    print(f"Packages processed: {len(results)}")
    print(f"Successfully installed/verified: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    if verification_passed and successful == len(results):
        print("\nOK: All domain-specific packages ready for use!")
        print("\nYou can now:")
        print("  - Train classification models (LogisticRegression, DecisionTree, RandomForest)")
        print("  - Create statistical visualizations (seaborn)")
        print("  - Handle class imbalance (SMOTE, class_weight)")
        return 0
    else:
        print("\nWARNING: Some packages failed to install or verify")
        print("Please check the errors above and try manual installation:")
        print("  pip install scikit-learn seaborn imbalanced-learn --break-system-packages")
        return 1

if __name__ == "__main__":
    sys.exit(main())