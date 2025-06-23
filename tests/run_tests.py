"""
Test runner for the Statistical Model Suggester application.
This script runs all tests and provides a comprehensive report.
"""
import pytest
import sys
import os
from pathlib import Path
# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
def run_all_tests():
    """Run all tests with appropriate configuration."""
    test_args = [
        '-v',  # Verbose output
        '--tb=short',  # Short traceback format
        '--strict-markers',  # Strict marker handling
        '--disable-warnings',  # Disable warnings for cleaner output
        '--color=yes',  # Colored output
        str(Path(__file__).parent)  # Test directory
    ]
    # Add coverage if pytest-cov is available
    try:
        import pytest_cov
        test_args.extend([
            '--cov=.',  # Coverage for current directory
            '--cov-report=html',  # HTML coverage report
            '--cov-report=term-missing',  # Terminal coverage with missing lines
        ])
    except ImportError:
        print("pytest-cov not available, running without coverage")
    return pytest.main(test_args)
def run_specific_test_category(category):
    """Run tests for a specific category."""
    test_files = {
        'models': 'test_models.py',
        'auth': 'test_auth_routes.py',
        'main': 'test_main_routes.py',
        'admin': 'test_admin_routes.py',
        'utils': 'test_utils.py'
    }
    if category not in test_files:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(test_files.keys())}")
        return 1
    test_file = Path(__file__).parent / test_files[category]
    return pytest.main(['-v', str(test_file)])
if __name__ == '__main__':
    if len(sys.argv) > 1:
        category = sys.argv[1]
        exit_code = run_specific_test_category(category)
    else:
        exit_code = run_all_tests()
    sys.exit(exit_code)
