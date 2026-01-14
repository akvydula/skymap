#!/usr/bin/env python
"""
Test runner script that uses pytest for better output formatting.

Usage:
    python tests/run_tests.py
    python tests/run_tests.py -v
    python tests/run_tests.py::TestBuildGBTTelescope
"""

import sys
import pytest

if __name__ == "__main__":
    # Use pytest's programmatic API
    # Pass through any command-line arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else ["-v", "--tb=short"]
    
    # Add the test file if no specific test is specified
    if not any("::" in arg or arg.startswith("test_") for arg in args):
        args.insert(0, __file__.replace("run_tests.py", "test_plot_sky_coverage_jan13_19.py"))
    
    exit_code = pytest.main(args)
    sys.exit(exit_code)

