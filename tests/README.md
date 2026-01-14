# Test Suite for `plot_sky_coverage_jan13_19.py`

This directory contains a comprehensive unit test suite for the sky coverage plotting script.

## Test Structure

The test suite is organized into several test classes, each focusing on a specific component of the script:

### 1. `TestBuildGBTTelescope`
Tests for the `build_gbt_telescope()` function that constructs the Green Bank Telescope instance.

**Test Cases:**
- **`test_returns_telescope_instance`**: Verifies the function returns a `Telescope` object
- **`test_telescope_location`**: Validates the telescope location is set to Green Bank Telescope
- **`test_azimuth_slew_rate`**: Checks azimuth slew rate is 17.6 deg/min
- **`test_tip_slew_rate`**: Checks tip (elevation) slew rate is 17.6 deg/min
- **`test_azimuth_acceleration`**: Validates azimuth acceleration is 0.2 deg/s²
- **`test_tip_acceleration`**: Validates tip acceleration is 0.2 deg/s²
- **`test_altitude_limits`**: Ensures altitude range is 5° to 90°
- **`test_azimuth_limits`**: Ensures azimuth range is -5° to 365°
- **`test_angular_resolution`**: Verifies angular resolution is 1 deg²

**Purpose**: These tests ensure the telescope configuration matches the specifications from the notebook, which is critical for accurate sky coverage simulations.

### 2. `TestBuildRegularScans`
Tests for the `build_regular_scans()` function that creates the scan pattern sequence.

**Test Cases:**
- **`test_returns_list`**: Verifies the function returns a list
- **`test_all_scans_are_backandforthscan`**: Ensures all scans are `BackAndForthScan` instances
- **`test_correct_number_of_scans`**: Validates exactly 8 scans are returned (4 pairs)
- **`test_alternating_scan_modes`**: Checks scans alternate between azimuth and tip modes
- **`test_azimuth_scan_parameters`**: Validates azimuth scan parameters (180° max, fixed at 38° elevation)
- **`test_tip_scan_parameters`**: Validates tip scan parameters (38° to 5° range, fixed at 180° azimuth)

**Purpose**: These tests verify the scan pattern matches the notebook configuration, ensuring consistent observational coverage.

### 3. `TestBuildSessionsForRange`
Tests for the `build_sessions_for_range()` function that creates scan sessions for a date range.

**Test Cases:**
- **`test_returns_list`**: Verifies the function returns a list
- **`test_all_sessions_are_scansession`**: Ensures all items are `ScanSession` instances
- **`test_single_day_creates_one_session`**: Validates one day creates one session
- **`test_week_range_creates_seven_sessions`**: Verifies Jan 13-19 creates 7 sessions
- **`test_sessions_start_at_20_00`**: Checks all sessions start at 20:00 local time
- **`test_sessions_use_correct_timezone`**: Validates Eastern Time Zone is used
- **`test_sessions_are_sequential_days`**: Ensures sessions are created for consecutive days
- **`test_all_sessions_use_same_telescope`**: Verifies all sessions share the telescope instance
- **`test_sessions_have_regular_scans`**: Checks sessions use the regular scan pattern

**Purpose**: These tests ensure the date range processing works correctly and sessions are properly configured for the specified time period.

### 4. `TestParseArgs`
Tests for the `parse_args()` function that handles command-line argument parsing.

**Test Cases:**
- **`test_default_values`**: Verifies default values (galactic frame, threshold=10)
- **`test_frame_argument_galactic`**: Tests parsing of `--frame galactic`
- **`test_frame_argument_ra_dec`**: Tests parsing of `--frame ra_dec`
- **`test_threshold_argument`**: Tests parsing of `--threshold` with integer value
- **`test_both_arguments`**: Validates both arguments can be provided together
- **`test_invalid_frame_raises_error`**: Ensures invalid frame choices raise errors
- **`test_invalid_threshold_type_raises_error`**: Ensures non-integer thresholds raise errors

**Purpose**: These tests validate command-line interface behavior and error handling.

### 5. `TestIntegration`
Integration tests that verify components work together correctly.

**Test Cases:**
- **`test_full_pipeline_jan13_19`**: Tests the complete pipeline for Jan 13-19 date range
- **`test_telescope_and_scans_compatibility`**: Verifies telescope and scans are compatible
- **`test_sessions_can_be_created_with_scans`**: Ensures sessions can be instantiated with the scan pattern

**Purpose**: These tests validate that all components integrate properly and can be used together as intended.

## Running the Tests

**IMPORTANT**: Make sure to activate the skymap virtual environment before running tests!

From the project root directory:

```bash
# Activate the virtual environment first
# For conda:
conda activate skymap
# OR for venv:
source /path/to/skymap/venv/bin/activate

# Then run tests:
# Run all tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run a specific test class
python -m pytest tests/test_plot_sky_coverage_jan13_19.py::TestBuildGBTTelescope

# Run a specific test
python -m pytest tests/test_plot_sky_coverage_jan13_19.py::TestBuildGBTTelescope::test_azimuth_slew_rate

# Run with coverage report
python -m pytest tests/ --cov=scripts.plot_sky_coverage_jan13_19 --cov-report=html
```

Alternatively, using unittest:

```bash
# Make sure venv is activated first!
python -m unittest discover tests
python -m unittest tests.test_plot_sky_coverage_jan13_19
```

## Test Coverage

The test suite aims to achieve:
- **Function coverage**: All public functions are tested
- **Parameter validation**: All function parameters and return values are validated
- **Edge cases**: Boundary conditions and error cases are tested
- **Integration**: Components are tested together to ensure compatibility

## Dependencies

The tests require:
- `unittest` (standard library)
- `unittest.mock` (standard library)
- `astropy` (for units and coordinates)
- `AltAzSim` package (the module being tested)

All dependencies should be available if the main project dependencies are installed.

