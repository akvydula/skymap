"""
Unit test suite for plot_sky_coverage_jan13_19.py

This test suite validates the functionality of the sky coverage plotting script,
including telescope configuration, scan pattern creation, session building, and
argument parsing.

NOTE: Make sure to activate the skymap virtual environment before running tests:
    source /path/to/skymap/venv/bin/activate  # or conda activate skymap
    python -m pytest tests/
"""

import sys
import os
from pathlib import Path

# Add project root and AltAzSim directory to Python path to ensure imports work
project_root = Path(__file__).parent.parent.resolve()
altazsim_dir = project_root / "AltAzSim"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(altazsim_dir) not in sys.path:
    sys.path.insert(0, str(altazsim_dir))

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.coordinates import EarthLocation

from scripts.plot_sky_coverage_jan13_19 import (
    build_gbt_telescope,
    build_regular_scans,
    build_sessions_for_range,
    parse_args,
    EST_TZ,
)
from AltAzSim import Telescope, ScanSession, ScanProject
from AltAzSim.scantypes import BackAndForthScan


class TestBuildGBTTelescope(unittest.TestCase):
    """Test cases for build_gbt_telescope() function."""

    def test_returns_telescope_instance(self):
        """Test that the function returns a Telescope instance."""
        telescope = build_gbt_telescope()
        self.assertIsInstance(telescope, Telescope)

    def test_telescope_location(self):
        """Test that the telescope location is set to Green Bank Telescope."""
        telescope = build_gbt_telescope()
        # Verify location is set (EarthLocation.of_site should return GBT location)
        self.assertIsInstance(telescope.location, EarthLocation)

    def test_azimuth_slew_rate(self):
        """Test that azimuth slew rate is 17.6 deg/min."""
        telescope = build_gbt_telescope()
        expected = 17.6 * (u.deg / u.min)
        self.assertAlmostEqual(
            telescope.az_slew.to_value(u.deg / u.min),
            expected.to_value(u.deg / u.min),
            places=5,
        )

    def test_tip_slew_rate(self):
        """Test that tip (elevation) slew rate is 17.6 deg/min."""
        telescope = build_gbt_telescope()
        expected = 17.6 * (u.deg / u.min)
        self.assertAlmostEqual(
            telescope.tip_slew.to_value(u.deg / u.min),
            expected.to_value(u.deg / u.min),
            places=5,
        )

    def test_azimuth_acceleration(self):
        """Test that azimuth acceleration is 0.2 deg/s²."""
        telescope = build_gbt_telescope()
        expected = 0.2 * (u.deg / u.s**2)
        self.assertAlmostEqual(
            telescope.az_accel.to_value(u.deg / u.s**2),
            expected.to_value(u.deg / u.s**2),
            places=5,
        )

    def test_tip_acceleration(self):
        """Test that tip acceleration is 0.2 deg/s²."""
        telescope = build_gbt_telescope()
        expected = 0.2 * (u.deg / u.s**2)
        self.assertAlmostEqual(
            telescope.tip_accel.to_value(u.deg / u.s**2),
            expected.to_value(u.deg / u.s**2),
            places=5,
        )

    def test_altitude_limits(self):
        """Test that altitude limits are 5° to 90°."""
        telescope = build_gbt_telescope()
        self.assertAlmostEqual(telescope.min_alt.to_value(u.deg), 5.0, places=5)
        self.assertAlmostEqual(telescope.max_alt.to_value(u.deg), 90.0, places=5)

    def test_azimuth_limits(self):
        """Test that azimuth limits are -5° to 365°."""
        telescope = build_gbt_telescope()
        self.assertAlmostEqual(telescope.min_az.to_value(u.deg), -5.0, places=5)
        self.assertAlmostEqual(telescope.max_az.to_value(u.deg), 365.0, places=5)

    def test_angular_resolution(self):
        """Test that angular resolution is 1 deg²."""
        telescope = build_gbt_telescope()
        expected = 1 * (u.deg**2)
        self.assertAlmostEqual(
            telescope.angular_res.to_value(u.deg**2),
            expected.to_value(u.deg**2),
            places=5,
        )


class TestBuildRegularScans(unittest.TestCase):
    """Test cases for build_regular_scans() function."""

    def setUp(self):
        """Set up a telescope instance for testing."""
        self.telescope = build_gbt_telescope()

    def test_returns_list(self):
        """Test that the function returns a list."""
        scans = build_regular_scans(self.telescope)
        self.assertIsInstance(scans, list)

    def test_all_scans_are_backandforthscan(self):
        """Test that all returned scans are BackAndForthScan instances."""
        scans = build_regular_scans(self.telescope)
        self.assertTrue(all(isinstance(scan, BackAndForthScan) for scan in scans))

    def test_correct_number_of_scans(self):
        """Test that the function returns 8 scans (4 pairs of az/tip scans)."""
        scans = build_regular_scans(self.telescope)
        self.assertEqual(len(scans), 8)

    def test_alternating_scan_modes(self):
        """Test that scans alternate between azimuth and tip modes."""
        scans = build_regular_scans(self.telescope)
        # Even indices should be azimuth scans, odd indices should be tip scans
        for i in range(0, len(scans), 2):
            self.assertEqual(scans[i].scan_mode, "az", f"Scan {i} should be azimuth")
        for i in range(1, len(scans), 2):
            self.assertEqual(scans[i].scan_mode, "tip", f"Scan {i} should be tip")

    def test_azimuth_scan_parameters(self):
        """Test that azimuth scans have correct parameters."""
        scans = build_regular_scans(self.telescope)
        az_scan = scans[0]  # First scan should be azimuth

        self.assertEqual(az_scan.scan_mode, "az")
        self.assertAlmostEqual(az_scan.max_angle.to_value(u.deg), 180.0, places=5)
        self.assertAlmostEqual(
            az_scan.min_angle.to_value(u.deg),
            self.telescope.min_az.to_value(u.deg),
            places=5,
        )
        self.assertAlmostEqual(
            az_scan.fixed_deg.to_value(u.deg), 38.0, places=5
        )

    def test_tip_scan_parameters(self):
        """Test that tip scans have correct parameters."""
        scans = build_regular_scans(self.telescope)
        tip_scan = scans[1]  # Second scan should be tip

        self.assertEqual(tip_scan.scan_mode, "tip")
        self.assertAlmostEqual(tip_scan.max_angle.to_value(u.deg), 38.0, places=5)
        self.assertAlmostEqual(tip_scan.min_angle.to_value(u.deg), 5.0, places=5)
        self.assertAlmostEqual(
            tip_scan.fixed_deg.to_value(u.deg), 180.0, places=5
        )


class TestBuildSessionsForRange(unittest.TestCase):
    """Test cases for build_sessions_for_range() function."""

    def setUp(self):
        """Set up a telescope instance for testing."""
        self.telescope = build_gbt_telescope()

    def test_returns_list(self):
        """Test that the function returns a list."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 13, 23, 59, 59, tzinfo=EST_TZ)
        sessions = build_sessions_for_range(self.telescope, start, end)
        self.assertIsInstance(sessions, list)

    def test_all_sessions_are_scansession(self):
        """Test that all returned items are ScanSession instances."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 13, 23, 59, 59, tzinfo=EST_TZ)
        sessions = build_sessions_for_range(self.telescope, start, end)
        self.assertTrue(
            all(isinstance(session, ScanSession) for session in sessions)
        )

    def test_single_day_creates_one_session(self):
        """Test that a single day range creates exactly one session."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 13, 23, 59, 59, tzinfo=EST_TZ)
        sessions = build_sessions_for_range(self.telescope, start, end)
        self.assertEqual(len(sessions), 1)

    def test_week_range_creates_seven_sessions(self):
        """Test that a week range (Jan 13-19) creates 7 sessions."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 19, 23, 59, 59, tzinfo=EST_TZ)
        sessions = build_sessions_for_range(self.telescope, start, end)
        self.assertEqual(len(sessions), 7)

    def test_sessions_start_at_20_00(self):
        """Test that all sessions start at 20:00 local time."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 15, 23, 59, 59, tzinfo=EST_TZ)
        sessions = build_sessions_for_range(self.telescope, start, end)

        for session in sessions:
            self.assertEqual(session.start_time.hour, 20)
            self.assertEqual(session.start_time.minute, 0)
            self.assertEqual(session.start_time.second, 0)

    def test_sessions_use_correct_timezone(self):
        """Test that all sessions use Eastern Time Zone."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 15, 23, 59, 59, tzinfo=EST_TZ)
        sessions = build_sessions_for_range(self.telescope, start, end)

        for session in sessions:
            self.assertEqual(session.start_time.tzinfo, EST_TZ)

    def test_sessions_are_sequential_days(self):
        """Test that sessions are created for consecutive days."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 15, 23, 59, 59, tzinfo=EST_TZ)
        sessions = build_sessions_for_range(self.telescope, start, end)

        expected_dates = [13, 14, 15]
        for i, session in enumerate(sessions):
            self.assertEqual(session.start_time.day, expected_dates[i])

    def test_all_sessions_use_same_telescope(self):
        """Test that all sessions use the same telescope instance."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 15, 23, 59, 59, tzinfo=EST_TZ)
        sessions = build_sessions_for_range(self.telescope, start, end)

        for session in sessions:
            self.assertIs(session.telescope, self.telescope)

    def test_sessions_have_regular_scans(self):
        """Test that all sessions use the regular scan pattern."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 13, 23, 59, 59, tzinfo=EST_TZ)
        sessions = build_sessions_for_range(self.telescope, start, end)

        regular_scans = build_regular_scans(self.telescope)
        session = sessions[0]

        # Check that the session has the same number of scans
        if isinstance(session.scans, list):
            self.assertEqual(len(session.scans), len(regular_scans))
        else:
            # If it's a single scan, it should still match the pattern
            self.assertIsInstance(session.scans, BackAndForthScan)


class TestParseArgs(unittest.TestCase):
    """Test cases for parse_args() function."""

    def test_default_values(self):
        """Test that default values are used when no arguments provided."""
        with patch("sys.argv", ["script_name"]):
            args = parse_args()
            self.assertEqual(args.frame, "galactic")
            self.assertEqual(args.threshold, 10)

    def test_frame_argument_galactic(self):
        """Test that --frame galactic is parsed correctly."""
        with patch("sys.argv", ["script_name", "--frame", "galactic"]):
            args = parse_args()
            self.assertEqual(args.frame, "galactic")

    def test_frame_argument_ra_dec(self):
        """Test that --frame ra_dec is parsed correctly."""
        with patch("sys.argv", ["script_name", "--frame", "ra_dec"]):
            args = parse_args()
            self.assertEqual(args.frame, "ra_dec")

    def test_threshold_argument(self):
        """Test that --threshold argument is parsed correctly."""
        with patch("sys.argv", ["script_name", "--threshold", "15"]):
            args = parse_args()
            self.assertEqual(args.threshold, 15)

    def test_both_arguments(self):
        """Test that both arguments can be provided together."""
        with patch(
            "sys.argv", ["script_name", "--frame", "ra_dec", "--threshold", "20"]
        ):
            args = parse_args()
            self.assertEqual(args.frame, "ra_dec")
            self.assertEqual(args.threshold, 20)

    def test_invalid_frame_raises_error(self):
        """Test that an invalid frame choice raises an error."""
        with patch("sys.argv", ["script_name", "--frame", "invalid"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_invalid_threshold_type_raises_error(self):
        """Test that a non-integer threshold raises an error."""
        with patch("sys.argv", ["script_name", "--threshold", "not_a_number"]):
            with self.assertRaises(SystemExit):
                parse_args()


class TestIntegration(unittest.TestCase):
    """Integration tests for the script components working together."""

    def setUp(self):
        """Set up test fixtures."""
        self.telescope = build_gbt_telescope()

    def test_full_pipeline_jan13_19(self):
        """Test the full pipeline for Jan 13-19 date range."""
        start = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
        end = datetime(2026, 1, 19, 23, 59, 59, tzinfo=EST_TZ)

        sessions = build_sessions_for_range(self.telescope, start, end)
        self.assertEqual(len(sessions), 7)

        # Verify we can create a ScanProject with these sessions
        project = ScanProject(sessions, "Test Project")
        self.assertEqual(len(project.sessions), 7)
        self.assertEqual(project.project_name, "Test Project")

    def test_telescope_and_scans_compatibility(self):
        """Test that the telescope and scans are compatible."""
        telescope = build_gbt_telescope()
        scans = build_regular_scans(telescope)

        # All scans should validate for the telescope
        for scan in scans:
            # This should not raise an exception
            scan.validate_for_tel(telescope)

    def test_sessions_can_be_created_with_scans(self):
        """Test that sessions can be created with the regular scan pattern."""
        telescope = build_gbt_telescope()
        scans = build_regular_scans(telescope)
        start_time = datetime(2026, 1, 13, 20, 0, 0, tzinfo=EST_TZ)

        session = ScanSession(telescope, start_time, scans)
        self.assertIsInstance(session, ScanSession)
        self.assertEqual(session.telescope, telescope)
        self.assertEqual(session.start_time, start_time)


if __name__ == "__main__":
    # Use pytest when run directly for better output formatting
    try:
        import pytest
        import sys
        # Run pytest with the current file and verbose output
        # This gives much better output than unittest.main()
        sys.exit(pytest.main([__file__, "-v", "--tb=short", "--color=yes"]))
    except ImportError:
        # Fallback to unittest if pytest is not available
        import warnings
        warnings.warn("pytest not available, using unittest. Install pytest for better output.")
        unittest.main()

