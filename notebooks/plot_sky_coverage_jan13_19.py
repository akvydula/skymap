"""
Plot GBT sky coverage for a sequence of nights (Jan 13–19) using AltAzSim.

This script recreates the scan configuration from `GBT_Project_Observation.ipynb`
and bundles one `ScanSession` per night into a `ScanProject`, then displays the
combined sky coverage map.

Usage (from the project root):
    python -m scripts.plot_sky_coverage_jan13_19

Optional flags:
    --frame {galactic,ra_dec}   Coordinate frame for the maps (default: galactic)
    --threshold N               Cap number of passes shown per pixel (default: 10)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import astropy.units as u
from astropy.coordinates import EarthLocation

from AltAzSim import Telescope, ScanSession, ScanProject
from AltAzSim.scantypes import BackAndForthScan


# The GBT is located in the Eastern Time Zone
EST_TZ = ZoneInfo("America/New_York")


def build_gbt_telescope() -> Telescope:
    """Construct the Green Bank Telescope instance with the same parameters as the notebook."""
    return Telescope(
        location=EarthLocation.of_site("Green Bank Telescope"),
        az_slew=17.6 * (u.deg / u.min),
        tip_slew=17.6 * (u.deg / u.min),
        az_accel=0.2 * (u.deg / u.s**2),
        tip_accel=0.2 * (u.deg / u.s**2),
        max_alt=90 * u.deg,
        min_alt=5 * u.deg,
        max_az=365 * u.deg,
        min_az=-5 * u.deg,
        angular_res=1 * (u.deg**2),
    )


def build_regular_scans(gbt: Telescope) -> list[BackAndForthScan]:
    """
    Create the sequence of scans used in the notebook:
      - half-azimuth back-and-forth scan for 2 hours at 38 deg elevation
      - one tip scan from 38 to 5 deg
    repeated several times.
    """
    half_az_2hr = BackAndForthScan.from_duration(
        scan_mode="az",
        max_angle=180 * u.deg,
        min_angle=gbt.min_az,
        slew=gbt.az_slew,
        fixed_deg=38 * u.deg,
        duration=1 * u.hr,
    )

    tip_scan_1x = BackAndForthScan.from_count(
        "tip",
        38 * u.deg,
        5 * u.deg,
        gbt.tip_slew,
        180 * u.deg,
        count=1,
    )

    # Repeat the two-scan pattern as in the notebook
    regular_scans: list[BackAndForthScan] = [
        half_az_2hr,
        tip_scan_1x,
        half_az_2hr,
        tip_scan_1x,
        half_az_2hr,
        tip_scan_1x,
        half_az_2hr,
        tip_scan_1x,
    ]
    return regular_scans


def build_sessions_for_range(
    telescope: Telescope,
    start_date: datetime,
    end_date: datetime,
) -> list[ScanSession]:
    """
    Build one ScanSession per calendar day in [start_date, end_date], inclusive.

    Each session starts at 20:00 local time and uses the same regular scan
    pattern as the notebook.
    """
    # Ensure we only work with date part for stepping
    cur = datetime(start_date.year, start_date.month, start_date.day, 20, 0, 0, tzinfo=EST_TZ)
    end = datetime(end_date.year, end_date.month, end_date.day, 20, 0, 0, tzinfo=EST_TZ)

    regular_scans = build_regular_scans(telescope)
    sessions: list[ScanSession] = []

    while cur <= end:
        sessions.append(ScanSession(telescope, cur, regular_scans))
        cur = cur + timedelta(days=1)

    return sessions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GBT sky coverage from Jan 13th to Jan 19th using AltAzSim.",
    )
    parser.add_argument(
        "--frame",
        choices=["galactic", "ra_dec"],
        default="galactic",
        help="Coordinate frame for the coverage maps (default: galactic).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Maximum number of passes to display per pixel (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gbt = build_gbt_telescope()

    # By default, use Jan 13–19, 2026 to be consistent with the example notebook year.
    start_dt = datetime(2026, 1, 13, 0, 0, 0, tzinfo=EST_TZ)
    end_dt = datetime(2026, 1, 19, 23, 59, 59, tzinfo=EST_TZ)

    sessions = build_sessions_for_range(gbt, start_dt, end_dt)

    project_name = "GBT Jan 13–19 Sky Coverage"
    project = ScanProject(sessions, project_name)

    print(f"Running scans for {len(sessions)} nights from Jan 13 to Jan 19...")
    project.run_scans(args.frame)

    print("Displaying combined sky coverage map...")
    project.display_maps(threshold=args.threshold, overlays=None)


if __name__ == "__main__":
    main()

