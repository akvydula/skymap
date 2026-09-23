"""Tests for az/el ↔ ra/dec helpers and pointing offset scan logic."""

import numpy as np
import pytest
from astropy import units as u
from astropy.time import Time

from skymap import io


def test_gbt_observer_location_matches_gbo_coordinates():
    loc = io.gbt_observer_location()
    np.testing.assert_allclose(loc.lat.to_value(u.deg), 38.433121111111, rtol=0, atol=1e-6)
    np.testing.assert_allclose(loc.lon.to_value(u.deg), -79.839835, rtol=0, atol=1e-6)
    np.testing.assert_allclose(loc.height.to_value(u.m), 807.43, rtol=0, atol=0.01)


def test_radec_az_el_round_trip_at_gbt():
    """Catalog RA/Dec → Az/El → RA/Dec is consistent at GBT."""
    obstime = Time("2026-01-15T12:00:00", scale="utc")
    time_arr = np.array([obstime.mjd])

    src_ra, src_dec = 187.7059, 12.3911  # 3C353 approx
    az, el = io.radec_to_az_el_deg(src_ra, src_dec, time_arr)
    ra_back, dec_back = io.az_el_to_radec_deg(az, el, time_arr)

    np.testing.assert_allclose(ra_back, src_ra, rtol=0, atol=1e-6)
    np.testing.assert_allclose(dec_back, src_dec, rtol=0, atol=1e-6)


def test_local_sidereal_time_at_gbt():
    obstime = Time("2026-01-15T12:00:00", scale="utc")
    lst = io.local_sidereal_time_deg(np.array([obstime.mjd]))
    assert lst.shape == (1,)
    assert 0.0 <= lst[0] < 360.0


def test_split_time_ordered_scans_four_legs():
    legs = io.split_time_ordered_scans(20, n_scans=4)
    assert len(legs) == 4
    assert np.concatenate(legs).tolist() == list(range(20))


def test_split_pointing_scans_by_gaps_four_blocks():
    """Four recording blocks separated by large gaps → four segments."""
    start_mjd = 61000.0
    # 5 samples per block at 0.1 s cadence; gaps of 30/60/40 s between blocks.
    blocks = []
    t0 = 0.0
    for i, gap_before in enumerate((0.0, 30.0, 60.0, 40.0)):
        if i == 0:
            t0 = 0.0
        else:
            t0 = float(blocks[-1][-1]) + gap_before
        blocks.append(t0 + np.arange(5) * 0.1)
    mjd = start_mjd + np.concatenate(blocks) / 86400.0
    legs = io.split_pointing_scans_by_gaps(mjd, gap_threshold_seconds=1.0)
    assert len(legs) == 4
    assert [leg.size for leg in legs] == [5, 5, 5, 5]
    assert legs[0][0] == 0 and legs[0][-1] == 4
    assert legs[3][0] == 15 and legs[3][-1] == 19
    resolved, mode, diag = io.resolve_scan_legs(mjd, n_scans=4)
    assert mode == "gaps"
    assert len(resolved) == 4
    assert diag["gap_seconds"] == pytest.approx([30.0, 60.0, 40.0], abs=1e-6)


def test_split_pointing_scans_by_gaps_continuous_one_segment():
    """Continuous timeline (no gaps) → a single segment; resolve falls back."""
    start_mjd = 61000.0
    mjd = start_mjd + np.arange(20) * 0.1 / 86400.0
    legs = io.split_pointing_scans_by_gaps(mjd, gap_threshold_seconds=1.0)
    assert len(legs) == 1
    assert legs[0].tolist() == list(range(20))
    resolved, mode, diag = io.resolve_scan_legs(mjd, n_scans=4)
    assert mode == "equal"
    assert len(resolved) == 4
    assert diag["gap_seconds"] == []


def test_estimate_timing_offset_from_source_peaks(monkeypatch):
    """El/az scan closest-approach vs spectrum peaks recover a 60 ms lag."""
    start_mjd = 61000.0
    delay = 0.060
    src_az, src_el = 105.0, 45.0

    # Four short legs; each has an exact on-source pointing sample.
    pointing_seconds = np.arange(20, dtype=float) * 0.5  # 0, 0.5, ..., 9.5
    az = np.full(20, src_az)
    el = np.full(20, src_el)
    # leg0 [0:5]: el 40..50, on-source at index 2 (t=1.0)
    el[0:5] = np.array([40.0, 42.5, 45.0, 47.5, 50.0])
    # leg1 [5:10]: el 50..40, on-source at index 7 (t=3.5)
    el[5:10] = np.array([50.0, 47.5, 45.0, 42.5, 40.0])
    # leg2 [10:15]: az 100..110, on-source at index 12 (t=6.0)
    az[10:15] = np.array([100.0, 102.5, 105.0, 107.5, 110.0])
    # leg3 [15:20]: az 110..100, on-source at index 17 (t=8.5)
    az[15:20] = np.array([110.0, 107.5, 105.0, 102.5, 100.0])

    on_source_times = np.array([1.0, 3.5, 6.0, 8.5])
    pointing = io.PointingData(
        dmjd=start_mjd + pointing_seconds / 86400.0,
        az=az,
        el=el,
        ra=np.full(20, 187.7),
        dec=np.full(20, 12.4),
    )

    # Dense spectrum timeline plus exact delayed peak stamps at on-source times.
    spec_stamp_s = np.unique(
        np.concatenate(
            [np.arange(-0.1, 10.0, 0.02), on_source_times - delay]
        )
    )
    brightness = np.zeros(spec_stamp_s.size)
    for t_true in on_source_times:
        i = int(np.where(np.isclose(spec_stamp_s, t_true - delay))[0][0])
        brightness[i] = 1.0

    cal = io.CalibratedSpec()
    cal.AB_ = brightness[:, None] * np.ones((brightness.size, 2))
    data = io.HDF5Data(
        freq=np.array([309.0, 310.0]),
        time=start_mjd + spec_stamp_s / 86400.0,
        spec=None,
        calibrated_spec=cal,
    )

    monkeypatch.setattr("skymap.Beam.get_source_radec", lambda _name: (187.7059, 12.3911))
    monkeypatch.setattr(
        io,
        "expected_source_altaz_deg",
        lambda _ra, _dec, time_arr: (
            np.full(np.asarray(time_arr).shape, src_az),
            np.full(np.asarray(time_arr).shape, src_el),
        ),
    )

    info = io.estimate_timing_offset_from_source_peaks(
        data, pointing, "3C353", n_scans=4, el_scans=(0, 1), az_scans=(2, 3)
    )
    # Continuous pointing → equal-split fallback.
    assert info["scan_split"] == "equal"
    assert info["el_timing_offset_seconds"] == pytest.approx(delay, abs=1e-3)
    assert info["az_timing_offset_seconds"] == pytest.approx(delay, abs=1e-3)
    assert info["timing_offset_seconds"] == pytest.approx(delay, abs=1e-3)

    matched = io.match_data_and_pointing_with_timing_offset(
        data,
        pointing,
        "3C353",
        n_scans=4,
        el_scans=(0, 1),
        az_scans=(2, 3),
        apply_spatial_offset_correction=False,
    )
    assert matched.timing_offset_seconds == pytest.approx(delay, abs=1e-3)
    assert matched.el_timing_offset_seconds == pytest.approx(info["el_timing_offset_seconds"])
    assert matched.az_timing_offset_seconds == pytest.approx(info["az_timing_offset_seconds"])


def test_estimate_timing_offset_with_spatial_correction(monkeypatch):
    """Spatial El offset is removed before lag estimation; 60 ms lag recovered."""
    start_mjd = 61000.0
    delay = 0.060
    src_az, src_el = 105.0, 45.0
    el_spatial = 0.15  # measured - expected at brightness peak

    # Dense el legs so catalog-closest (el≈45) and true-source peak (el≈45.15)
    # land on different samples.
    def _el_leg(up: bool) -> np.ndarray:
        el_vals = np.arange(40.0, 50.0 + 1e-9, 0.05)
        return el_vals if up else el_vals[::-1]

    el0 = _el_leg(True)
    el1 = _el_leg(False)
    n_leg = el0.size
    dt = 0.05
    gap = 30.0
    t0 = np.arange(n_leg) * dt
    t1 = float(t0[-1]) + gap + np.arange(n_leg) * dt
    # Short az legs (no spatial az offset) after gaps.
    az2 = np.linspace(100.0, 110.0, 11)
    az3 = np.linspace(110.0, 100.0, 11)
    t2 = float(t1[-1]) + gap + np.arange(az2.size) * dt
    t3 = float(t2[-1]) + gap + np.arange(az3.size) * dt

    pointing_seconds = np.concatenate([t0, t1, t2, t3])
    az = np.concatenate(
        [
            np.full(n_leg, src_az),
            np.full(n_leg, src_az),
            az2,
            az3,
        ]
    )
    el = np.concatenate(
        [
            el0,
            el1,
            np.full(az2.size, src_el),
            np.full(az3.size, src_el),
        ]
    )

    # Brightness peaks when mount crosses true source El / Az.
    i_el0 = int(np.argmin(np.abs(el0 - (src_el + el_spatial))))
    i_el1 = n_leg + int(np.argmin(np.abs(el1 - (src_el + el_spatial))))
    i_az2 = 2 * n_leg + int(np.argmin(np.abs(az2 - src_az)))
    i_az3 = 2 * n_leg + az2.size + int(np.argmin(np.abs(az3 - src_az)))
    on_source_times = pointing_seconds[[i_el0, i_el1, i_az2, i_az3]]

    pointing = io.PointingData(
        dmjd=start_mjd + pointing_seconds / 86400.0,
        az=az,
        el=el,
        ra=np.full(pointing_seconds.size, 187.7),
        dec=np.full(pointing_seconds.size, 12.4),
    )

    t_end = float(pointing_seconds[-1]) + 1.0
    spec_stamp_s = np.unique(
        np.concatenate([np.arange(-0.1, t_end, 0.01), on_source_times - delay])
    )
    brightness = np.zeros(spec_stamp_s.size)
    for t_true in on_source_times:
        i = int(np.where(np.isclose(spec_stamp_s, t_true - delay))[0][0])
        brightness[i] = 1.0

    cal = io.CalibratedSpec()
    cal.AB_ = brightness[:, None] * np.ones((brightness.size, 2))
    data = io.HDF5Data(
        freq=np.array([309.0, 310.0]),
        time=start_mjd + spec_stamp_s / 86400.0,
        spec=None,
        calibrated_spec=cal,
    )

    monkeypatch.setattr("skymap.Beam.get_source_radec", lambda _name: (187.7059, 12.3911))
    monkeypatch.setattr(
        io,
        "expected_source_altaz_deg",
        lambda _ra, _dec, time_arr: (
            np.full(np.asarray(time_arr).shape, src_az),
            np.full(np.asarray(time_arr).shape, src_el),
        ),
    )

    # Without spatial correction, opposite el legs disagree (spatial masquerades as time).
    info_raw = io.estimate_timing_offset_from_source_peaks(
        data, pointing, "3C353", n_scans=4, el_scans=(0, 1), az_scans=(2, 3)
    )
    assert abs(info_raw["el_timing_offsets"][0] - info_raw["el_timing_offsets"][1]) > 0.2

    info = io.estimate_timing_offset_after_spatial_correction(
        data, pointing, "3C353", n_scans=4, el_scans=(0, 1), az_scans=(2, 3)
    )
    assert info["el_offset_deg"] == pytest.approx(el_spatial, abs=0.03)
    assert info["timing_offset_seconds"] == pytest.approx(delay, abs=0.02)
    assert info["el_timing_offset_seconds"] == pytest.approx(delay, abs=0.02)

    matched = io.match_data_and_pointing_with_timing_offset(
        data, pointing, "3C353", n_scans=4, el_scans=(0, 1), az_scans=(2, 3)
    )
    assert matched.timing_offset_seconds == pytest.approx(delay, abs=0.02)
    assert matched.el_offset_deg == pytest.approx(info["el_offset_deg"], abs=1e-6)


def test_estimate_timing_offset_with_pointing_gaps(monkeypatch):
    """Inter-leg pointing gaps select true legs; 60 ms lag still recovered."""
    start_mjd = 61000.0
    delay = 0.060
    src_az, src_el = 105.0, 45.0

    # Four legs of 5 samples at 0.5 s cadence, separated by 30 s gaps
    # (start of next block = end of previous + 30 s).
    leg_local = np.arange(5, dtype=float) * 0.5  # 0 .. 2.0
    blocks = [leg_local.copy()]
    for _ in range(3):
        blocks.append(float(blocks[-1][-1]) + 30.0 + leg_local)
    pointing_seconds = np.concatenate(blocks)
    az = np.full(20, src_az)
    el = np.full(20, src_el)
    el[0:5] = np.array([40.0, 42.5, 45.0, 47.5, 50.0])
    el[5:10] = np.array([50.0, 47.5, 45.0, 42.5, 40.0])
    az[10:15] = np.array([100.0, 102.5, 105.0, 107.5, 110.0])
    az[15:20] = np.array([110.0, 107.5, 105.0, 102.5, 100.0])

    # On-source times within each block (local index 2).
    on_source_times = pointing_seconds[[2, 7, 12, 17]]
    pointing = io.PointingData(
        dmjd=start_mjd + pointing_seconds / 86400.0,
        az=az,
        el=el,
        ra=np.full(20, 187.7),
        dec=np.full(20, 12.4),
    )

    t_end = float(pointing_seconds[-1]) + 1.0
    spec_stamp_s = np.unique(
        np.concatenate(
            [np.arange(-0.1, t_end, 0.02), on_source_times - delay]
        )
    )
    brightness = np.zeros(spec_stamp_s.size)
    for t_true in on_source_times:
        i = int(np.where(np.isclose(spec_stamp_s, t_true - delay))[0][0])
        brightness[i] = 1.0

    cal = io.CalibratedSpec()
    cal.AB_ = brightness[:, None] * np.ones((brightness.size, 2))
    data = io.HDF5Data(
        freq=np.array([309.0, 310.0]),
        time=start_mjd + spec_stamp_s / 86400.0,
        spec=None,
        calibrated_spec=cal,
    )

    monkeypatch.setattr("skymap.Beam.get_source_radec", lambda _name: (187.7059, 12.3911))
    monkeypatch.setattr(
        io,
        "expected_source_altaz_deg",
        lambda _ra, _dec, time_arr: (
            np.full(np.asarray(time_arr).shape, src_az),
            np.full(np.asarray(time_arr).shape, src_el),
        ),
    )

    info = io.estimate_timing_offset_from_source_peaks(
        data, pointing, "3C353", n_scans=4, el_scans=(0, 1), az_scans=(2, 3)
    )
    assert info["scan_split"] == "gaps"
    assert info["gap_seconds"] == pytest.approx([30.0, 30.0, 30.0], abs=1e-3)
    assert info["el_timing_offset_seconds"] == pytest.approx(delay, abs=1e-3)
    assert info["az_timing_offset_seconds"] == pytest.approx(delay, abs=1e-3)
    assert info["timing_offset_seconds"] == pytest.approx(delay, abs=1e-3)


def test_match_data_and_pointing_with_timing_offset_explicit():
    """Explicit lag is applied without needing a source name."""
    start_mjd = 61000.0
    pointing_seconds = np.arange(0.0, 2.0, 0.1)
    pointing = io.PointingData(
        dmjd=start_mjd + pointing_seconds / 86400.0,
        az=100.0 + 0.1 * pointing_seconds,
        el=40.0 + 0.05 * pointing_seconds,
        ra=np.zeros(pointing_seconds.size),
        dec=np.zeros(pointing_seconds.size),
    )
    data_seconds = np.array([0.5, 1.0, 1.5])
    data = io.HDF5Data(
        freq=np.array([310.0]),
        time=start_mjd + data_seconds / 86400.0,
        spec=None,
    )
    matched = io.match_data_and_pointing_with_timing_offset(
        data, pointing, timing_offset_seconds=0.06
    )
    assert matched.timing_offset_seconds == pytest.approx(0.06)
    np.testing.assert_allclose(matched.time, start_mjd + (data_seconds + 0.06) / 86400.0)


def test_match_data_and_pointing_with_timing_offset_rejects_large_offset():
    pointing = io.PointingData(
        dmjd=np.array([61000.0, 61000.001]),
        az=np.array([100.0, 101.0]),
        el=np.array([40.0, 41.0]),
        ra=np.zeros(2),
        dec=np.zeros(2),
    )
    data = io.HDF5Data(freq=np.array([310.0]), time=np.array([61000.0]), spec=None)
    with pytest.raises(ValueError, match="exceeds max_offset_seconds"):
        io.match_data_and_pointing_with_timing_offset(
            data, pointing, timing_offset_seconds=0.6
        )


def test_match_data_and_pointing_with_timing_offset_requires_source_name():
    pointing = io.PointingData(
        dmjd=np.array([61000.0, 61000.001]),
        az=np.array([100.0, 101.0]),
        el=np.array([40.0, 41.0]),
        ra=np.zeros(2),
        dec=np.zeros(2),
    )
    data = io.HDF5Data(freq=np.array([310.0]), time=np.array([61000.0]), spec=None)

    with pytest.raises(ValueError, match="source_name is required"):
        io.match_data_and_pointing_with_timing_offset(data, pointing)


class _FakeCalSpec:
    def __init__(self, mean: np.ndarray):
        self.AA_ = mean
        self.BB_ = mean


class _FakeMatched:
    def __init__(self, val: np.ndarray, *, time=None):
        n = val.size
        self.ra = np.linspace(187.0, 188.0, n)
        self.dec = np.full(n, 12.4)
        self.az = np.linspace(100.0, 110.0, n)
        self.el = np.linspace(40.0, 50.0, n)
        self.time = np.arange(n, dtype=float) if time is None else time
        self.calibrated_spec_mean = _FakeCalSpec(val[:, None])


def test_radec_corrected_for_pointing_offset_round_trip():
    """Subtracting measured offsets from az/el recovers catalog RA/Dec at peak."""
    obstime = Time("2026-01-15T12:00:00", scale="utc")
    time_arr = np.array([obstime.mjd])
    src_ra, src_dec = 187.7059, 12.3911

    src_az, src_el = io.radec_to_az_el_deg(src_ra, src_dec, time_arr)
    az_offset = 0.05
    el_offset = -0.03
    az_meas = src_az + az_offset
    el_meas = src_el + el_offset

    ra_corr, dec_corr = io.radec_corrected_for_pointing_offset(
        az_meas,
        el_meas,
        time_arr,
        az_offset_deg=float(az_offset),
        el_offset_deg=float(el_offset),
    )
    np.testing.assert_allclose(ra_corr, src_ra, rtol=0, atol=1e-5)
    np.testing.assert_allclose(dec_corr, src_dec, rtol=0, atol=1e-5)


def test_get_pointing_offset_per_leg_peaks_and_means(monkeypatch):
    """One peak per el/az leg; returned offsets are means of per-leg values."""
    n = 40
    val = np.zeros(n)
    legs = io.split_time_ordered_scans(n, n_scans=4)
    i_el_0 = int(legs[0][3])
    i_el_1 = int(legs[1][3])
    i_az_2 = int(legs[2][3])
    i_az_3 = int(legs[3][3])
    val[i_el_0] = 1.0
    val[i_el_1] = 1.2
    val[i_az_2] = 2.0
    val[i_az_3] = 2.2

    # Continuous integer times → equal-split fallback (same legs as above).
    data = _FakeMatched(val)
    monkeypatch.setattr("skymap.Beam.get_source_radec", lambda _name: (187.7059, 12.3911))
    monkeypatch.setattr(
        io,
        "expected_source_altaz_deg",
        lambda _ra, _dec, time_arr: (
            np.full(time_arr.shape, 105.0),
            np.full(time_arr.shape, 45.0),
        ),
    )

    out = io.get_pointing_offset(data, "3C353", el_scans=(0, 1), az_scans=(2, 3))

    expected_el = [
        float(data.el[i_el_0] - 45.0),
        float(data.el[i_el_1] - 45.0),
    ]
    expected_az = [
        float(data.az[i_az_2] - 105.0),
        float(data.az[i_az_3] - 105.0),
    ]
    assert out["scan_split"] == "equal"
    assert out["el_peak_indices"] == [i_el_0, i_el_1]
    assert out["az_peak_indices"] == [i_az_2, i_az_3]
    assert out["el_offsets"] == pytest.approx(expected_el)
    assert out["az_offsets"] == pytest.approx(expected_az)
    assert out["el_offset"] == pytest.approx(np.mean(np.abs(expected_el)))
    assert out["az_offset"] == pytest.approx(np.mean(np.abs(expected_az)))
    expected_dec = [
        float(data.dec[i_el_0] - 12.3911),
        float(data.dec[i_el_1] - 12.3911),
    ]
    expected_ra = [
        io._ra_offset_deg(data.ra[i_az_2], 187.7059),
        io._ra_offset_deg(data.ra[i_az_3], 187.7059),
    ]
    assert out["dec_offsets"] == pytest.approx(expected_dec)
    assert out["ra_offsets"] == pytest.approx(expected_ra)
    assert out["dec_offset"] == pytest.approx(np.mean(np.abs(expected_dec)))
    assert out["ra_offset"] == pytest.approx(np.mean(np.abs(expected_ra)))
    assert out["el_scans"] == (0, 1)
    assert out["az_scans"] == (2, 3)
