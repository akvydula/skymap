"""Tests for the dashboard's windowed loaders (checked against skymap.io.read_obs_hdf5)."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from skymap.dashboard import loaders as L
from skymap import Calibrator, io
from skymap.io import CAL_POL_NAMES, CalibratedSpec, HDF5Data, SpecData
from tests.dashboard_helpers import N_FREQ, make_cal_data, make_raw_file


@pytest.fixture()
def raw_file(tmp_path):
    return make_raw_file(tmp_path / "GBTOBS_test.hdf5", n_time=200)


def test_peek_reads_only_time_axis(raw_file):
    pk = L.peek_obs_hdf5(raw_file)
    assert (pk.n_time, pk.n_pol, pk.n_freq) == (200, 10, N_FREQ)
    assert pk.has_spec and not pk.has_calibrated
    assert pk.start_utc == datetime(2026, 1, 16, 16, 27, 49, 726000)
    assert pk.cadence_s == pytest.approx(0.1, rel=1e-4)
    assert pk.end_mjd > pk.start_mjd


def test_peek_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        L.peek_obs_hdf5(tmp_path / "nope.hdf5")


@pytest.mark.parametrize("tz", ["UTC", "America/New_York", "EST"])
def test_slice_from_wall_times_matches_io(raw_file, tz):
    pk = L.peek_obs_hdf5(raw_file)
    ref = io.read_obs_hdf5(raw_file)
    start_utc = pk.start_utc + timedelta(seconds=2)
    end_utc = pk.start_utc + timedelta(seconds=7.5)
    offset = {"UTC": 0, "America/New_York": -5, "EST": -5}[tz]  # January: EST == ET
    start = start_utc + timedelta(hours=offset)
    end = end_utc + timedelta(hours=offset)
    expected = io._slice_from_observation_times(ref, start, end, tz)
    got = L.slice_from_wall_times(pk, start, end, tz)
    assert got == expected
    assert got.stop - got.start == 55 or got.stop - got.start == 56


def test_slice_errors_and_indices(raw_file):
    pk = L.peek_obs_hdf5(raw_file)
    with pytest.raises(ValueError):
        L.slice_from_wall_times(pk, pk.start_utc + timedelta(seconds=5), pk.start_utc, "UTC")
    assert L.slice_from_indices(pk, -5, 1000) == slice(0, 200)
    with pytest.raises(ValueError):
        L.slice_from_indices(pk, 50, 10)
    a, b = L.window_utc_bounds(pk, slice(10, 20))
    assert b - a == timedelta(seconds=0.9)
    with pytest.raises(ValueError):
        L.window_utc_bounds(pk, slice(10, 10))


def test_read_window_all_pols_matches_io(raw_file):
    sl = slice(30, 90)
    ref = io.read_obs_hdf5(raw_file, time_slice=sl)
    got = L.read_obs_window(raw_file, sl)
    assert list(got.time) == list(ref.time)
    np.testing.assert_allclose(got.freq, ref.freq)
    for name in CAL_POL_NAMES:
        np.testing.assert_array_equal(getattr(got.spec, name), getattr(ref.spec, name))


def test_read_window_single_pol_leaves_others_none(raw_file):
    sl = slice(10, 50)
    ref = io.read_obs_hdf5(raw_file, time_slice=sl)
    got = L.read_obs_window(raw_file, sl, ["AB_"])
    np.testing.assert_array_equal(got.spec.AB_, ref.spec.AB_)
    assert all(getattr(got.spec, n) is None for n in CAL_POL_NAMES if n != "AB_")
    assert got.spec.AB_.shape == (40, N_FREQ)
    assert "AB_: (40, 8)" in repr(got.spec)


def test_read_window_validation(raw_file):
    with pytest.raises(ValueError):
        L.read_obs_window(raw_file, slice(0, 10), ["XX_"])
    with pytest.raises(ValueError):
        L.read_obs_window(raw_file, slice(0, 10, 2))
    with pytest.raises(FileNotFoundError):
        L.read_obs_window(raw_file.parent / "missing.hdf5")


def test_calibrate_window_matches_lab_cal(raw_file):
    cal = make_cal_data()
    sl = slice(0, 40)
    got = L.read_obs_window(raw_file, sl, ["AA_", "AB_"])
    L.calibrate_window(got, cal, ["AA_", "AB_"])
    ref = Calibrator.lab_cal(io.read_obs_hdf5(raw_file, time_slice=sl), cal)
    for name in ("AA_", "AB_"):
        np.testing.assert_allclose(getattr(got.calibrated_spec, name).value, getattr(ref.calibrated_spec, name).value)
    assert got.calibrated_spec.BB_ is None
    # idempotent: calibrating again must not raise
    L.calibrate_window(got, cal, ["AA_", "AB_"])


def test_calibrate_window_channel_mismatch(raw_file):
    got = L.read_obs_window(raw_file, slice(0, 10), ["AA_"])
    bad = make_cal_data(n_freq=N_FREQ + 1)
    with pytest.raises(ValueError, match="channels"):
        L.calibrate_window(got, bad, ["AA_"])


def test_mjd_time_axis_and_calibrated_spec_roundtrip(tmp_path):
    """Files written by io.write_obs_hdf5 use time_unit='mjd' and may hold calibrated_spec."""
    n_time = 30
    t0 = datetime(2026, 1, 16, 18, 4, 0)
    times = np.array([t0 + timedelta(seconds=0.1 * i) for i in range(n_time)], dtype=object)
    rng = np.random.default_rng(1)
    spec = SpecData(**{n: rng.normal(size=(n_time, N_FREQ)).astype(np.complex64) for n in CAL_POL_NAMES})
    cal = CalibratedSpec()
    for n in CAL_POL_NAMES:
        setattr(cal, n, rng.uniform(30, 100, size=(n_time, N_FREQ)))
    data = HDF5Data(freq=np.linspace(300, 320, N_FREQ), time=times, spec=spec, calibrated_spec=cal)
    path = tmp_path / "derived.hdf5"
    io.write_obs_hdf5(data, path)

    pk = L.peek_obs_hdf5(path)
    assert pk.time_unit == "mjd" and pk.has_calibrated
    assert abs((pk.start_utc - t0).total_seconds()) < 1e-3

    sl = slice(5, 15)
    ref = io.read_obs_hdf5(path, time_slice=sl)
    got = L.read_obs_window(path, sl, ["AB_"])
    assert max(abs((a - b).total_seconds()) for a, b in zip(got.time, ref.time)) < 1e-3
    np.testing.assert_allclose(got.calibrated_spec.AB_, ref.calibrated_spec.AB_)
    assert got.calibrated_spec.AA_ is None
    # derived files are already calibrated: calibrate_window must leave them untouched
    before = got.calibrated_spec.AB_.copy()
    L.calibrate_window(got, make_cal_data(), ["AB_"])
    np.testing.assert_array_equal(got.calibrated_spec.AB_, before)


def test_memory_estimate_scales_linearly():
    a = L.estimate_window_memory_mb(1000, 1, 256)
    assert L.estimate_window_memory_mb(2000, 1, 256) == pytest.approx(2 * a)
    assert L.estimate_window_memory_mb(1000, 10, 256) == pytest.approx(10 * a)
