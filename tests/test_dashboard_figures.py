"""Smoke tests for dashboard figure capture and analysis helpers (Agg backend, synthetic data)."""

from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from skymap.dashboard import analysis, figures, loaders
from skymap import healmap, io, plots
from skymap.io import CAL_POL_NAMES, CalibratedSpec, HDF5Data
from tests.dashboard_helpers import N_FREQ, make_cal_data, make_raw_file


@pytest.fixture()
def calibrated_obs(tmp_path):
    path = make_raw_file(tmp_path / "GBTOBS_fig.hdf5", n_time=120)
    obs = loaders.read_obs_window(path, slice(0, 120), ["AA_", "AB_"])
    loaders.calibrate_window(obs, make_cal_data(), ["AA_", "AB_"])
    return obs


def test_capture_neutralises_show_and_close_and_leaves_no_figures(calibrated_obs):
    real_show, real_close = plt.show, plt.close
    fig = figures.capture(
        plots.plot_waterfall, calibrated_obs, calibrated=True, attribute="AB_", time_zone="UTC"
    )
    assert isinstance(fig, Figure)
    assert plt.show is real_show and plt.close is real_close
    assert plt.get_fignums() == []
    fig.savefig(__import__("io").BytesIO(), format="png")  # still renderable after detaching


def test_capture_raises_when_nothing_drawn():
    with pytest.raises(RuntimeError):
        figures.capture(lambda: None)


def test_capture_restores_pyplot_on_error():
    real_show = plt.show

    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        figures.capture(boom)
    assert plt.show is real_show


def test_decimate_obs_block_average(calibrated_obs):
    dec = figures.decimate_obs(calibrated_obs, max_rows=30, magnitude=True)
    assert len(dec.time) == 30
    assert dec.calibrated_spec.AB_.shape == (30, N_FREQ)
    assert getattr(dec.calibrated_spec.AB_, "unit", None) is None  # plain Kelvin floats
    raw = np.abs(calibrated_obs.spec.AB_[:4]).mean(axis=0)
    np.testing.assert_allclose(dec.spec.AB_[0], raw, rtol=1e-5)
    assert dec.spec.BB_ is None


def test_waterfall_cal_and_amplitude_figures(calibrated_obs):
    f1 = figures.waterfall_figure(calibrated_obs, ["AB_"], calibrated=True, tz="UTC", max_rows=50)
    f2 = figures.waterfall_figure(calibrated_obs, ["AA_", "AB_"], calibrated=False, tz="America/New_York")
    f3 = figures.amplitude_time_figure(calibrated_obs, ["AA_", "AB_"], calibrated=True, tz="EST")
    f4 = figures.raw_vs_calibrated_figure(calibrated_obs, "AB_", "UTC", max_rows=40)
    f5 = figures.cal_figure(make_cal_data(), ["AA_", "AB_"])
    assert all(isinstance(f, Figure) for f in (f1, f2, f3, f4, f5))
    assert len(f1.axes) == 2  # image + colourbar
    assert plt.get_fignums() == []


def _matched(n=60, pols=("AB_", "AA_"), nan_rows=()):
    ra = 260.0 + 0.2 * np.sin(np.linspace(0, 6, n))
    dec = -1.0 + 0.2 * np.cos(np.linspace(0, 6, n))
    cs = CalibratedSpec()
    rng = np.random.default_rng(0)
    for name in CAL_POL_NAMES:
        arr = rng.uniform(40, 60, size=(n, N_FREQ)) if name in pols else None
        if arr is not None:
            arr[list(nan_rows), :] = np.nan
        setattr(cs, name, arr)
    return HDF5Data(
        freq=np.linspace(300, 320, N_FREQ),
        time=np.array([datetime(2026, 1, 16) + timedelta(seconds=i) for i in range(n)], dtype=object),
        ra=ra, dec=dec, az=ra, el=dec, calibrated_spec_mean=cs,
    )


def test_analysis_amplitude_and_restrict():
    matched = _matched()
    both = analysis.amplitude_series(matched, ["AB_", "AA_"])
    one = analysis.amplitude_series(matched, ["AB_"])
    assert both.shape == one.shape == (60,)
    np.testing.assert_allclose(one, matched.calibrated_spec_mean.AB_.mean(axis=1))
    with pytest.raises(ValueError):
        analysis.amplitude_series(matched, ["DD_"])
    analysis.restrict_matched(matched, ["AB_"])
    assert matched.calibrated_spec_mean.AA_ is None and matched.calibrated_spec_mean.AB_ is not None


def test_scan_leg_args_validation():
    assert analysis.scan_leg_args(4, 2) == {"n_scans": 4, "el_scans": (0, 1), "az_scans": (2, 3)}
    assert analysis.scan_leg_args(2, 1)["az_scans"] == (1,)
    with pytest.raises(ValueError):
        analysis.scan_leg_args(2, 2)
    assert analysis.timing_attribute(["AB_"]) == "AB_"
    assert analysis.timing_attribute(list(CAL_POL_NAMES)) is None


def test_healpix_skips_nan_pointings_and_plots():
    matched = _matched(nan_rows=(3, 4))
    hmap = analysis.build_healpix(matched, ["AB_"], nside=256)
    m = hmap.get_map("AB_")
    assert np.all(np.isfinite(m))
    assert np.count_nonzero(hmap.get_hit_count() > 0) > 0
    f_patch = figures.healpix_patch_figure(hmap, ["AB_"], center_ra=260.0, center_dec=-1.0, patch_deg=4.0)
    f_full = figures.healpix_full_figure(hmap, ["AB_"])
    f_gnom = figures.healpix_gnomview_figure(hmap, "AB_", center_ra=260.0, center_dec=-1.0, patch_deg=4.0)
    assert all(isinstance(f, Figure) for f in (f_patch, f_full, f_gnom))
    assert plt.get_fignums() == []
    with pytest.raises(ValueError):
        analysis.build_healpix(matched, ["DD_"], nside=64)


def test_beam_fit_on_synthetic_gaussian():
    rng = np.random.default_rng(3)
    r = np.abs(rng.uniform(0, 3, 600))
    sigma = 0.3
    y = 35 + 100 * np.exp(-(r**2) / (2 * sigma**2)) + rng.normal(0, 0.5, r.size)
    params, warns = analysis.fit_beam(r, y, baseline_outer_deg=1.0)
    assert params["sigma_deg"] == pytest.approx(sigma, rel=0.05)
    assert params["baseline_k"] == pytest.approx(35, abs=1.0)
    fig, p2 = figures.beam_fit_figure(r, y, baseline_outer_deg=1.0)
    assert isinstance(fig, Figure) and p2["FWHM_deg"] == pytest.approx(params["FWHM_deg"])
    fig2 = figures.beam_approximation_figure(p2, x_max_deg=3.0, normalize=False)
    assert isinstance(fig2, Figure)
