"""Tests for |AB_| mean gridding and spatial Gaussian contribution subtraction."""

from types import SimpleNamespace

import numpy as np
import pytest

from skymap import Beam


def _spec_container(**channels):
    ns = SimpleNamespace()
    for name, arr in channels.items():
        setattr(ns, name, arr)
    return ns


def _pointing(ra, dec, ab):
    return SimpleNamespace(
        ra=np.asarray(ra, dtype=float),
        dec=np.asarray(dec, dtype=float),
        calibrated_spec_mean=_spec_container(AB_=np.asarray(ab)),
    )


def test_pol_channel_magnitude_mean_uses_abs_then_freq_mean():
    ab = np.array(
        [
            [3 + 4j, 0 + 0j],
            [0 + 8j, 6 + 8j],
        ],
        dtype=np.complex128,
    )
    spec = _spec_container(AB_=ab)
    out = Beam.pol_channel_magnitude_mean(spec, "AB_")
    np.testing.assert_allclose(out, [2.5, 9.0])
    one_chan = Beam.pol_channel_magnitude_mean(spec, "AB_", freq_index=0)
    np.testing.assert_allclose(one_chan, [5.0, 8.0])


def test_pol_channel_default_mean_uses_real_part():
    ab = np.array([[3 + 4j, 1 + 0j]], dtype=np.complex128)
    spec = _spec_container(AB_=ab)
    out = Beam._pol_channel_values_1d(spec, "AB_", None, reduce="mean")
    np.testing.assert_allclose(out, [2.0])


def test_dense_raster_fills_observed_bbox():
    """Kernel-weighted fill of a dense raster; grid stays on the pointing bbox."""
    n = 21
    ra = np.linspace(180.0, 182.0, n)
    dec = np.linspace(10.0, 12.0, n)
    RA, DEC = np.meshgrid(ra, dec)
    ab = np.ones(RA.size, dtype=float)
    data = _pointing(RA.ravel(), DEC.ravel(), ab)
    params = {"FWHM_deg": 0.5, "sigma_deg": 0.21, "baseline": 0.0}
    grid = Beam.grid_patch_fixed_shape(data, params, n_pix=32, reduce="mean")
    z = grid["maps"]["AB_"]
    w = grid["weight_sum"]["AB_"]
    assert z.shape == (32, 32)
    np.testing.assert_allclose(grid["ra_edges"][0], 180.0)
    np.testing.assert_allclose(grid["ra_edges"][-1], 182.0)
    np.testing.assert_allclose(grid["dec_edges"][0], 10.0)
    np.testing.assert_allclose(grid["dec_edges"][-1], 12.0)
    frac = float(np.mean(np.isfinite(z) & (w > 0)))
    assert frac > 0.8


def test_linear_field_recovered_on_fine_grid():
    """A linear sky is recovered without a spreading kernel."""
    n = 11
    ra = np.linspace(180.0, 181.0, n)
    dec = np.linspace(10.0, 11.0, n)
    RA, DEC = np.meshgrid(ra, dec)
    ab = RA.ravel() + 2.0 * DEC.ravel()
    data = _pointing(RA.ravel(), DEC.ravel(), ab)
    params = {"FWHM_deg": 0.5, "sigma_deg": 0.21, "baseline": 0.0}
    grid = Beam.grid_patch_fixed_shape(
        data, params, n_pix=48, reduce="mean", method="bin_linear"
    )
    z = grid["maps"]["AB_"]
    RA_g, DEC_g = np.meshgrid(grid["ra_centers"], grid["dec_centers"])
    expected = RA_g + 2.0 * DEC_g
    valid = np.isfinite(z)
    assert float(np.mean(valid)) > 0.8
    interior = valid.copy()
    interior[:4, :] = False
    interior[-4:, :] = False
    interior[:, :4] = False
    interior[:, -4:] = False
    np.testing.assert_allclose(z[interior], expected[interior], atol=0.04)


def test_interpolation_does_not_fill_padded_exterior():
    """Samples in a small patch must not paint padded sky far from any pointing."""
    n = 9
    ra = np.linspace(180.0, 180.2, n)
    dec = np.linspace(10.0, 10.2, n)
    RA, DEC = np.meshgrid(ra, dec)
    data = _pointing(RA.ravel(), DEC.ravel(), np.ones(RA.size))
    params = {"FWHM_deg": 0.5, "sigma_deg": 0.21, "baseline": 0.0}
    grid = Beam.grid_patch_fixed_shape(
        data, params, n_pix=64, padding_pixels=16, reduce="mean"
    )
    z = grid["maps"]["AB_"]
    assert np.isfinite(z[32, 32])
    assert not np.isfinite(z[0, 0])
    assert not np.isfinite(z[0, -1])
    assert not np.isfinite(z[-1, 0])
    assert not np.isfinite(z[-1, -1])


def test_scan_rows_fill_between_lines():
    """Cross-track kernel must fill pixels between RA scans."""
    ra = np.linspace(180.0, 181.0, 40)
    decs = np.array([10.0, 10.12, 10.24])
    ra_all = np.concatenate([ra, ra, ra])
    dec_all = np.concatenate([np.full_like(ra, d) for d in decs])
    data = _pointing(ra_all, dec_all, np.ones(ra_all.size))
    params = {"FWHM_deg": 0.5, "sigma_deg": 0.21, "baseline": 0.0}
    grid = Beam.grid_patch_fixed_shape(data, params, n_pix=64, reduce="mean")
    z = grid["maps"]["AB_"]
    dec_c = grid["dec_centers"]
    i_mid = int(np.argmin(np.abs(dec_c - 10.06)))
    assert np.isfinite(z[i_mid]).mean() > 0.8


def test_gaussian_peak_survives_scan_line_fill():
    """A compact Gaussian sampled on scan rows stays peaked at the center."""
    ra = np.linspace(-1.0, 1.0, 41)
    decs = np.linspace(-1.0, 1.0, 9)
    ra_all = []
    dec_all = []
    for d in decs:
        ra_all.append(ra)
        dec_all.append(np.full_like(ra, d))
    ra_all = np.concatenate(ra_all)
    dec_all = np.concatenate(dec_all)
    sigma = 0.25
    ab = np.exp(-0.5 * (ra_all**2 + dec_all**2) / sigma**2)
    data = _pointing(180.0 + ra_all, 10.0 + dec_all, ab)
    params = {"FWHM_deg": 0.5, "sigma_deg": 0.21, "baseline": 0.0}
    grid = Beam.grid_patch_fixed_shape(
        data, params, n_pix=96, reduce="mean", method="bin_linear"
    )
    z = grid["maps"]["AB_"]
    assert np.nanmax(z) > 0.90
    i, j = np.unravel_index(int(np.nanargmax(z)), z.shape)
    ra_pk = grid["ra_centers"][j]
    dec_pk = grid["dec_centers"][i]
    np.testing.assert_allclose(ra_pk, 180.0, atol=0.15)
    np.testing.assert_allclose(dec_pk, 10.0, atol=0.15)
    row = z[i, :]
    col = z[:, j]
    finite_row = row[np.isfinite(row)]
    finite_col = col[np.isfinite(col)]
    assert finite_row.size > 10
    assert finite_col.size > 5
    assert float(np.nanmax(row)) == pytest.approx(float(np.nanmax(z)))
    core = np.abs(grid["ra_centers"] - 180.0) < 0.08
    assert np.all(np.isfinite(row[core]))
    assert float(np.min(row[core])) > 0.80


def test_map_source_axis_cuts_through_peak():
    z = np.full((5, 7), np.nan)
    z[2, :] = np.arange(7, dtype=float)
    z[:, 3] = 10.0 + np.arange(5, dtype=float)
    z[2, 3] = 50.0
    ra = np.linspace(10.0, 16.0, 7)
    dec = np.linspace(-2.0, 2.0, 5)
    cuts = Beam.map_source_axis_cuts(z, ra, dec)
    np.testing.assert_allclose(cuts["source_ra_deg"], ra[3])
    np.testing.assert_allclose(cuts["source_dec_deg"], dec[2])
    np.testing.assert_allclose(cuts["amp_vs_ra"][3], 50.0)
    np.testing.assert_allclose(cuts["amp_vs_dec"][2], 50.0)


def test_peak_normalized_psf_has_unit_center():
    dec = np.linspace(-1.0, 1.0, 64)
    psf = Beam._gaussian_psf_2d_on_grid(
        (64, 64), 0.05, dec, 0.2, normalize="peak"
    )
    cy, cx = 32, 32
    np.testing.assert_allclose(psf[cy, cx], 1.0, atol=1e-12)
    assert float(np.max(psf)) == pytest.approx(1.0)


def test_isolated_pixel_stays_in_map_units():
    n = 48
    pix = 0.05
    obs = np.zeros((n, n))
    obs[n // 2, n // 2] = 7.0
    hits = np.ones_like(obs)
    dec = (np.arange(n) - n // 2) * pix
    params = {"sigma_deg": 0.15, "baseline": 0.0}
    out = Beam.subtract_gaussian_psf_contributions(
        obs,
        hits,
        params,
        pixel_size_deg=pix,
        dec_centers=dec,
        niter=1,
        baseline=0.0,
    )
    np.testing.assert_allclose(out["g00"], 1.0, atol=1e-6)
    peak = float(out["corrected_map"][n // 2, n // 2])
    assert peak == pytest.approx(7.0, rel=0, abs=8.0)
    assert np.nanmax(np.abs(out["corrected_map"])) < 30.0
    assert np.nanmax(np.abs(out["reconvolved_map"])) < 30.0


def test_uniform_map_matches_baseline():
    n = 48
    pix = 0.05
    floor = 12.0
    obs = np.full((n, n), floor)
    hits = np.ones_like(obs)
    dec = (np.arange(n) - n // 2) * pix
    params = {"sigma_deg": 0.15, "baseline": floor}
    out = Beam.subtract_gaussian_psf_contributions(
        obs,
        hits,
        params,
        pixel_size_deg=pix,
        dec_centers=dec,
        niter=1,
    )
    np.testing.assert_allclose(out["corrected_map"], floor, atol=1e-6)


def test_compact_plus_plateau_stays_in_data_units():
    n = 96
    pix = 0.04
    sigma_beam = 0.20
    plateau = 10.0
    amp = 40.0
    sigma_src = 0.06
    y = (np.arange(n) - n // 2) * pix
    Y, X = np.meshgrid(y, y, indexing="ij")
    r = np.sqrt(Y**2 + X**2)
    sky = plateau + amp * np.exp(-0.5 * (r / sigma_src) ** 2)
    g_sum = Beam._gaussian_psf_2d_on_grid(
        (n, n), pix, y, sigma_beam, normalize="sum"
    )
    observed = Beam._fft_convolve_centered(sky, g_sum)
    hits = np.ones_like(observed)
    params = {"sigma_deg": sigma_beam, "baseline": plateau, "FWHM_deg": 2.355 * sigma_beam}
    out = Beam.subtract_gaussian_psf_contributions(
        observed,
        hits,
        params,
        pixel_size_deg=pix,
        dec_centers=y,
        niter=1,
        baseline=plateau,
    )
    corr = out["corrected_map"]
    corners = np.array(
        [corr[2, 2], corr[2, -3], corr[-3, 2], corr[-3, -3]]
    )
    np.testing.assert_allclose(np.mean(corners), plateau, atol=1.5)
    obs_max = float(np.nanmax(np.abs(observed)))
    assert float(np.nanmax(np.abs(corr))) < 5.0 * obs_max
    assert float(np.nanmax(np.abs(out["reconvolved_map"]))) < 5.0 * obs_max
