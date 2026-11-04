'''
Beam estimation for 310 MHz observations

- Get RA and Dec of the input source from calibrators.dat.
- Using matched pointing + spectrum data (e.g. ``match_data_and_pointing``), compute
  per-pointing great-circle offset from the source (same convention as ``plot_offset_map``
  and ``grid_beam_obs_pointing``).
- Fit a Gaussian on a constant floor (mean brightness beyond ~1° offset) to estimate beam width.
- Fine-grid mapping: build an Astropy Gaussian PSF from the fit (``make_beam_psf``), then
  form each map pixel as a PSF-weighted average of nearby pointings
  (``map_pointings_with_psf`` / ``grid_beam_obs_pointing``). Pixel size defaults to FWHM/4
  so pixels are much smaller than the beam.
- Dirty-map + PSF convolution (Astropy ``convolve_fft``): bin pointings to a fine grid
  (``make_dirty_maps``), optionally form Stokes I/Q/U, then convolve with the beam PSF
  (``convolve_map_with_psf`` / ``convolve_dirty_stokes_with_psf``), following
  https://learn.astropy.org/tutorials/synthetic-images.html
- Image-plane FFT deconvolution to remove the PSF (``deconvolve_map_with_psf`` /
  ``deconvolve_stokes_with_psf`` / ``deconvolve_gridded_map``), using a regularized
  Fourier inverse filter ``F = H* / (|H|^2 + alpha)``.
- Optionally approximate removal of the fitted circular Gaussian PSF on a HealPix
  grid (``convolve_beam_with_fit``), using radio-beam (https://radio-beam.readthedocs.io/en/latest/)
  for the PSF model and healpy for regularized harmonic ``1 / B_\\ell``.
- Spatial Gaussian contribution subtraction on a fixed 1024×1024 RA/Dec patch
  (``grid_patch_fixed_shape`` + ``subtract_gaussian_psf_contributions``): keep
  pixel ``i``, subtract Gaussian leakage from every other pixel. Gridding uses
  linear interpolation among nearby pointings so a fine lattice does not
  resolve one circular kernel per sample.
'''

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import healpy as hp
import matplotlib.pyplot as plt
import warnings

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import lsqr
from skymap.healmap import HealPixMap
from skymap.io import get_available_pol_names, get_pol_source

# Default catalogue next to this module
_CALIBRATORS_PATH = Path(__file__).resolve().parent / "calibrators.dat"


def _to_jsonable(obj):
    """Recursively convert numpy/scalar objects to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):  # np.float64, np.int64, etc.
        return obj.item()
    return obj
def save_beam_fit(beam_params: dict, path: Path | str) -> Path:
    """Save full input beam_params dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(beam_params), f, indent=2)
    return path
def load_beam_fit(path: Path | str) -> dict:
    """Load beam params JSON for Beam.convolve_beam_with_fit."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_ra_hms(hms: str) -> float:
    """Parse 'HH:MM:SS.S' or 'HH:MM:SS' to decimal hours."""
    parts = hms.strip().split(":")
    h = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0.0
    s = float(parts[2]) if len(parts) > 2 else 0.0
    return h + m / 60.0 + s / 3600.0


def _parse_dec_dms(dms: str) -> float:
    """Parse '+DD:MM:SS.S' or '-DD:MM:SS.S' to decimal degrees."""
    s = dms.strip()
    sign = -1.0 if s.startswith("-") else 1.0
    s = s.lstrip("+-")
    parts = s.split(":")
    d = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0.0
    sec = float(parts[2]) if len(parts) > 2 else 0.0
    return sign * (d + m / 60.0 + sec / 3600.0)


def load_calibrators(path: Path | str | None = None) -> list[tuple[str, float, float]]:
    """
    Load calibrator catalogue. Returns list of (name, ra_deg, dec_deg).
    Skips comment lines and lines that don't have RA/Dec.
    """
    path = Path(path) if path is not None else _CALIBRATORS_PATH
    result: list[tuple[str, float, float]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expect at least: name, RA_str (HH:MM:SS), Dec_str (+DD:MM:SS), [optional peak/onoff]
            if len(parts) < 3:
                continue
            name = parts[0]
            ra_str = parts[1]
            dec_str = parts[2]
            try:
                ra_h = _parse_ra_hms(ra_str)
                ra_deg = ra_h * 15.0
                dec_deg = _parse_dec_dms(dec_str)
            except (ValueError, IndexError):
                continue
            result.append((name, ra_deg, dec_deg))
    return result


def get_source_radec(source_name: str, path: Path | str | None = None) -> tuple[float, float]:
    """
    Get (ra_deg, dec_deg) for a calibrator by name (first match).
    Raises KeyError if source is not found.
    """
    catalog = load_calibrators(path)
    for name, ra, dec in catalog:
        if name.strip() == source_name.strip():
            return (ra, dec)
    raise KeyError(f"Source {source_name!r} not found in calibrators. Known: {[n for n, _, _ in catalog]}")


def _normalize_ra_deg(ra: np.ndarray) -> np.ndarray:
    """Put RA in [0, 360) to avoid -180/360 wrap issues."""
    ra = np.asarray(ra, dtype=float)
    return ra % 360.0


def _unwrap_ra_deg(ra_deg: np.ndarray) -> np.ndarray:
    """Unwrap RA to a continuous interval around the sample median (degrees)."""
    ra = _normalize_ra_deg(ra_deg)
    finite = ra[np.isfinite(ra)]
    if finite.size == 0:
        return ra
    med = float(np.median(finite))
    return ra - 360.0 * np.round((ra - med) / 360.0)


def _characteristic_sample_spacing_deg(ra_deg: np.ndarray, dec_deg: np.ndarray) -> float:
    """
    Typical gap to close when filling a scan map (deg).

    Along-track nearest-neighbor spacing is much smaller than the distance
    *between* scan rows. Use the larger of mean cell size, median NN, and the
    median longest Delaunay edge (cross-track / cell diagonal) so pixels
    between scan lines are not left NaN.
    """
    from scipy.spatial import QhullError, cKDTree, Delaunay

    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    valid = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[valid]
    dec = dec[valid]
    if ra.size < 2:
        return 0.0
    ra_span = float(np.nanmax(ra) - np.nanmin(ra))
    dec_span = float(np.nanmax(dec) - np.nanmin(dec))
    area = max(ra_span * dec_span, 0.0)
    mean_cell = float(np.sqrt(area / float(ra.size))) if ra.size else 0.0
    cos_dec = max(abs(float(np.cos(np.radians(np.nanmean(dec))))), 1e-6)
    xy = np.column_stack(
        [
            (ra - float(np.nanmean(ra))) * cos_dec,
            dec - float(np.nanmean(dec)),
        ]
    )
    k = 2 if ra.size >= 2 else 1
    dists = cKDTree(xy).query(xy, k=k)[0]
    nn = dists[:, 1] if dists.ndim == 2 else np.array([])
    nn = nn[np.isfinite(nn) & (nn > 0)]
    median_nn = float(np.median(nn)) if nn.size else 0.0
    cross = 0.0
    if ra.size >= 4:
        try:
            tri = Delaunay(xy)
            pts = xy[tri.simplices]
            d01 = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
            d12 = np.linalg.norm(pts[:, 1] - pts[:, 2], axis=1)
            d20 = np.linalg.norm(pts[:, 2] - pts[:, 0], axis=1)
            longest = np.maximum(np.maximum(d01, d12), d20)
            longest = longest[np.isfinite(longest) & (longest > 0)]
            if longest.size:
                # Midpoint between scan rows is ~0.5 × row spacing; longest
                # Delaunay edges are that spacing (or the cell diagonal).
                cross = 0.6 * float(np.median(longest))
        except (QhullError, ValueError):
            cross = 0.0
    return max(mean_cell, median_nn, cross, 0.0)


def _scan_line_halfwidth_deg(coord: np.ndarray, pixel_size_deg: float | None) -> float:
    """Half-width that keeps samples on one scan line of ``coord`` (RA or Dec)."""
    c = np.asarray(coord, dtype=float)
    c = c[np.isfinite(c)]
    pix = float(pixel_size_deg) if pixel_size_deg is not None else 0.01
    if c.size < 2:
        return max(2.0 * pix, 0.01)
    rounded = np.round(c, decimals=4)
    uniq = np.unique(rounded)
    if uniq.size < 2:
        return max(2.0 * pix, 0.01)
    gaps = np.diff(np.sort(uniq))
    gaps = gaps[gaps > 1e-5]
    if gaps.size == 0:
        return max(2.0 * pix, 0.01)
    return 0.35 * float(np.median(gaps))


def _angular_separation_deg(
    ra1_deg: np.ndarray,
    dec1_deg: np.ndarray,
    ra2_deg: np.ndarray,
    dec2_deg: np.ndarray,
) -> np.ndarray:
    """
    Great-circle angular separation in degrees (haversine).

    Accounts for declination convergence: RA differences are scaled by
    ``cos(dec)`` on the sphere. Broadcasts over leading dimensions.
    """
    ra1 = np.radians(_normalize_ra_deg(np.asarray(ra1_deg, dtype=float)))
    dec1 = np.radians(np.asarray(dec1_deg, dtype=float))
    ra2 = np.radians(_normalize_ra_deg(np.asarray(ra2_deg, dtype=float)))
    dec2 = np.radians(np.asarray(dec2_deg, dtype=float))
    dra = ((ra2 - ra1 + np.pi) % (2.0 * np.pi)) - np.pi
    ddec = dec2 - dec1
    sin_dra = np.sin(0.5 * dra)
    sin_ddec = np.sin(0.5 * ddec)
    a = sin_ddec * sin_ddec + np.cos(dec1) * np.cos(dec2) * sin_dra * sin_dra
    a = np.clip(a, 0.0, 1.0)
    return np.degrees(2.0 * np.arcsin(np.sqrt(a)))


def _flat_sky_radial_offset_deg(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    source_ra_deg: float,
    source_dec_deg: float,
) -> np.ndarray:
    """
    Great-circle offset from the source in degrees.

    Uses spherical geometry (haversine), not a flat ``sqrt(dRA^2 + dDec^2)``
    plane. Suitable for wide patches (tens of degrees).
    """
    ra_deg = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec_deg = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    src_ra = float(_normalize_ra_deg(np.array([float(source_ra_deg)]))[0])
    src_dec = float(source_dec_deg)
    return _angular_separation_deg(
        ra_deg,
        dec_deg,
        np.full_like(ra_deg, src_ra, dtype=float),
        np.full_like(dec_deg, src_dec, dtype=float),
    )


def radial_offsets_from_source(
    beam_obs_pointing: object,
    *,
    attribute: str,
    source_name: str | None = None,
    source_ra_deg: float | None = None,
    source_dec_deg: float | None = None,
    calibrators_path: Path | str | None = None,
    max_offset_deg: float | None = None,
    freq_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-pointing radial offset (deg) from the source and brightness for one pol channel.

    ``beam_obs_pointing`` should be the result of ``match_data_and_pointing`` (an
    ``HDF5Data`` with ``ra``, ``dec``, and ``calibrated_spec_mean`` or ``spec_mean``).

    Values are taken from the chosen ``attribute`` (e.g. ``\"AB_\"``), same rules as
    ``HealPixMap.fill_from_pointing_data``: if the channel is ``(n_pointing, n_freq)``,
    use ``freq_index`` or else mean over frequency.

    Source is specified by either ``source_name`` (looked up in calibrators.dat)
    or explicit ``source_ra_deg`` + ``source_dec_deg``.

    Offset convention matches ``plot_offset_map`` (great-circle, RA wrapped).
    """
    ra_deg = np.atleast_1d(np.asarray(getattr(beam_obs_pointing, "ra"), dtype=float))
    dec_deg2 = np.atleast_1d(np.asarray(getattr(beam_obs_pointing, "dec"), dtype=float))
    if ra_deg.size != dec_deg2.size:
        raise ValueError("beam_obs_pointing must have ra and dec of the same length")

    spec_source = get_pol_source(beam_obs_pointing, kind="mean")
    if spec_source is None:
        raise ValueError(
            "beam_obs_pointing must have calibrated_spec_mean or spec_mean "
            "(e.g. output of match_data_and_pointing)"
        )
    available = get_available_pol_names(beam_obs_pointing, kind="mean")
    if attribute not in available:
        raise ValueError(
            f"Attribute {attribute!r} not available on beam_obs_pointing. Available: {available}"
        )
    arr = getattr(spec_source, attribute)
    if getattr(arr, "unit", None) is not None:
        arr = getattr(arr, "value", arr)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        if freq_index is not None:
            values2 = arr[:, int(freq_index)]
        else:
            values2 = np.nanmean(arr, axis=1)
    else:
        values2 = arr.ravel()
    values2 = np.atleast_1d(values2)

    if source_ra_deg is not None and source_dec_deg is not None:
        src_ra, src_dec = source_ra_deg, source_dec_deg
    elif source_name is not None:
        src_ra, src_dec = get_source_radec(source_name, path=calibrators_path)
    else:
        raise ValueError("Provide either source_name or (source_ra_deg, source_dec_deg)")

    values2 = np.atleast_1d(np.asarray(values2, dtype=float))
    if ra_deg.size != values2.size:
        raise ValueError(
            f"Length mismatch: ra/dec have {ra_deg.size} pointings, "
            f"{attribute!r} has {values2.size} values"
        )
    if ra_deg.size == 0:
        return np.array([]), np.array([])

    offset_deg = _flat_sky_radial_offset_deg(ra_deg, dec_deg2, src_ra, src_dec)

    return offset_deg, values2


def plot_offset_map(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    offset_deg: np.ndarray | None = None,
    *,
    source_ra_deg: float | None = None,
    source_dec_deg: float | None = None,
    resolution_deg: float = 0.1,
    ax: Any = None,
    xlabel: str = "RA (deg)",
    ylabel: str = "Dec (deg)",
    cbar_label: str = "Radial offset from source (deg)",
    mark_source: bool = True,
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Plot offset from source as color on RA/Dec in the **absolute** frame: x=RA,
    y=Dec, color=offset_deg. All RA/Dec in degrees (RA 0--360 or -180--180;
    Dec -90--90). RA is normalized to [0, 360) so offset is consistent.

    If `source_ra_deg` and `source_dec_deg` are provided, offset is the
    **great-circle** separation from the source (includes ``cos(dec)`` convergence).
    """
    ra_deg = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec_deg = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    # Normalize RA to [0, 360) so angular distance and grid are consistent
    ra_deg = _normalize_ra_deg(ra_deg)
    src_ra = _normalize_ra_deg(np.array([float(source_ra_deg)]))[0] if source_ra_deg is not None else None
    src_dec = float(source_dec_deg) if source_dec_deg is not None else None
    if src_ra is not None and src_dec is not None:
        offset_deg = _flat_sky_radial_offset_deg(ra_deg, dec_deg, src_ra, src_dec)
    else:
        if offset_deg is None:
            raise ValueError("Provide offset_deg or (source_ra_deg, source_dec_deg)")
        offset_deg = np.atleast_1d(np.asarray(offset_deg, dtype=float))
    if ra_deg.size != dec_deg.size or ra_deg.size != offset_deg.size:
        raise ValueError("ra_deg, dec_deg, offset_deg must have the same length")
    if ra_deg.size == 0:
        if ax is None:
            ax = plt.gca()
        return ax

    x = ra_deg
    y = dec_deg
    x_label = xlabel
    y_label = ylabel
    mark_x, mark_y = (src_ra, src_dec) if (src_ra is not None and src_dec is not None) else (None, None)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    x_edges = np.arange(x_min, x_max + resolution_deg * 0.5, resolution_deg)
    y_edges = np.arange(y_min, y_max + resolution_deg * 0.5, resolution_deg)
    if x_edges.size < 2 or y_edges.size < 2:
        x_edges = np.linspace(x_min, x_max, max(2, int((x_max - x_min) / resolution_deg) + 1))
        y_edges = np.linspace(y_min, y_max, max(2, int((y_max - y_min) / resolution_deg) + 1))

    j = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, len(x_edges) - 2)
    i = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, len(y_edges) - 2)
    n_y, n_x = len(y_edges) - 1, len(x_edges) - 1
    sum_off = np.full((n_y, n_x), 0.0)
    count = np.zeros((n_y, n_x), dtype=float)
    np.add.at(sum_off, (i, j), offset_deg)
    np.add.at(count, (i, j), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.where(count > 0, sum_off / count, np.nan)

    if ax is None:
        fig, ax = plt.subplots()
    pc = ax.pcolormesh(x_edges, y_edges, Z, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    if mark_source and mark_x is not None and mark_y is not None:
        ax.plot(mark_x, mark_y, "k*", ms=12, label="Source")
    cb = plt.colorbar(pc, ax=ax)
    cb.set_label(cbar_label)
    if show:
        plt.show()
    return ax


def _gaussian_only(x: np.ndarray, A: float, sigma: float) -> np.ndarray:
    return A * np.exp(-(x**2) / (2.0 * sigma**2))


def _sigma_deg_from_beam_params(beam_params: dict[str, Any]) -> float:
    """σ in degrees from fit dict (FWHM keys use FWHM → σ = FWHM / (2√(2 ln 2)))."""
    _sqrt_2_ln2 = np.sqrt(2.0 * np.log(2.0))
    if "FWHM_deg" in beam_params:
        fwhm_deg = float(beam_params["FWHM_deg"])
        if not np.isfinite(fwhm_deg) or fwhm_deg <= 0:
            raise ValueError("FWHM_deg must be positive and finite")
        return fwhm_deg / (2.0 * _sqrt_2_ln2)
    if "half_power_radius_deg" in beam_params:
        fwhm_deg = float(beam_params["half_power_radius_deg"])
        if not np.isfinite(fwhm_deg) or fwhm_deg <= 0:
            raise ValueError("half_power_radius_deg must be positive and finite")
        return fwhm_deg / (2.0 * _sqrt_2_ln2)
    if "sigma_deg" in beam_params:
        sigma_deg = float(beam_params["sigma_deg"])
        if not np.isfinite(sigma_deg) or sigma_deg <= 0:
            raise ValueError("sigma_deg must be positive and finite")
        return sigma_deg
    raise ValueError(
        "beam_params must include 'FWHM_deg', 'half_power_radius_deg' "
        "(each interpreted as Gaussian FWHM in degrees), or 'sigma_deg'."
    )


def beam_approximation(
    radial_offset_deg: np.ndarray,
    beam_params: dict[str, Any],
) -> np.ndarray:
    """
    Axisymmetric beam (radial offset in **degrees**):

        ``baseline_k + A exp(-x²/(2σ²))``

    ``baseline_k`` comes from ``fit_beam_gaussian`` (mean of samples with
    ``x > baseline_outer_deg``); if missing (legacy dicts), it defaults to 0.

    Requires ``A`` and a width key (``sigma_deg``, ``FWHM_deg``, or
    ``half_power_radius_deg``).
    """
    if "A" not in beam_params:
        raise ValueError("beam_params must include 'A' (output of fit_beam_gaussian).")
    x = np.asarray(radial_offset_deg, dtype=float)
    b = float(beam_params.get("baseline_k", 0.0))
    return b + _gaussian_excess_from_params(x, beam_params)


# Radial edge (deg) for [0, 1] beam normalization: 1 at r=0, 0 at this radius.
_BEAM_NORM_R_MAX_DEG = 5.0
# Default outer radius (deg) for estimating constant floor in ``fit_beam_gaussian``.
_BASELINE_OUTER_RADIUS_DEG = 1.0


def _gaussian_excess_from_params(x: np.ndarray, beam_params: dict[str, Any]) -> np.ndarray:
    """Gaussian bump ``A exp(-x²/(2σ²))`` only (no floor); used for kernels / beam2bl."""
    A = float(beam_params["A"])
    sigma_deg = _sigma_deg_from_beam_params(beam_params)
    return _gaussian_only(np.asarray(x, dtype=float), A, sigma_deg)


def _beam_approx_endpoint_affine(
    beam_params: dict[str, Any],
    r_edge_deg: float = _BEAM_NORM_R_MAX_DEG,
) -> tuple[float, float, float]:
    """
    Return ``(B(0), B(r_edge), B(0) - B(r_edge))`` from ``beam_approximation``.

    Used to affine-map the fitted radial profile to 1 at the origin and 0 at
    ``r_edge_deg`` (requires ``B(0) > B(r_edge)``). Used by ``plot_beam_approximation``.
    """
    r_edge_deg = float(r_edge_deg)
    if r_edge_deg <= 0:
        raise ValueError("r_edge_deg must be positive")
    B0 = float(beam_approximation(np.array([0.0]), beam_params)[0])
    Be = float(beam_approximation(np.array([r_edge_deg]), beam_params)[0])
    denom = B0 - Be
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(
            "Endpoint beam normalization requires beam_approximation(0°) > "
            f"beam_approximation({r_edge_deg:g}°); got B(0)={B0}, B({r_edge_deg:g}°)={Be}."
        )
    return B0, Be, denom


def _beam_approx_radial_endpoint_normalized(
    radial_offset_deg: np.ndarray,
    beam_params: dict[str, Any],
    *,
    r_edge_deg: float = _BEAM_NORM_R_MAX_DEG,
    endpoint_affine: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Scale ``beam_approximation`` to the ``[0, 1]`` interval with **W(0°)=1** and
    **W(5°)=0** (default ``r_edge_deg``), matching the normalized profile in
    ``plot_beam_approximation``. Values are forced to 0 for ``r >= r_edge_deg``;
    intermediate radii follow the fit, then values are clipped to ``[0, 1]``.

    If ``endpoint_affine`` is given, it must be ``(B(0), B(r_edge), denom)``
    from ``_beam_approx_endpoint_affine`` (avoids recomputing B at 0 and r_edge).
    """
    if endpoint_affine is None:
        _, Be, denom = _beam_approx_endpoint_affine(beam_params, r_edge_deg)
    else:
        _, Be, denom = endpoint_affine
    r = np.asarray(radial_offset_deg, dtype=float)
    B = np.asarray(beam_approximation(r, beam_params), dtype=float)
    W = (B - Be) / denom
    W = np.where(r >= float(r_edge_deg), 0.0, W)
    return np.clip(W, 0.0, 1.0)


def plot_beam_approximation(
    beam_params: dict[str, Any],
    *,
    x_max_deg: float = 5.0,
    npts: int = 500,
    normalize: bool = True,
    show_components: bool = False,
    ax: Any = None,
    show: bool = True,
    title: str | None = "Gaussian beam",
) -> Any:
    """
    Plot ``beam_approximation`` vs radial offset (degrees).

    Parameters
    ----------
    beam_params : dict
        Same dictionary passed to ``beam_approximation`` (e.g. return value of
        ``fit_beam_gaussian``).
    x_max_deg : float
        Upper limit of the horizontal axis (deg); capped at 5° to match the
        default endpoint-normalization radius.
    npts : int
        Number of samples along the radius.
    normalize : bool
        If True (default), scale the profile to ``[0, 1]`` with amplitude 1 at
        0° and 0 at 5° (affine map of ``beam_approximation``).
    show_components : bool
        If True, also draw the analytic Gaussian (dotted) behind the plotted curve.
    ax : matplotlib.axes.Axes or None
        Axis to draw on; if ``None``, a new figure is created.
    show : bool
        If True, call ``plt.show()``.
    title : str or None
        Axes title; ``None`` to omit.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if x_max_deg <= 0:
        raise ValueError("x_max_deg must be positive")
    npts = int(max(2, npts))
    x_hi = min(float(x_max_deg), _BEAM_NORM_R_MAX_DEG)
    x = np.linspace(0.0, x_hi, npts)

    if normalize:
        affine = _beam_approx_endpoint_affine(beam_params, _BEAM_NORM_R_MAX_DEG)
        y = _beam_approx_radial_endpoint_normalized(
            x, beam_params, endpoint_affine=affine
        )
    else:
        y = np.asarray(beam_approximation(x, beam_params), dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if show_components:
        b0 = float(beam_params.get("baseline_k", 0.0))
        g_ex = _gaussian_excess_from_params(x, beam_params)
        if normalize:
            g0e = float(_gaussian_excess_from_params(np.array([0.0]), beam_params)[0])
            g5e = float(
                _gaussian_excess_from_params(np.array([_BEAM_NORM_R_MAX_DEG]), beam_params)[0]
            )
            den_g = g0e - g5e
            if den_g > 0:
                g_full = np.clip((g_ex - g5e) / den_g, 0.0, 1.0)
            else:
                g_full = np.zeros_like(g_ex)
        else:
            g_full = b0 + g_ex
        ax.plot(
            x,
            g_full,
            ":",
            color="0.6",
            lw=1.2,
            alpha=0.85,
            label=(
                r"Model $T_0 + A e^{-r^2/(2\sigma^2)}$"
                if b0 != 0.0
                else r"Gaussian $A e^{-r^2/(2\sigma^2)}$"
            ),
        )

    ax.plot(
        x,
        y,
        "b-",
        lw=2,
        label="Beam B(x)" if not normalize else r"$B(r)$: 1 at $0°$, 0 at $5°$",
    )

    ax.set_xlabel("Radial offset (deg)")
    ax.set_ylabel("Normalized amplitude" if normalize else "Amplitude")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    ax.set_xlim(0.0, x_hi)
    if show:
        plt.show()
    return ax



def _baseline_k_from_outer_ring(
    x: np.ndarray, y: np.ndarray, r_outer_deg: float
) -> tuple[float, bool]:
    """Mean *y* where *x* > ``r_outer_deg``. Returns ``(baseline_k, used_outer_mask)``."""
    m = x > float(r_outer_deg)
    if not np.any(m):
        return float(np.nanmean(y)), False
    return float(np.nanmean(y[m])), True


def _gaussian_plus_fixed_baseline(baseline_k: float):
    """``baseline_k + A exp(-x²/(2σ²))`` for ``curve_fit`` in (A, σ)."""

    def f(xx: np.ndarray, A: float, sigma: float) -> np.ndarray:
        return baseline_k + _gaussian_only(xx, A, sigma)

    return f


def fit_beam_gaussian(
    radial_offset_deg: np.ndarray,
    values_k: np.ndarray,
    *,
    baseline_outer_deg: float = _BASELINE_OUTER_RADIUS_DEG,
    ax: Any = None,
    show: bool = True,
    title: str = "Beam profile",
) -> dict[str, Any]:
    """
    Fit a Gaussian **on top of a constant floor** (radial offset in degrees):

        ``T(x) = baseline_k + A exp(-x²/(2σ²))``

    ``baseline_k`` is set to the mean of all samples with
    ``radial_offset_deg > baseline_outer_deg`` (default 1°), not fitted. If no
    sample lies beyond that radius, the mean of all *y* is used with a warning.

    ``radial_offset_deg`` and ``values_k`` are parallel arrays (one per pointing),
    e.g. from ``radial_offsets_from_source``.

    **Fitting strategy**

    ``curve_fit`` for ``A`` and ``σ`` on the full valid ``(x, y)``. Then **raise**
    ``A`` so the global maximum lies on or under the model.

    **FWHM / half-power radius** refer to the Gaussian **bump** only (same
    formulas in ``σ``).

    Parameters
    ----------
    baseline_outer_deg : float
        Floor estimate: mean *y* where offset exceeds this (degrees); default 1°.
    """

    mask = np.isfinite(values_k) & np.isfinite(radial_offset_deg)
    x = np.asarray(radial_offset_deg[mask], dtype=float)
    y = np.asarray(values_k[mask], dtype=float)

    if x.size < 2:
        raise ValueError(f"Need at least 2 valid data points, got {x.size}")

    baseline_k, used_outer = _baseline_k_from_outer_ring(x, y, baseline_outer_deg)
    if not used_outer:
        warnings.warn(
            f"No samples with radial offset > {baseline_outer_deg:g}°; "
            "using mean of all y for baseline_k.",
            UserWarning,
            stacklevel=2,
        )

    x_fit = np.linspace(0.0, float(np.max(x)), 500)
    model = _gaussian_plus_fixed_baseline(baseline_k)

    A0 = float(max(np.nanmax(y) - baseline_k, 1e-6))
    i_pk = int(np.nanargmax(y))
    sigma0 = max(float(x[i_pk]) / 2.0, 1e-4)
    try:
        (A, sigma), _ = curve_fit(
            model,
            x,
            y,
            p0=[A0, sigma0],
            absolute_sigma=False,
            bounds=([0.0, 1e-6], [np.inf, 180.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        (A, sigma), _ = curve_fit(
            model,
            x,
            y,
            p0=[A0, sigma0],
            absolute_sigma=False,
            maxfev=20000,
        )
    A, sigma = float(A), float(sigma)
    idx_max = int(np.nanargmax(y))
    phi_pk = float(np.exp(-(float(x[idx_max]) ** 2) / (2.0 * sigma**2)))
    need = float(y[idx_max]) - baseline_k
    if need > 0:
        A = max(A, need / max(phi_pk, 1e-15))

    y_model = baseline_k + _gaussian_only(x_fit, A, sigma)

    sqrt_2_ln2 = np.sqrt(2.0 * np.log(2.0))
    hpbw = float(sqrt_2_ln2 * sigma)
    fwhm = float(2.0 * hpbw)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, s=12, label="Data (K)", zorder=2, alpha=0.35, color="C0")
    ax.axhline(
        baseline_k,
        color="k",
        ls=":",
        lw=1.0,
        zorder=1,
        label=rf"baseline (mean $r>{baseline_outer_deg:g}°$)",
    )
    ax.plot(
        x_fit,
        y_model,
        color="black",
        lw=2.0,
        zorder=4,
        label=rf"Model: $\sigma={sigma:.3f}$, HPBW={hpbw:.3f}$, FWHM={fwhm:.3f}$",
    )
    ax.set_xlabel("Radial offset (deg)")
    ax.set_ylabel("K")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if show:
        plt.show()

    return {
        "A": A,
        "sigma_deg": sigma,
        "FWHM_deg": fwhm,
        "half_power_radius_deg": hpbw,
        "baseline_k": baseline_k,
        "baseline_outer_deg": float(baseline_outer_deg),
        "covariance": None,
        "y_fit_gaussian": y_model,
        "x_fit": x_fit,
    }


def _fwhm_deg_from_beam_params(beam_params: dict[str, Any]) -> float:
    """Circular Gaussian FWHM in degrees from ``fit_beam_gaussian`` output."""
    if "FWHM_deg" in beam_params:
        fwhm_deg = float(beam_params["FWHM_deg"])
    else:
        sig = _sigma_deg_from_beam_params(beam_params)
        fwhm_deg = float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sig)
    if not np.isfinite(fwhm_deg) or fwhm_deg <= 0:
        raise ValueError("beam_params must define a positive Gaussian FWHM in degrees.")
    return fwhm_deg


def _radio_beam_psf_from_fit(beam_params: dict[str, Any]) -> tuple[Any, float]:
    """
    Circular ``radio_beam.Beam`` matching the fitted FWHM (see radio-beam docs).

    Returns
    -------
    psf_beam, fwhm_deg
        Astropy-quantity-based PSF object and FWHM in degrees for metadata.
    """
    from astropy import units as u
    from radio_beam import Beam as RadioBeam

    fwhm_deg = _fwhm_deg_from_beam_params(beam_params)
    psf = RadioBeam((fwhm_deg * u.deg).to(u.arcsec))
    return psf, fwhm_deg


def _deconvolve_healpix_gaussian_psf(
    map_ring: np.ndarray,
    hit_count: np.ndarray,
    nside: int,
    fwhm_rad: float,
    *,
    bl_floor: float,
    lmax: int | None,
    map2alm_iter: int,
    mask_unseen: bool = True,
    method: str = "regularized",
    regularization_alpha: float = 1e-6,
    signal_cl: np.ndarray | None = None,
    noise_cl: np.ndarray | None = None,
) -> np.ndarray:
    """
    Deconvolve an axisymmetric Gaussian beam in spherical-harmonic space.

    ``map_ring`` and the Gaussian beam are represented in spherical harmonics with
    healpy:
      - ``alm_obs = hp.map2alm(map_ring)``
      - ``B_l = hp.gauss_beam(fwhm_rad)``

    Then one of two filters is applied:
      - ``regularized`` (default): ``F_l = B_l / (B_l^2 + alpha)``
      - ``wiener`` / ``weiner``: ``F_l = B_l C_l^S / (B_l^2 C_l^S + C_l^N)``

    Pixels with ``hit_count <= 0`` are set to ``UNSEEN`` before ``map2alm``.
    """
    lmax_i = int(lmax if lmax is not None else 3 * int(nside) - 1)
    m = np.asarray(map_ring, dtype=np.float64).copy()
    hits = np.asarray(hit_count, dtype=float)
    if mask_unseen:
        m[hits <= 0] = hp.UNSEEN
    bl = hp.gauss_beam(float(fwhm_rad), lmax=lmax_i)
    bl_max = float(np.max(bl[1:])) if bl.size > 1 else float(bl[0])
    thresh = float(bl_floor) * bl_max
    mth = str(method).strip().lower()
    filt = np.zeros(lmax_i + 1, dtype=np.float64)
    if mth == "regularized":
        alpha = float(regularization_alpha)
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("regularization_alpha must be finite and >= 0")
        denom = bl * bl + alpha
        good = (bl > thresh) & (denom > 0)
        filt[good] = bl[good] / denom[good]
    elif mth in {"wiener", "weiner"}:
        if signal_cl is None:
            obs_cl = hp.anafast(m, lmax=lmax_i)
            signal_eff = obs_cl / np.maximum(bl * bl, 1e-12)
            signal_eff = np.clip(signal_eff, 0.0, None)
        else:
            signal_eff = np.asarray(signal_cl, dtype=float).ravel()
        if signal_eff.size < (lmax_i + 1):
            raise ValueError(
                f"signal_cl must have at least {lmax_i + 1} elements for lmax={lmax_i}"
            )
        signal_eff = signal_eff[: lmax_i + 1]
        if noise_cl is None:
            obs_cl = hp.anafast(m, lmax=lmax_i)
            hi = max(2, int(0.8 * lmax_i))
            white = float(np.nanmedian(obs_cl[hi:])) if lmax_i > hi else float(np.nanmedian(obs_cl[1:]))
            white = max(white, 0.0)
            noise_eff = np.full(lmax_i + 1, white, dtype=float)
        else:
            noise_eff = np.asarray(noise_cl, dtype=float).ravel()
        if noise_eff.size < (lmax_i + 1):
            raise ValueError(
                f"noise_cl must have at least {lmax_i + 1} elements for lmax={lmax_i}"
            )
        noise_eff = np.clip(noise_eff[: lmax_i + 1], 0.0, None)
        denom = bl * bl * signal_eff + noise_eff
        good = (bl > thresh) & (denom > 0)
        filt[good] = (bl[good] * signal_eff[good]) / denom[good]
    else:
        raise ValueError("method must be 'regularized' or 'wiener'/'weiner'")
    alm = hp.map2alm(m, lmax=lmax_i, iter=int(map2alm_iter), pol=False)
    alm_out = hp.almxfl(alm, filt)
    out = hp.alm2map(alm_out, int(nside))
    out = np.asarray(out, dtype=float)
    out[hits <= 0] = 0.0
    return out


def _healpix_scalar_fallback(
    map_ring: np.ndarray,
    hit_count: np.ndarray,
    pad_value: float | str,
) -> float:
    """Scalar fallback for HealPix pixels far from any observation."""
    m = np.asarray(map_ring, dtype=float)
    hits = np.asarray(hit_count, dtype=float) > 0
    valid = hits & np.isfinite(m) & (m != hp.UNSEEN)
    if isinstance(pad_value, str):
        key = pad_value.strip().lower()
        if key in {"edge", "nearest", "boundary"}:
            return float(np.nanmedian(m[valid])) if np.any(valid) else 0.0
        if key == "median":
            return float(np.nanmedian(m[valid])) if np.any(valid) else 0.0
        if key == "mean":
            return float(np.nanmean(m[valid])) if np.any(valid) else 0.0
        if key in {"zero", "0"}:
            return 0.0
        raise ValueError(
            "pad_value must be a float or one of {'edge','median','mean','zero'}"
        )
    return float(pad_value)


def _healpix_edge_extrapolated_map(
    map_ring: np.ndarray,
    hit_count: np.ndarray,
    pad_value: float | str,
    *,
    max_extrap_sep_deg: float = 20.0,
) -> np.ndarray:
    """
    Per-pixel fill from the nearest observed HealPix pixel (local edge spectrum).

    Pixels farther than ``max_extrap_sep_deg`` from any hit use the scalar fallback.
    """
    from scipy.spatial import cKDTree

    m = np.asarray(map_ring, dtype=np.float64)
    hits = np.asarray(hit_count, dtype=float) > 0
    valid = hits & np.isfinite(m) & (m != hp.UNSEEN)
    fallback = _healpix_scalar_fallback(map_ring, hit_count, pad_value)
    npix = m.size
    edge = np.full(npix, fallback, dtype=np.float64)
    if not np.any(valid):
        return edge
    nside = hp.npix2nside(npix)
    obs_ipix = np.where(valid)[0]
    xo, yo, zo = hp.pix2vec(nside, obs_ipix)
    tree = cKDTree(np.column_stack([xo, yo, zo]))
    all_ipix = np.arange(npix, dtype=int)
    x, y, z = hp.pix2vec(nside, all_ipix)
    chord, nn = tree.query(np.column_stack([x, y, z]), k=1, workers=-1)
    sep_deg = np.degrees(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)))
    use = sep_deg <= float(max_extrap_sep_deg)
    edge[all_ipix[use]] = m[obs_ipix[nn[use]]]
    edge[obs_ipix] = m[obs_ipix]
    return edge


def _pad_unobserved_healpix_map(
    map_ring: np.ndarray,
    hit_count: np.ndarray,
    *,
    pad_value: float | str = "edge",
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = None,
    taper_fwhm_deg: float | None = None,
    taper_normalize: bool = True,
    max_extrap_sep_deg: float = 20.0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Taper HealPix maps toward per-pixel edge extrapolation (nearest observed pixel).

    taper_width_deg : float or None
        If set and ``gaussian_taper_fwhm_deg`` is None, uses this value (degrees) as
        ``gaussian_taper_fwhm_deg`` (Gaussian smooth of the hit mask on the sphere).

    Returns
    -------
    padded_map_ring, scalar_fallback, taper_weight, edge_target_map
    """
    m = np.asarray(map_ring, dtype=np.float64).copy()
    hits = np.asarray(hit_count, dtype=float)

    observed = hits > 0
    valid_obs = observed & np.isfinite(m) & (m != hp.UNSEEN)
    fallback = _healpix_scalar_fallback(map_ring, hit_count, pad_value)
    edge_target = _healpix_edge_extrapolated_map(
        map_ring,
        hit_count,
        pad_value,
        max_extrap_sep_deg=max_extrap_sep_deg,
    )
    taper_w = observed.astype(np.float64)

    m[~valid_obs & observed] = edge_target[~valid_obs & observed]
    m[m == hp.UNSEEN] = edge_target[m == hp.UNSEEN]
    m[~np.isfinite(m)] = edge_target[~np.isfinite(m)]
    m[~observed] = edge_target[~observed]

    gtf = taper_fwhm_deg
    if gtf is None and taper_width_deg is not None:
        gtf = float(taper_width_deg)

    if apodize_fwhm_deg is not None:
        fwhm_deg = float(apodize_fwhm_deg)
        if not np.isfinite(fwhm_deg) or fwhm_deg <= 0:
            raise ValueError("apodize_fwhm_deg must be positive and finite when provided")
        w = hp.smoothing(observed.astype(np.float64), fwhm=np.radians(fwhm_deg), verbose=False)
        w = np.asarray(w, dtype=np.float64)
        w = np.clip(w, 0.0, 1.0)
        taper_w = w.copy()
        m_obs = np.where(valid_obs, m, edge_target)
        m = edge_target * (1.0 - w) + m_obs * w

    if gtf is not None:
        tf = float(gtf)
        if not np.isfinite(tf) or tf <= 0:
            raise ValueError("taper_fwhm_deg / taper_width_deg must be positive and finite")
        w = hp.smoothing(observed.astype(np.float64), fwhm=np.radians(tf), verbose=False)
        w = np.asarray(w, dtype=np.float64)
        w = np.clip(w, 0.0, 1.0)
        if taper_normalize:
            wmax = float(np.nanmax(w)) if w.size else 1.0
            if np.isfinite(wmax) and wmax > 0:
                w = np.clip(w / wmax, 0.0, 1.0)
        taper_w = w.copy()
        m_obs = np.where(valid_obs, m, edge_target)
        m = edge_target * (1.0 - w) + m_obs * w

    return m, fallback, taper_w, edge_target


def estimate_noise_spectrum_from_deconvolution(
    observed_map_ring: np.ndarray,
    deconvolved_map_ring: np.ndarray,
    hit_count: np.ndarray,
    *,
    fwhm_rad: float,
    lmax: int | None = None,
    map2alm_iter: int = 3,
) -> dict[str, np.ndarray | float]:
    # NOTE: This intentionally uses UNSEEN outside the observed region; it estimates
    # residuals only where hits>0.
    """
    Estimate noise angular power spectrum ``C_l^N`` from deconvolution residuals.

    Procedure:
      1. Compute sky ``alm`` from the deconvolved map.
      2. Re-convolve by multiplying by Gaussian ``B_l``.
      3. Residual map: ``noise ~= observed - reconvolved_model``.
      4. Estimate noise spectrum with ``hp.anafast(residual)``.
    """
    obs = np.asarray(observed_map_ring, dtype=float).copy()
    dec = np.asarray(deconvolved_map_ring, dtype=float).copy()
    hits = np.asarray(hit_count, dtype=float)
    if obs.shape != dec.shape or obs.shape != hits.shape:
        raise ValueError("observed_map_ring, deconvolved_map_ring, hit_count must have identical shape")
    nside = hp.npix2nside(obs.size)
    lmax_i = int(lmax if lmax is not None else 3 * int(nside) - 1)
    obs[hits <= 0] = hp.UNSEEN
    dec[hits <= 0] = hp.UNSEEN

    alm_sky = hp.map2alm(dec, lmax=lmax_i, iter=int(map2alm_iter), pol=False)
    bl = hp.gauss_beam(float(fwhm_rad), lmax=lmax_i)
    alm_model_obs = hp.almxfl(alm_sky, bl)
    model_obs = hp.alm2map(alm_model_obs, int(nside))

    resid = np.asarray(obs, dtype=float).copy()
    valid = hits > 0
    resid[valid] = obs[valid] - model_obs[valid]
    resid[~valid] = hp.UNSEEN
    noise_cl = hp.anafast(resid, lmax=lmax_i)
    hi = max(2, int(0.8 * lmax_i))
    white_level = (
        float(np.nanmedian(noise_cl[hi:])) if lmax_i > hi else float(np.nanmedian(noise_cl[1:]))
    )
    if not np.isfinite(white_level):
        white_level = 0.0
    return {
        "noise_cl": np.asarray(noise_cl, dtype=float),
        "noise_white_level": float(max(white_level, 0.0)),
        "residual_map": np.asarray(np.where(valid, resid, 0.0), dtype=float),
    }


def _interp_healpix_ring_to_radec_deg(
    map_ring: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation (RING) at (ra_deg, dec_deg)."""
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    theta = np.radians(90.0 - dec_deg)
    phi = np.radians(ra_deg % 360.0)
    return np.asarray(hp.get_interp_val(map_ring, theta, phi), dtype=float)


def _pol_channel_values_1d(
    spec_source: object,
    attribute: str,
    freq_index: int | None,
    *,
    reduce: str = "mean",
) -> np.ndarray:
    """One row per pointing for pol channel ``attribute``.

    ``reduce='mean'`` (default): real part, then mean over frequency.
    ``reduce='mag_mean'``: ``|z|`` first, then mean over frequency.
    """
    arr = getattr(spec_source, attribute)
    if getattr(arr, "unit", None) is not None:
        arr = getattr(arr, "value", arr)
    arr = np.asarray(arr)
    key = str(reduce).strip().lower()
    if key in {"mag_mean", "magnitude_mean", "abs_mean", "mag"}:
        arr = np.abs(arr)
    elif key in {"mean", "real_mean", "real"}:
        if np.iscomplexobj(arr):
            arr = np.real(arr)
    else:
        raise ValueError("reduce must be 'mean' or 'mag_mean'")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        if freq_index is not None:
            return arr[:, int(freq_index)]
        return np.nanmean(arr, axis=1)
    return arr.ravel()


def pol_channel_magnitude_mean(
    spec_source: object,
    attribute: str,
    freq_index: int | None = None,
) -> np.ndarray:
    """Frequency-mean of ``|channel|`` (one value per pointing)."""
    return _pol_channel_values_1d(
        spec_source, attribute, freq_index, reduce="mag_mean"
    )


def _angular_separation_deg_broadcast(
    ra_deg: float,
    dec_deg: float,
    ra_grid_deg: np.ndarray,
    dec_grid_deg: np.ndarray,
) -> np.ndarray:
    """Great-circle separation (deg) from one (ra, dec) to each grid point."""
    ra_s = float(_normalize_ra_deg(np.array([ra_deg]))[0])
    dec_s = float(dec_deg)
    ra_g = _normalize_ra_deg(np.asarray(ra_grid_deg, dtype=float))
    dec_g = np.asarray(dec_grid_deg, dtype=float)
    ra1 = np.radians(ra_s)
    dec1 = np.radians(dec_s)
    ra2 = np.radians(ra_g)
    dec2 = np.radians(dec_g)
    dra = ((ra2 - ra1 + np.pi) % (2.0 * np.pi)) - np.pi
    ddec = dec2 - dec1
    sin_dra = np.sin(0.5 * dra)
    sin_ddec = np.sin(0.5 * ddec)
    a = sin_ddec * sin_ddec + np.cos(dec1) * np.cos(dec2) * sin_dra * sin_dra
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _kernel_pad_pixels(trunc_deg: float, pixel_size_deg: float) -> int:
    """Pixels of margin so Gaussian gridding is not clipped at the map edge."""
    pix = float(pixel_size_deg)
    trunc = float(trunc_deg)
    if not np.isfinite(pix) or pix <= 0:
        raise ValueError("pixel_size_deg must be positive and finite")
    if not np.isfinite(trunc) or trunc < 0:
        raise ValueError("trunc_deg must be finite and >= 0")
    return int(np.ceil(trunc / pix)) + 1


def _grid_extent_ra_dec(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pixel_size_deg: float,
    padding_pixels: int,
    *,
    trunc_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    RA/Dec grid centers and edges covering all samples plus kernel margin.

    Padding is at least ``padding_pixels`` and, when ``trunc_deg`` is given,
    enough extra pixels that a sample on the bounding box can deposit its full
    Gaussian out to ``trunc_deg`` without hitting the grid boundary.
    """
    pix = float(pixel_size_deg)
    if not np.isfinite(pix) or pix <= 0:
        raise ValueError("pixel_size_deg must be positive and finite")
    pad_pix = int(max(0, padding_pixels))
    if trunc_deg is not None:
        pad_pix = max(pad_pix, _kernel_pad_pixels(float(trunc_deg), pix))
    pad_deg = pad_pix * pix + 0.5 * pix

    ra_n = _normalize_ra_deg(ra_deg)
    dec_arr = np.asarray(dec_deg, dtype=float)
    cos_dec = float(np.cos(np.radians(np.nanmean(dec_arr))))
    cos_dec = max(abs(cos_dec), 1e-6)
    ra_pad_deg = pad_deg / cos_dec
    ra_lo = float(np.nanmin(ra_n)) - ra_pad_deg
    ra_hi = float(np.nanmax(ra_n)) + ra_pad_deg
    dec_lo = float(np.nanmin(dec_arr)) - pad_deg
    dec_hi = float(np.nanmax(dec_arr)) + pad_deg

    n_ra = max(2, int(np.ceil((ra_hi - ra_lo) / pix)) + 1)
    n_dec = max(2, int(np.ceil((dec_hi - dec_lo) / pix)) + 1)
    ra_edges = ra_lo + np.arange(n_ra + 1, dtype=float) * pix
    dec_edges = dec_lo + np.arange(n_dec + 1, dtype=float) * pix
    while ra_edges[-1] < ra_hi:
        n_ra += 1
        ra_edges = ra_lo + np.arange(n_ra + 1, dtype=float) * pix
    while dec_edges[-1] < dec_hi:
        n_dec += 1
        dec_edges = dec_lo + np.arange(n_dec + 1, dtype=float) * pix
    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    return ra_centers, dec_centers, ra_edges, dec_edges, pad_pix


def _grid_extent_fixed_n_pix(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    n_pix: int,
    padding_pixels: int = 0,
    *,
    trunc_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """``n_pix`` × ``n_pix`` lattice spanning the pointing min/max only (no pad)."""
    n = int(n_pix)
    if n < 2:
        raise ValueError("n_pix must be >= 2")
    del trunc_deg  # extent is the observed bbox; do not grow it for a kernel
    pad_pix = int(max(0, padding_pixels))
    ra_n = _unwrap_ra_deg(ra_deg)
    dec_arr = np.asarray(dec_deg, dtype=float)
    valid = np.isfinite(ra_n) & np.isfinite(dec_arr)
    if not np.any(valid):
        raise ValueError("No finite RA/Dec samples to grid")
    ra_n = ra_n[valid]
    dec_arr = dec_arr[valid]
    ra_lo = float(np.nanmin(ra_n))
    ra_hi = float(np.nanmax(ra_n))
    dec_lo = float(np.nanmin(dec_arr))
    dec_hi = float(np.nanmax(dec_arr))
    if ra_hi <= ra_lo:
        ra_hi = ra_lo + 1e-12
    if dec_hi <= dec_lo:
        dec_hi = dec_lo + 1e-12
    if pad_pix > 0:
        pix_ra_g = (ra_hi - ra_lo) / float(n)
        pix_dec_g = (dec_hi - dec_lo) / float(n)
        ra_lo -= pad_pix * pix_ra_g
        ra_hi += pad_pix * pix_ra_g
        dec_lo -= pad_pix * pix_dec_g
        dec_hi += pad_pix * pix_dec_g
    ra_edges = np.linspace(ra_lo, ra_hi, n + 1, dtype=float)
    dec_edges = np.linspace(dec_lo, dec_hi, n + 1, dtype=float)
    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    pix_ra = float(ra_edges[1] - ra_edges[0])
    pix_dec = float(dec_edges[1] - dec_edges[0])
    return ra_centers, dec_centers, ra_edges, dec_edges, pad_pix, pix_ra, pix_dec


def make_beam_psf(
    beam_params: dict[str, Any],
    pixel_size_deg: float,
    *,
    trunc_sigma: float = 3.0,
    kernel_sigma_deg: float | None = None,
) -> dict[str, Any]:
    """
    Build a circular Gaussian PSF from fitted beam parameters.

    Follows the Astropy synthetic-image pattern
    (``Gaussian2DKernel`` with ``sigma_pix = sigma_sky / pixel_size``); see
    https://learn.astropy.org/tutorials/synthetic-images.html#prepare-a-point-spread-function-psf

    Parameters
    ----------
    beam_params : dict
        Output of ``fit_beam_gaussian`` (needs ``FWHM_deg`` or ``sigma_deg``).
    pixel_size_deg : float
        Map pixel spacing in degrees. Should be much smaller than the beam
        (typical default elsewhere is ``FWHM_deg / 4``).
    trunc_sigma : float
        Kernel extent in units of ``sigma`` (Astropy default support is ~4 sigma;
        gridding uses this truncation radius).
    kernel_sigma_deg : float or None
        Override PSF sigma in degrees; default from ``beam_params``.

    Returns
    -------
    dict
        ``psf_array`` (normalized 2D kernel), ``kernel`` (``Gaussian2DKernel``),
        ``sigma_deg``, ``sigma_pix``, ``fwhm_deg``, ``pixel_size_deg``,
        ``trunc_sigma``, ``trunc_deg``.
    """
    from astropy.convolution import Gaussian2DKernel

    pix = float(pixel_size_deg)
    if not np.isfinite(pix) or pix <= 0:
        raise ValueError("pixel_size_deg must be positive and finite")
    sigma = (
        float(kernel_sigma_deg)
        if kernel_sigma_deg is not None
        else _sigma_deg_from_beam_params(beam_params)
    )
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("kernel_sigma_deg must be positive and finite")
    trunc = float(trunc_sigma)
    if not np.isfinite(trunc) or trunc <= 0:
        raise ValueError("trunc_sigma must be positive and finite")

    fwhm_deg = _fwhm_deg_from_beam_params(beam_params)
    sigma_pix = sigma / pix
    # x_size/y_size odd so the kernel is centered on a pixel (Astropy convention).
    support = max(3, int(np.ceil(2.0 * trunc * sigma_pix)) | 1)
    kernel = Gaussian2DKernel(
        x_stddev=sigma_pix,
        y_stddev=sigma_pix,
        x_size=support,
        y_size=support,
    )
    psf_array = np.asarray(kernel.array, dtype=float)
    total = float(np.sum(psf_array))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Gaussian PSF normalization failed")
    psf_array = psf_array / total

    return {
        "psf_array": psf_array,
        "kernel": kernel,
        "sigma_deg": float(sigma),
        "sigma_pix": float(sigma_pix),
        "fwhm_deg": float(fwhm_deg),
        "pixel_size_deg": pix,
        "trunc_sigma": trunc,
        "trunc_deg": float(trunc * sigma),
    }


def estimate_pointing_spacings(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    n_probe: int = 2000,
) -> dict[str, float]:
    """
    Estimate along-scan and cross-scan pointing spacings (degrees).

    Median nearest-neighbor distance is dominated by the dense along-scan
    sampling. Cross-scan spacing is estimated from large successive steps
    (row changes) and from the nearest mostly-perpendicular neighbor.
    """
    from scipy.spatial import cKDTree

    ra = np.asarray(ra_deg, dtype=float).ravel()
    dec = np.asarray(dec_deg, dtype=float).ravel()
    ok = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[ok]
    dec = dec[ok]
    if ra.size < 3:
        raise ValueError("Need at least 3 finite pointings to estimate spacings")

    xy = np.column_stack([ra * np.cos(np.radians(dec)), dec])
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
    along_nn = float(np.nanmedian(nn))

    dra = np.diff(ra)
    ddec = np.diff(dec)
    step = np.hypot(dra * np.cos(np.radians(dec[:-1])), ddec)
    med_step = float(np.nanmedian(step))
    jumps = step[step > 5.0 * med_step] if med_step > 0 else step
    cross_from_jumps = float(np.nanmedian(jumps)) if jumps.size else np.nan

    along_vec = np.column_stack(
        [dra * np.cos(np.radians(dec[:-1])), ddec]
    )
    norms = np.linalg.norm(along_vec, axis=1, keepdims=True)
    along_vec = along_vec / np.maximum(norms, 1e-12)
    along_vec = np.vstack([along_vec[:1], along_vec])
    perp_vec = np.column_stack([-along_vec[:, 1], along_vec[:, 0]])

    tree = cKDTree(xy)
    n_probe = int(max(100, min(n_probe, ra.size)))
    probe = np.linspace(0, ra.size - 1, n_probe, dtype=int)
    cross_list: list[float] = []
    for i in probe:
        dists, ids = tree.query(xy[i], k=min(40, ra.size))
        rel = xy[ids[1:]] - xy[i]
        p = np.abs(rel @ perp_vec[i])
        a = np.abs(rel @ along_vec[i])
        sel = (p > 2.0 * a) & (p > 2.0 * med_step)
        if np.any(sel):
            cross_list.append(float(np.min(p[sel])))
    cross_from_perp = float(np.nanmedian(cross_list)) if cross_list else np.nan

    cross_candidates = [c for c in (cross_from_jumps, cross_from_perp) if np.isfinite(c)]
    cross = float(np.nanmedian(cross_candidates)) if cross_candidates else along_nn
    return {
        "along_scan_deg": along_nn,
        "cross_scan_deg": cross,
        "successive_step_deg": med_step,
        "cross_from_jumps_deg": cross_from_jumps,
        "cross_from_perp_deg": cross_from_perp,
    }


def stokes_iqu_from_pointing(
    beam_obs_pointing: object,
    *,
    freq_index: int | None = None,
    i_from: str = "auto",
    baseline_subtract: bool = False,
    baseline_percentile: float = 10.0,
) -> dict[str, np.ndarray]:
    """
    Dual-pol Stokes I, Q, U from calibrated ``AA_``, ``BB_``, ``AB_``.

    Default (classical):

        ``I = 0.5 (AA + BB)``, ``Q = 0.5 (AA - BB)``, ``U = Re(AB)``

    Parameters
    ----------
    i_from : {\"auto\", \"cross\"}
        ``auto`` (default): classical Stokes I from the autos.
        ``cross``: set ``I = Re(AB)`` and ``U = 0`` (optional; not the default).
    baseline_subtract : bool
        If True, subtract a robust low percentile per auto channel before
        forming Q (and auto-based I).
    baseline_percentile : float
        Percentile used as the per-channel baseline (default 10).
    """
    spec_source = get_pol_source(beam_obs_pointing, kind="mean")
    if spec_source is None:
        raise ValueError(
            "beam_obs_pointing must have calibrated_spec_mean or spec_mean"
        )
    for name in ("AA_", "BB_", "AB_"):
        if not hasattr(spec_source, name) or getattr(spec_source, name) is None:
            raise ValueError(f"Stokes IQU requires channel {name!r} on the spectrum")
    aa = np.asarray(_pol_channel_values_1d(spec_source, "AA_", freq_index), dtype=float)
    bb = np.asarray(_pol_channel_values_1d(spec_source, "BB_", freq_index), dtype=float)
    ab = np.asarray(_pol_channel_values_1d(spec_source, "AB_", freq_index), dtype=float)
    if np.iscomplexobj(ab):
        ab = np.real(ab)

    key = str(i_from).strip().lower()
    if key not in {"cross", "auto", "ab", "ab_"}:
        raise ValueError("i_from must be 'cross' or 'auto'")

    aa_use = aa
    bb_use = bb
    if baseline_subtract:
        p = float(baseline_percentile)
        aa_use = aa - np.nanpercentile(aa, p)
        bb_use = bb - np.nanpercentile(bb, p)

    if key in {"cross", "ab", "ab_"}:
        stokes_i = ab
        stokes_u = np.zeros_like(ab, dtype=float)
    else:
        stokes_i = 0.5 * (aa_use + bb_use)
        stokes_u = ab.copy()

    return {
        "I": np.asarray(stokes_i, dtype=float),
        "Q": 0.5 * (aa_use - bb_use),
        "U": np.asarray(stokes_u, dtype=float),
    }


def _bin_values_to_radec_grid(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    values: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    pixel_size_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor mean binning onto an RA/Dec pixel grid."""
    ra = _normalize_ra_deg(np.asarray(ra_deg, dtype=float))
    dec = np.asarray(dec_deg, dtype=float)
    vals = np.asarray(values, dtype=float)
    if not (ra.size == dec.size == vals.size):
        raise ValueError("ra, dec, and values must have the same length")
    pix = float(pixel_size_deg)
    n_dec, n_ra = len(dec_centers), len(ra_centers)
    numer = np.zeros((n_dec, n_ra), dtype=float)
    counts = np.zeros((n_dec, n_ra), dtype=float)
    for k in range(ra.size):
        v = float(vals[k])
        if not np.isfinite(v) or not np.isfinite(ra[k]) or not np.isfinite(dec[k]):
            continue
        j = int(np.round((float(ra[k]) - float(ra_centers[0])) / pix))
        i = int(np.round((float(dec[k]) - float(dec_centers[0])) / pix))
        if i < 0 or i >= n_dec or j < 0 or j >= n_ra:
            continue
        numer[i, j] += v
        counts[i, j] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        dirty = np.where(counts > 0, numer / counts, np.nan)
    return dirty, counts


def _bin_values_to_ra_dec_edges(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    values: np.ndarray,
    ra_edges: np.ndarray,
    dec_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of samples in each RA/Dec bin; pixels with no sample stay NaN."""
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    vals = np.asarray(values, dtype=float)
    ra_e = np.asarray(ra_edges, dtype=float)
    dec_e = np.asarray(dec_edges, dtype=float)
    n_ra = ra_e.size - 1
    n_dec = dec_e.size - 1
    numer = np.zeros((n_dec, n_ra), dtype=float)
    counts = np.zeros((n_dec, n_ra), dtype=float)
    inside = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(vals)
        & (ra >= ra_e[0])
        & (ra <= ra_e[-1])
        & (dec >= dec_e[0])
        & (dec <= dec_e[-1])
    )
    if not np.any(inside):
        return np.full((n_dec, n_ra), np.nan), counts
    j = np.digitize(ra[inside], ra_e) - 1
    i = np.digitize(dec[inside], dec_e) - 1
    j = np.clip(j, 0, n_ra - 1)
    i = np.clip(i, 0, n_dec - 1)
    v = vals[inside]
    np.add.at(numer, (i, j), v)
    np.add.at(counts, (i, j), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        gridded = np.where(counts > 0, numer / counts, np.nan)
    return gridded, counts


def _bin_then_linear_fill(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    values: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    ra_edges: np.ndarray,
    dec_edges: np.ndarray,
    max_dist_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean-bin samples onto the lattice, then linearly fill empty pixels
    that are still close to a hit pixel.

    Occupied pixels keep the in-bin mean (the peak is not kernel-averaged).
    Gaps between scan tracks are filled by interpolation, not a Gaussian
    whose width is a large fraction of the beam.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError, cKDTree

    z, counts = _bin_values_to_ra_dec_edges(
        ra_deg, dec_deg, values, ra_edges, dec_edges
    )
    n_dec, n_ra = z.shape
    hit = (counts > 0) & np.isfinite(z)
    if np.count_nonzero(hit) < 3:
        w = np.where(hit, counts, 0.0)
        n_hits = np.where(hit, counts, 0.0).astype(int)
        return z, w, n_hits

    ra0 = float(np.mean(ra_centers))
    dec0 = float(np.mean(dec_centers))
    cos_dec = max(abs(float(np.cos(np.radians(dec0)))), 1e-6)

    def _xy(r: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                (np.asarray(r, dtype=float) - ra0) * cos_dec,
                np.asarray(d, dtype=float) - dec0,
            ]
        )

    yy, xx = np.nonzero(hit)
    pts = _xy(ra_centers[xx], dec_centers[yy])
    ra_g, dec_g = np.meshgrid(ra_centers, dec_centers)
    query = _xy(ra_g.ravel(), dec_g.ravel())
    dist = cKDTree(pts).query(query, k=1)[0].reshape(n_dec, n_ra)
    need = (~np.isfinite(z)) & (dist <= float(max_dist_deg))
    if np.any(need):
        try:
            filled = LinearNDInterpolator(pts, z[hit], fill_value=np.nan)(query)
            filled = np.asarray(filled, dtype=float).reshape(n_dec, n_ra)
            z = z.copy()
            take = need & np.isfinite(filled)
            z[take] = filled[take]
        except (QhullError, ValueError):
            pass
    finite = np.isfinite(z)
    weight = np.where(finite, np.maximum(counts, 1.0), 0.0)
    n_hits = np.where(finite, np.maximum(counts, 1.0), 0.0).astype(int)
    return z, weight, n_hits


def _bin_then_linear_fill(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    values: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    ra_edges: np.ndarray,
    dec_edges: np.ndarray,
    max_dist_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean-bin samples onto the lattice, then linearly fill empty pixels
    that are still close to a hit pixel.

    Occupied pixels keep the in-bin mean (the peak is not kernel-averaged).
    Gaps between scan tracks are filled by interpolation, not a Gaussian
    whose width is a large fraction of the beam.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import QhullError, cKDTree

    z, counts = _bin_values_to_ra_dec_edges(ra_deg, dec_deg, values, ra_edges, dec_edges)
    n_dec, n_ra = z.shape
    hit = (counts > 0) & np.isfinite(z)
    if np.count_nonzero(hit) < 3:
        w = np.where(hit, counts, 0.0)
        n_hits = np.where(hit, counts, 0.0).astype(int)
        return z, w, n_hits

    ra0 = float(np.mean(ra_centers))
    dec0 = float(np.mean(dec_centers))
    cos_dec = max(abs(float(np.cos(np.radians(dec0)))), 1e-6)

    def _xy(r: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                (np.asarray(r, dtype=float) - ra0) * cos_dec,
                np.asarray(d, dtype=float) - dec0,
            ]
        )

    yy, xx = np.nonzero(hit)
    pts = _xy(ra_centers[xx], dec_centers[yy])
    ra_g, dec_g = np.meshgrid(ra_centers, dec_centers)
    query = _xy(ra_g.ravel(), dec_g.ravel())
    dist = cKDTree(pts).query(query, k=1)[0].reshape(n_dec, n_ra)
    need = (~np.isfinite(z)) & (dist <= float(max_dist_deg))
    if np.any(need):
        try:
            filled = LinearNDInterpolator(pts, z[hit], fill_value=np.nan)(query)
            filled = np.asarray(filled, dtype=float).reshape(n_dec, n_ra)
            z = z.copy()
            take = need & np.isfinite(filled)
            z[take] = filled[take]
        except (QhullError, ValueError):
            pass
    finite = np.isfinite(z)
    weight = np.where(finite, np.maximum(counts, 1.0), 0.0)
    n_hits = np.where(finite, np.maximum(counts, 1.0), 0.0).astype(int)
    return z, weight, n_hits


def _accumulate_psf_weighted(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    values: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    pixel_size_deg: float,
    sigma_deg: float,
    trunc_deg: float,
    *,
    wrap_ra: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PSF-weighted mean of samples onto an RA/Dec grid.

    Each sample is spread over pixels within ``trunc_deg`` with Gaussian weights
    ``w = exp(-r^2 / (2 sigma^2))`` (``r`` great-circle, degrees), so pixels
    without a sample of their own still receive flux from neighboring beams.

    Returns ``(gridded_mean, weight_sum, sample_count)``.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    vals = np.asarray(values, dtype=float)
    if not (ra.size == dec.size == vals.size):
        raise ValueError("ra, dec, and values must have the same length")
    n_dec, n_ra = len(dec_centers), len(ra_centers)
    pix_ra = float(ra_centers[1] - ra_centers[0]) if n_ra > 1 else float(pixel_size_deg)
    pix_dec = float(dec_centers[1] - dec_centers[0]) if n_dec > 1 else float(pixel_size_deg)
    if not np.isfinite(pix_ra) or pix_ra == 0:
        pix_ra = float(pixel_size_deg)
    if not np.isfinite(pix_dec) or pix_dec == 0:
        pix_dec = float(pixel_size_deg)
    numer = np.zeros((n_dec, n_ra), dtype=float)
    denom = np.zeros((n_dec, n_ra), dtype=float)
    n_hits = np.zeros((n_dec, n_ra), dtype=int)
    pad_pix_ra = int(np.ceil(float(trunc_deg) / max(abs(pix_ra), 1e-12))) + 1
    pad_pix_dec = int(np.ceil(float(trunc_deg) / max(abs(pix_dec), 1e-12))) + 1

    for k in range(ra.size):
        v_k = float(vals[k])
        if not np.isfinite(v_k):
            continue
        ra_k = float(ra[k])
        if wrap_ra:
            ra_k = float(_normalize_ra_deg(np.array([ra_k]))[0])
        dec_k = float(dec[k])
        if not np.isfinite(ra_k) or not np.isfinite(dec_k):
            continue
        j_c = int(np.clip(np.round((ra_k - ra_centers[0]) / pix_ra), 0, n_ra - 1))
        i_c = int(np.clip(np.round((dec_k - dec_centers[0]) / pix_dec), 0, n_dec - 1))
        j0 = max(0, j_c - pad_pix_ra)
        j1 = min(n_ra, j_c + pad_pix_ra + 1)
        i0 = max(0, i_c - pad_pix_dec)
        i1 = min(n_dec, i_c + pad_pix_dec + 1)
        if j0 >= j1 or i0 >= i1:
            continue
        RA, DEC = np.meshgrid(ra_centers[j0:j1], dec_centers[i0:i1])
        r = _angular_separation_deg_broadcast(ra_k, dec_k, RA, DEC)
        w = np.exp(-0.5 * (r / float(sigma_deg)) ** 2)
        w = np.where(r <= trunc_deg, w, 0.0)
        if not np.any(w > 0):
            continue
        numer[i0:i1, j0:j1] += v_k * w
        denom[i0:i1, j0:j1] += w
        n_hits[i0:i1, j0:j1] += (w > 0).astype(int)

    with np.errstate(invalid="ignore", divide="ignore"):
        gridded = np.where(denom > 0, numer / denom, np.nan)
    return gridded, denom, n_hits


def _interpolate_samples_to_grid(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    values: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    max_dist_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear interpolation of irregular samples onto an RA/Dec lattice.

    Pixels farther than ``max_dist_deg`` from every sample, or outside the
    Delaunay hull, stay NaN. Unlike a per-sample Gaussian, a finer lattice
    only samples the same interpolant more densely — no pointing-scale
    scallops at the coverage edge.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import QhullError, cKDTree

    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    vals = np.asarray(values, dtype=float)
    if not (ra.size == dec.size == vals.size):
        raise ValueError("ra, dec, and values must have the same length")
    n_dec, n_ra = len(dec_centers), len(ra_centers)
    empty = np.full((n_dec, n_ra), np.nan)
    zero_w = np.zeros((n_dec, n_ra), dtype=float)
    zero_n = np.zeros((n_dec, n_ra), dtype=int)
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(vals)
    ra = ra[valid]
    dec = dec[valid]
    vals = vals[valid]
    if ra.size == 0:
        return empty, zero_w, zero_n

    dec0 = float(np.mean(dec))
    ra0 = float(np.mean(ra))
    cos_dec = max(abs(float(np.cos(np.radians(dec0)))), 1e-6)

    def _xy(r: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.column_stack([(np.asarray(r, dtype=float) - ra0) * cos_dec,
                                np.asarray(d, dtype=float) - dec0])

    pts = _xy(ra, dec)
    ra_g, dec_g = np.meshgrid(ra_centers, dec_centers)
    query = _xy(ra_g.ravel(), dec_g.ravel())
    tree = cKDTree(pts)
    dist = tree.query(query, k=1)[0].reshape(n_dec, n_ra)
    near = dist <= float(max_dist_deg)

    if ra.size == 1:
        z = np.full((n_dec, n_ra), vals[0], dtype=float)
        z = np.where(near, z, np.nan)
    else:
        try:
            z = LinearNDInterpolator(pts, vals, fill_value=np.nan)(query)
            z = np.asarray(z, dtype=float).reshape(n_dec, n_ra)
        except (QhullError, ValueError):
            z = NearestNDInterpolator(pts, vals)(query)
            z = np.asarray(z, dtype=float).reshape(n_dec, n_ra)
        z = np.where(near, z, np.nan)

    weight = np.where(np.isfinite(z), 1.0, 0.0)
    n_hits = np.where(np.isfinite(z), 1, 0).astype(int)
    return z, weight, n_hits


def _drop_low_gridding_weight(
    gridded: np.ndarray,
    weight_sum: np.ndarray,
    *,
    frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide Gaussian-kernel fringes: keep pixels with substantial coverage."""
    z = np.asarray(gridded, dtype=float)
    w = np.asarray(weight_sum, dtype=float)
    hit = np.isfinite(w) & (w > 0) & np.isfinite(z)
    if not np.any(hit):
        return z, w
    thresh = float(frac) * float(np.median(w[hit]))
    keep = hit & (w >= thresh)
    out_z = np.where(keep, z, np.nan)
    out_w = np.where(keep, w, 0.0)
    return out_z, out_w


def _fill_nan_inside_sample_bbox(
    map2d: np.ndarray,
    weight_sum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear/nearest fill of NaNs inside the bounding box of gridded samples."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    z = np.asarray(map2d, dtype=np.float64)
    w = np.asarray(weight_sum, dtype=np.float64)
    valid = (w > 0) & np.isfinite(z)
    if np.count_nonzero(valid) < 3:
        return z, w
    yy, xx = np.nonzero(valid)
    pts = np.column_stack([yy.astype(float), xx.astype(float)])
    vals = z[valid]
    ny, nx = z.shape
    y0, y1 = int(yy.min()), int(yy.max())
    x0, x1 = int(xx.min()), int(xx.max())
    bbox = np.zeros(z.shape, dtype=bool)
    bbox[y0 : y1 + 1, x0 : x1 + 1] = True
    need = bbox & ~np.isfinite(z)
    if not np.any(need):
        return z, w
    gy, gx = np.nonzero(need)
    query = np.column_stack([gy.astype(float), gx.astype(float)])
    filled_need = LinearNDInterpolator(pts, vals, fill_value=np.nan)(query)
    still = ~np.isfinite(filled_need)
    if np.any(still):
        filled_need[still] = NearestNDInterpolator(pts, vals)(query[still])
    out = z.copy()
    out_w = w.copy()
    out[need] = filled_need
    interpolated = need & (w <= 0)
    if np.any(interpolated):
        med_w = float(np.nanmedian(w[valid])) if np.any(valid) else 1.0
        out_w[interpolated] = max(med_w, 1e-6)
    return out, out_w


def make_dirty_maps(
    beam_obs_pointing: object,
    beam_params: dict[str, Any],
    *,
    attributes: list[str] | None = None,
    stokes: bool = False,
    freq_index: int | None = None,
    pixel_size_deg: float | None = None,
    padding_pixels: int = 2,
    reference_grid: dict[str, Any] | None = None,
    method: str = "psf",
    kernel_sigma_deg: float | None = None,
    trunc_sigma: float = 3.0,
    i_from: str = "auto",
    baseline_subtract: bool = False,
) -> dict[str, Any]:
    """
    Grid matched pointings onto a fine RA/Dec map, per channel or Stokes.

    ``method='psf'`` (default) spreads each sample over nearby pixels with the
    beam Gaussian, matching ``map_pointings_with_psf``, so the map has no empty
    pixels inside the covered region. ``method='nearest'`` assigns each sample
    to its nearest pixel only; pixels with no sample are NaN, which leaves
    visible holes when the pixel size is much smaller than the sample spacing.

    If ``stokes=True``, maps ``I``, ``Q``, ``U`` from ``AA_``/``BB_``/``AB_``
    (see ``stokes_iqu_from_pointing``; default classical ``I = 0.5(AA+BB)``).
    Otherwise grid the requested ``attributes`` (default ``['AB_']``).
    Pass ``reference_grid`` (e.g. output of ``map_pointings_with_psf``) to reuse
    the same pixel geometry.
    """
    method_key = str(method).strip().lower()
    if method_key not in {"psf", "nearest"}:
        raise ValueError("method must be 'psf' or 'nearest'")

    fwhm_deg = _fwhm_deg_from_beam_params(beam_params)
    sigma = (
        float(kernel_sigma_deg)
        if kernel_sigma_deg is not None
        else _sigma_deg_from_beam_params(beam_params)
    )
    trunc_deg = float(trunc_sigma) * sigma

    if reference_grid is not None:
        ra_centers = np.asarray(reference_grid["ra_centers"], dtype=float)
        dec_centers = np.asarray(reference_grid["dec_centers"], dtype=float)
        ra_edges = np.asarray(reference_grid["ra_edges"], dtype=float)
        dec_edges = np.asarray(reference_grid["dec_edges"], dtype=float)
        pix = float(reference_grid.get("pixel_size_deg", pixel_size_deg or fwhm_deg / 4.0))
        pad_pix = int(reference_grid.get("padding_pixels", padding_pixels))
    else:
        pix = float(pixel_size_deg) if pixel_size_deg is not None else fwhm_deg / 4.0
        ra = np.atleast_1d(np.asarray(getattr(beam_obs_pointing, "ra"), dtype=float))
        dec = np.atleast_1d(np.asarray(getattr(beam_obs_pointing, "dec"), dtype=float))
        ra_centers, dec_centers, ra_edges, dec_edges, pad_pix = _grid_extent_ra_dec(
            ra,
            dec,
            pix,
            padding_pixels,
            trunc_deg=trunc_deg if method_key == "psf" else None,
        )

    ra = np.atleast_1d(np.asarray(getattr(beam_obs_pointing, "ra"), dtype=float))
    dec = np.atleast_1d(np.asarray(getattr(beam_obs_pointing, "dec"), dtype=float))

    value_map: dict[str, np.ndarray] = {}
    if stokes:
        value_map.update(
            stokes_iqu_from_pointing(
                beam_obs_pointing,
                freq_index=freq_index,
                i_from=i_from,
                baseline_subtract=baseline_subtract,
            )
        )
    else:
        attrs = list(attributes) if attributes is not None else ["AB_"]
        _, _, spec_source, resolved = _beam_obs_pointing_ra_dec_attrs(
            beam_obs_pointing, None, attrs
        )
        for attr in resolved:
            value_map[attr] = _pol_channel_values_1d(spec_source, attr, freq_index)

    maps: dict[str, np.ndarray] = {}
    sample_count: dict[str, np.ndarray] = {}
    weight_sum: dict[str, np.ndarray] = {}
    for name, vals in value_map.items():
        if method_key == "psf":
            gridded, denom, counts = _accumulate_psf_weighted(
                ra, dec, vals, ra_centers, dec_centers, pix, sigma, trunc_deg
            )
        else:
            gridded, counts = _bin_values_to_radec_grid(
                ra, dec, vals, ra_centers, dec_centers, pix
            )
            denom = counts
        maps[name] = gridded
        sample_count[name] = counts
        weight_sum[name] = denom

    return {
        "maps": maps,
        "sample_count": sample_count,
        "weight_sum": weight_sum,
        "ra_centers": ra_centers,
        "dec_centers": dec_centers,
        "ra_edges": ra_edges,
        "dec_edges": dec_edges,
        "pixel_size_deg": pix,
        "kernel_sigma_deg": float(sigma),
        "fwhm_deg": float(fwhm_deg),
        "trunc_deg": float(trunc_deg),
        "padding_pixels": int(pad_pix),
        "stokes": bool(stokes),
        "method": method_key,
    }


def convolve_map_with_psf(
    image: np.ndarray,
    psf: dict[str, Any] | np.ndarray,
    *,
    boundary: str = "wrap",
    fill_value: float | str = "interpolate",
    restore_nan_mask: bool = False,
) -> np.ndarray:
    """
    Convolve a 2D image with a Gaussian PSF via Astropy ``convolve_fft``.

    Same pattern as the Astropy synthetic-images tutorial
    (https://learn.astropy.org/tutorials/synthetic-images.html#convolve-image-with-psf).

    ``psf`` may be the dict from ``make_beam_psf`` or a 2D kernel array.
    ``fill_value='interpolate'`` (default) lets the kernel fill empty pixels from
    their neighbors, so the output has no holes; a float or ``'median'`` /
    ``'mean'`` / ``'zero'`` replaces empty pixels with a constant first. Set
    ``restore_nan_mask=True`` to put the original empty pixels back as NaN.
    """
    from astropy.convolution import convolve_fft

    img = np.asarray(image, dtype=float)
    if isinstance(psf, dict):
        kernel = psf.get("kernel", psf.get("psf_array"))
    else:
        kernel = psf
    if kernel is None:
        raise ValueError("psf must be make_beam_psf output or a 2D array/kernel")

    finite = np.isfinite(img)
    interpolate = False
    if isinstance(fill_value, str):
        key = fill_value.strip().lower()
        if key in {"interpolate", "nan", "neighbors"}:
            interpolate = True
            fill = 0.0
        elif key == "median":
            fill = float(np.nanmedian(img[finite])) if np.any(finite) else 0.0
        elif key == "mean":
            fill = float(np.nanmean(img[finite])) if np.any(finite) else 0.0
        elif key in {"zero", "0"}:
            fill = 0.0
        else:
            raise ValueError(
                "fill_value must be float or 'interpolate'/'median'/'mean'/'zero'"
            )
    else:
        fill = float(fill_value)

    if interpolate:
        work = np.where(finite, img, np.nan)
        nan_treatment = "interpolate"
    else:
        work = np.where(finite, img, fill)
        nan_treatment = "fill"

    convolved = convolve_fft(
        work,
        kernel,
        boundary=boundary,
        nan_treatment=nan_treatment,
        normalize_kernel=True,
        allow_huge=True,
    )
    out = np.asarray(convolved, dtype=float)
    if restore_nan_mask:
        out = np.where(finite, out, np.nan)
    return out


def convolve_dirty_stokes_with_psf(
    beam_obs_pointing: object,
    beam_params: dict[str, Any],
    *,
    pixel_size_deg: float | None = None,
    freq_index: int | None = None,
    padding_pixels: int = 2,
    reference_grid: dict[str, Any] | None = None,
    boundary: str = "wrap",
    fill_value: float | str = "interpolate",
    trunc_sigma: float = 3.0,
    method: str = "psf",
) -> dict[str, Any]:
    """
    Grid Stokes I/Q/U, then convolve each with the beam PSF.

    ``method='psf'`` (default) produces gap-free input maps; ``method='nearest'``
    reproduces the sparse nearest-pixel dirty image.

    Returns ``dirty`` (``make_dirty_maps`` result), ``convolved`` (channel → 2D),
    and ``psf`` (from ``make_beam_psf``).
    """
    dirty = make_dirty_maps(
        beam_obs_pointing,
        beam_params,
        stokes=True,
        freq_index=freq_index,
        pixel_size_deg=pixel_size_deg,
        padding_pixels=padding_pixels,
        reference_grid=reference_grid,
        method=method,
        trunc_sigma=trunc_sigma,
    )
    pix = float(dirty["pixel_size_deg"])
    psf = make_beam_psf(beam_params, pix, trunc_sigma=trunc_sigma)
    convolved: dict[str, np.ndarray] = {}
    for name, img in dirty["maps"].items():
        convolved[name] = convolve_map_with_psf(
            img,
            psf,
            boundary=boundary,
            fill_value=fill_value,
        )
    return {"dirty": dirty, "convolved": convolved, "psf": psf}


def deconvolve_map_with_psf(
    image: np.ndarray,
    weight_sum: np.ndarray,
    beam_params: dict[str, Any],
    *,
    pixel_size_deg: float,
    dec_centers: np.ndarray,
    method: str = "wiener",
    regularization_alpha: float = 1e-2,
    bl_floor: float = 1e-3,
    pad_value: float | str = "edge",
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = None,
    taper_fwhm_deg: float | None = None,
) -> dict[str, Any]:
    """
    Remove a Gaussian PSF from a 2D map via FFT deconvolution.

    Default ``method='wiener'`` builds the filter from the map's radially
    averaged Fourier power (no free ``alpha``). Alternatives:

    - ``regularized``: ``F = H* / (|H|^2 + alpha |H|_max^2)``
    - ``residual_match``: choose Tikhonov ``alpha`` so residual RMS ≈ map noise RMS

    Modes with ``|H|`` below ``bl_floor * |H|_max`` are suppressed as a safety cut.

    Returns ``deconvolved_map``, ``psf``, ``fill_value``, plus diagnostics.
    """
    sigma_deg = _sigma_deg_from_beam_params(beam_params)
    deconv, psf, fill_value, diagnostics = _fft_deconvolve_gaussian_2d(
        image,
        weight_sum,
        pixel_size_deg=float(pixel_size_deg),
        dec_centers=np.asarray(dec_centers, dtype=float),
        sigma_deg=sigma_deg,
        method=method,
        regularization_alpha=regularization_alpha,
        bl_floor=bl_floor,
        pad_value=pad_value,
        taper_width_deg=taper_width_deg,
        apodize_fwhm_deg=apodize_fwhm_deg,
        taper_fwhm_deg=taper_fwhm_deg,
    )
    out = {
        "deconvolved_map": deconv,
        "psf": psf,
        "fill_value": float(fill_value),
        "sigma_deg": float(sigma_deg),
        "method": str(method),
        "regularization_alpha": float(
            diagnostics.get("regularization_alpha", regularization_alpha)
        ),
        "diagnostics": diagnostics,
    }
    return out


def deconvolve_stokes_with_psf(
    beam_obs_pointing: object,
    beam_params: dict[str, Any],
    *,
    pixel_size_deg: float | None = None,
    freq_index: int | None = None,
    padding_pixels: int = 2,
    reference_grid: dict[str, Any] | None = None,
    grid_method: str = "psf",
    kernel_sigma_deg: float | None = None,
    trunc_sigma: float = 3.0,
    method: str = "wiener",
    regularization_alpha: float = 1e-2,
    bl_floor: float = 1e-3,
    pad_value: float | str = "edge",
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = 5.0,
    taper_fwhm_deg: float | None = None,
    i_from: str = "auto",
    baseline_subtract: bool = False,
) -> dict[str, Any]:
    """
    Grid Stokes I/Q/U, then remove the beam PSF from each via FFT deconvolution.

    Companion to ``convolve_dirty_stokes_with_psf``. Default ``grid_method='psf'``
    builds gap-free kernel-weighted maps. Set ``kernel_sigma_deg`` independently
    of the physical beam to limit additional smoothing during gridding.

    Default ``method='wiener'`` estimates the filter from each map's Fourier power
    (see ``deconvolve_map_with_psf``). Stokes I defaults to classical
    ``0.5(AA+BB)`` (``i_from='auto'``).

    Returns ``grid`` (``make_dirty_maps`` result), ``deconvolved`` (channel → 2D),
    ``psf`` (normalized 2D array used in the FFT), and filter metadata.
    """
    grid = make_dirty_maps(
        beam_obs_pointing,
        beam_params,
        stokes=True,
        freq_index=freq_index,
        pixel_size_deg=pixel_size_deg,
        padding_pixels=padding_pixels,
        reference_grid=reference_grid,
        method=grid_method,
        kernel_sigma_deg=kernel_sigma_deg,
        trunc_sigma=trunc_sigma,
        i_from=i_from,
        baseline_subtract=baseline_subtract,
    )
    deconvolved: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    psf_out = None
    fill_value = 0.0
    alpha_out = float(regularization_alpha)
    for name, img in grid["maps"].items():
        weight = grid["weight_sum"][name]
        res = deconvolve_map_with_psf(
            img,
            weight,
            beam_params,
            pixel_size_deg=float(grid["pixel_size_deg"]),
            dec_centers=grid["dec_centers"],
            method=method,
            regularization_alpha=regularization_alpha,
            bl_floor=bl_floor,
            pad_value=pad_value,
            taper_width_deg=taper_width_deg,
            apodize_fwhm_deg=apodize_fwhm_deg,
            taper_fwhm_deg=taper_fwhm_deg,
        )
        deconvolved[name] = res["deconvolved_map"]
        diagnostics[name] = res.get("diagnostics", {})
        psf_out = res["psf"]
        fill_value = res["fill_value"]
        alpha_out = float(res["regularization_alpha"])

    return {
        "grid": grid,
        "deconvolved": deconvolved,
        "psf": psf_out,
        "fill_value": float(fill_value),
        "method": str(method),
        "regularization_alpha": alpha_out,
        "bl_floor": float(bl_floor),
        "psf_fwhm_deg": float(_fwhm_deg_from_beam_params(beam_params)),
        "diagnostics": diagnostics,
    }


def map_pointings_with_psf(
    beam_obs_pointing: object,
    beam_params: dict[str, Any],
    *,
    attribute: str | None = None,
    attributes: list[str] | None = None,
    freq_index: int | None = None,
    pixel_size_deg: float | None = None,
    kernel_sigma_deg: float | None = None,
    trunc_sigma: float = 3.0,
    padding_pixels: int = 2,
) -> dict[str, Any]:
    """
    Fine-grid map: PSF-weighted average of nearby beam pointings per pixel.

    Pixel size defaults to ``FWHM_deg / 4`` so many beams contribute to each
    pixel. For each grid cell, samples within ``trunc_sigma * sigma`` are
    combined as:

        ``map[i,j] = sum_k v_k w(r_k) / sum_k w(r_k)``,
        ``w(r) = exp(-r^2 / (2 sigma^2))``

    with great-circle separation ``r`` (degrees). Step 1 builds the PSF via
    ``make_beam_psf``; step 2 builds the RA/Dec grid and accumulates weights.

    Parameters
    ----------
    beam_obs_pointing
        Output of ``match_data_and_pointing`` with ``ra``, ``dec``, and
        ``calibrated_spec_mean`` or ``spec_mean``.
    beam_params : dict
        Fitted beam parameters; must include ``FWHM_deg`` or ``sigma_deg``.
    attribute, attributes, freq_index
        Same channel selection as ``convolve_beam_with_fit``.
    pixel_size_deg : float or None
        Grid spacing in degrees; default ``FWHM_deg / 4``.
    kernel_sigma_deg : float or None
        Gaussian PSF sigma in degrees; default from ``beam_params``.
    trunc_sigma : float
        Truncate PSF contributions beyond this many sigma.
    padding_pixels : int
        Minimum extra pixels around the data bounding box. The grid is always
        expanded by at least ``ceil(trunc_deg / pixel_size) + 1`` pixels so
        edge samples are not clipped by the map boundary.

    Returns
    -------
    dict
        ``maps`` (channel → 2D array), ``weight_sum``, ``sample_count``,
        ``ra_centers``, ``dec_centers``, ``ra_edges``, ``dec_edges``,
        ``pixel_size_deg``, ``kernel_sigma_deg``, ``fwhm_deg``, ``trunc_deg``,
        ``padding_pixels`` (effective margin used), ``psf`` (from
        ``make_beam_psf``).
    """
    fwhm_deg = _fwhm_deg_from_beam_params(beam_params)
    pix = float(pixel_size_deg) if pixel_size_deg is not None else fwhm_deg / 4.0
    psf = make_beam_psf(
        beam_params,
        pix,
        trunc_sigma=trunc_sigma,
        kernel_sigma_deg=kernel_sigma_deg,
    )
    sigma = float(psf["sigma_deg"])
    trunc_deg = float(psf["trunc_deg"])

    ra, dec, spec_source, attrs = _beam_obs_pointing_ra_dec_attrs(
        beam_obs_pointing, attribute, attributes
    )
    if ra.size == 0:
        empty: dict[str, np.ndarray] = {}
        return {
            "maps": empty,
            "weight_sum": empty,
            "sample_count": empty,
            "ra_centers": np.array([]),
            "dec_centers": np.array([]),
            "ra_edges": np.array([]),
            "dec_edges": np.array([]),
            "pixel_size_deg": pix,
            "kernel_sigma_deg": sigma,
            "fwhm_deg": float(fwhm_deg),
            "trunc_deg": trunc_deg,
            "padding_pixels": int(max(0, padding_pixels)),
            "psf": psf,
        }

    ra_centers, dec_centers, ra_edges, dec_edges, pad_pix = _grid_extent_ra_dec(
        ra, dec, pix, padding_pixels, trunc_deg=trunc_deg
    )

    maps: dict[str, np.ndarray] = {}
    weight_sum: dict[str, np.ndarray] = {}
    sample_count: dict[str, np.ndarray] = {}

    for attr in attrs:
        vals = _pol_channel_values_1d(spec_source, attr, freq_index)
        if vals.size != ra.size:
            raise ValueError(
                f"Length mismatch: ra/dec length {ra.size}, {attr!r} length {vals.size}"
            )
        gridded, denom, n_hits = _accumulate_psf_weighted(
            ra, dec, vals, ra_centers, dec_centers, pix, sigma, trunc_deg
        )
        maps[attr] = gridded
        weight_sum[attr] = denom
        sample_count[attr] = n_hits

    return {
        "maps": maps,
        "weight_sum": weight_sum,
        "sample_count": sample_count,
        "ra_centers": ra_centers,
        "dec_centers": dec_centers,
        "ra_edges": ra_edges,
        "dec_edges": dec_edges,
        "pixel_size_deg": pix,
        "kernel_sigma_deg": sigma,
        "fwhm_deg": float(fwhm_deg),
        "trunc_deg": trunc_deg,
        "padding_pixels": int(pad_pix),
        "psf": psf,
    }


def grid_beam_obs_pointing(
    beam_obs_pointing: object,
    beam_params: dict[str, Any],
    *,
    attribute: str | None = None,
    attributes: list[str] | None = None,
    freq_index: int | None = None,
    pixel_size_deg: float | None = None,
    kernel_sigma_deg: float | None = None,
    trunc_sigma: float = 3.0,
    padding_pixels: int = 2,
) -> dict[str, Any]:
    """
    Beam-weighted convolutional gridding of matched pointing + spectrum data.

    Thin wrapper around ``map_pointings_with_psf`` (explicit PSF via
    ``make_beam_psf``, then PSF-weighted average on a fine RA/Dec grid).

    See ``map_pointings_with_psf`` for full parameter documentation.
    """
    return map_pointings_with_psf(
        beam_obs_pointing,
        beam_params,
        attribute=attribute,
        attributes=attributes,
        freq_index=freq_index,
        pixel_size_deg=pixel_size_deg,
        kernel_sigma_deg=kernel_sigma_deg,
        trunc_sigma=trunc_sigma,
        padding_pixels=padding_pixels,
    )


def grid_patch_fixed_shape(
    beam_obs_pointing: object,
    beam_params: dict[str, Any],
    *,
    attribute: str = "AB_",
    n_pix: int = 1024,
    freq_index: int | None = None,
    reduce: str = "mag_mean",
    kernel_sigma_deg: float | None = None,
    trunc_sigma: float = 3.0,
    padding_pixels: int = 0,
    method: str = "gaussian",
) -> dict[str, Any]:
    """
    Grid one pol channel onto an ``n_pix`` × ``n_pix`` RA/Dec patch.

    ``method='gaussian'`` (default): convolutional fill with σ from the
    cross-track spacing, capped at 0.3×FWHM. ``method='bin_linear'``:
    mean-bin each sample into one pixel, then linearly interpolate empty
    pixels still within the cross-track spacing (peak is the in-bin mean).
    """
    ra, dec, spec_source, attrs = _beam_obs_pointing_ra_dec_attrs(
        beam_obs_pointing, attribute, None
    )
    attr = attrs[0]
    vals = _pol_channel_values_1d(
        spec_source, attr, freq_index, reduce=reduce
    )
    if vals.size != ra.size:
        raise ValueError(
            f"Length mismatch: ra/dec length {ra.size}, {attr!r} length {vals.size}"
        )
    ra_u = _unwrap_ra_deg(ra)
    ra_centers, dec_centers, ra_edges, dec_edges, pad_pix, pix_ra, pix_dec = (
        _grid_extent_fixed_n_pix(
            ra_u,
            dec,
            int(n_pix),
            padding_pixels,
        )
    )
    pix = 0.5 * (pix_ra + pix_dec)
    fwhm_deg = _fwhm_deg_from_beam_params(beam_params)
    spacing = _characteristic_sample_spacing_deg(ra_u, dec)
    method_key = str(method).strip().lower()
    if method_key in {"bin_linear", "linear", "interp"}:
        if kernel_sigma_deg is not None:
            max_dist = float(kernel_sigma_deg)
        else:
            max_dist = max(spacing, 1.5 * pix)
        if not np.isfinite(max_dist) or max_dist <= 0:
            max_dist = max(1.5 * pix, 1e-6)
        gridded, denom, n_hits = _bin_then_linear_fill(
            ra_u,
            dec,
            vals,
            ra_centers,
            dec_centers,
            ra_edges,
            dec_edges,
            max_dist,
        )
        sigma_fill = max_dist
        trunc_deg = max_dist
    elif method_key in {"gaussian", "psf"}:
        if kernel_sigma_deg is not None:
            sigma_fill = float(kernel_sigma_deg)
        else:
            sigma_fill = max(spacing, 1.5 * pix)
            sigma_fill = min(sigma_fill, 0.30 * fwhm_deg)
        if not np.isfinite(sigma_fill) or sigma_fill <= 0:
            sigma_fill = max(1.5 * pix, 1e-6)
        trunc_deg = float(trunc_sigma) * sigma_fill
        gridded, denom, n_hits = _accumulate_psf_weighted(
            ra_u,
            dec,
            vals,
            ra_centers,
            dec_centers,
            pix,
            sigma_fill,
            trunc_deg,
            wrap_ra=False,
        )
        gridded, denom = _drop_low_gridding_weight(gridded, denom, frac=0.25)
        n_hits = np.where(denom > 0, n_hits, 0)
    else:
        raise ValueError("method must be 'bin_linear' or 'gaussian'")
    return {
        "maps": {attr: gridded},
        "weight_sum": {attr: denom},
        "sample_count": {attr: n_hits},
        "ra_centers": ra_centers,
        "dec_centers": dec_centers,
        "ra_edges": ra_edges,
        "dec_edges": dec_edges,
        "pixel_size_deg": float(pix),
        "pixel_size_ra_deg": float(pix_ra),
        "pixel_size_dec_deg": float(pix_dec),
        "kernel_sigma_deg": float(sigma_fill),
        "fwhm_deg": float(fwhm_deg),
        "trunc_deg": float(trunc_deg),
        "padding_pixels": int(pad_pix),
        "n_pix": int(n_pix),
        "reduce": str(reduce),
        "attribute": attr,
        "method": method_key,
    }


def map_patch_remove_gaussian_beam(
    beam_obs_pointing: object,
    beam_params: dict[str, Any],
    *,
    attribute: str = "AB_",
    n_pix: int = 1024,
    freq_index: int | None = None,
    reduce: str = "mag_mean",
    kernel_sigma_deg: float | None = None,
    trunc_sigma: float = 3.0,
    padding_pixels: int = 0,
    niter: int = 1,
    baseline: float | None = None,
) -> dict[str, Any]:
    """Grid the observed RA/Dec bbox, then subtract Gaussian neighbor leakage."""
    grid = grid_patch_fixed_shape(
        beam_obs_pointing,
        beam_params,
        attribute=attribute,
        n_pix=n_pix,
        freq_index=freq_index,
        reduce=reduce,
        kernel_sigma_deg=kernel_sigma_deg,
        trunc_sigma=trunc_sigma,
        padding_pixels=padding_pixels,
    )
    attr = grid["attribute"]
    sub = subtract_gaussian_psf_contributions(
        grid["maps"][attr],
        grid["weight_sum"][attr],
        beam_params,
        pixel_size_deg=float(grid["pixel_size_deg"]),
        dec_centers=grid["dec_centers"],
        niter=niter,
        baseline=baseline,
        pixel_size_ra_deg=float(grid["pixel_size_ra_deg"]),
        pixel_size_dec_deg=float(grid["pixel_size_dec_deg"]),
    )
    out = dict(grid)
    out.update(sub)
    out["grid"] = grid
    return out


def _gridded_scalar_fallback(
    observed: np.ndarray,
    weight_sum: np.ndarray,
    pad_value: float | str,
) -> float:
    """Scalar fallback only where edge extrapolation is undefined."""
    valid = (weight_sum > 0) & np.isfinite(observed)
    if isinstance(pad_value, str):
        key = pad_value.strip().lower()
        if key in {"edge", "nearest", "boundary"}:
            return float(np.nanmedian(observed[valid])) if np.any(valid) else 0.0
        if key == "median":
            return float(np.nanmedian(observed[valid])) if np.any(valid) else 0.0
        if key == "mean":
            return float(np.nanmean(observed[valid])) if np.any(valid) else 0.0
        if key in {"zero", "0"}:
            return 0.0
        raise ValueError(
            "pad_value must be a float or one of {'edge','median','mean','zero'}"
        )
    return float(pad_value)


def _gridded_edge_extrapolated_map(
    observed: np.ndarray,
    weight_sum: np.ndarray,
    pad_value: float | str,
) -> np.ndarray:
    """
    Per-pixel extrapolation from the nearest observed (coverage) pixel.

    Each grid cell gets the spectrum value at the closest point with data, so
    padding follows the local edge brightness instead of a single map constant.
    """
    from scipy.ndimage import distance_transform_edt

    obs = np.asarray(observed, dtype=np.float64)
    hits = np.asarray(weight_sum, dtype=np.float64) > 0
    valid = hits & np.isfinite(obs)
    fallback = _gridded_scalar_fallback(obs, weight_sum, pad_value)
    edge = np.full(obs.shape, fallback, dtype=np.float64)
    if not np.any(valid):
        return edge
    work = np.where(valid, obs, 0.0)
    _, inds = distance_transform_edt(~valid, return_indices=True)
    edge = work[inds[0], inds[1]].astype(np.float64)
    edge = np.where(np.isfinite(edge), edge, fallback)
    return edge


def _gaussian_psf_2d_on_grid(
    shape: tuple[int, int],
    pixel_size_deg: float,
    dec_centers: np.ndarray,
    sigma_deg: float,
    *,
    normalize: str = "sum",
    pixel_size_ra_deg: float | None = None,
    pixel_size_dec_deg: float | None = None,
) -> np.ndarray:
    """
    Circular Gaussian PSF embedded in a 2D array for FFT convolution.

    RA pixel spacing is scaled by ``cos(mean_dec)`` so the kernel is round on the sky.
    ``normalize='sum'`` (default) is a local average (same units as the map).
    ``normalize='peak'`` sets ``G(0)=1``.
    """
    ny, nx = int(shape[0]), int(shape[1])
    sigma = float(sigma_deg)
    pix = float(pixel_size_deg)
    pix_ra = float(pixel_size_ra_deg) if pixel_size_ra_deg is not None else pix
    pix_dec = float(pixel_size_dec_deg) if pixel_size_dec_deg is not None else pix
    if ny < 1 or nx < 1:
        raise ValueError("PSF grid shape must be positive")
    dec_mean = float(np.nanmean(dec_centers))
    cos_dec = max(abs(np.cos(np.radians(dec_mean))), 1e-6)
    cy, cx = ny // 2, nx // 2
    y = (np.arange(ny, dtype=float) - cy) * pix_dec
    x = (np.arange(nx, dtype=float) - cx) * pix_ra
    Y, X = np.meshgrid(y, x, indexing="ij")
    r_deg = np.sqrt(Y * Y + (X / cos_dec) ** 2)
    psf = np.exp(-0.5 * (r_deg / sigma) ** 2)
    key = str(normalize).strip().lower()
    if key == "sum":
        total = float(np.sum(psf))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Gaussian PSF normalization failed")
        return psf / total
    if key == "peak":
        peak = float(psf[cy, cx])
        if not np.isfinite(peak) or peak <= 0:
            raise ValueError("Gaussian PSF peak normalization failed")
        return psf / peak
    raise ValueError("normalize must be 'sum' or 'peak'")


def _fft_convolve_centered(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``image`` with a kernel whose origin is at the array center."""
    img = np.asarray(image, dtype=np.float64)
    ker = np.asarray(kernel, dtype=np.float64)
    if img.shape != ker.shape:
        raise ValueError("image and kernel must have the same shape")
    H = np.fft.fft2(np.fft.fftshift(ker))
    O = np.fft.fft2(np.nan_to_num(img, nan=0.0))
    return np.real(np.fft.ifft2(O * H))


def subtract_gaussian_psf_contributions(
    observed: np.ndarray,
    weight_sum: np.ndarray,
    beam_params: dict[str, Any],
    *,
    pixel_size_deg: float,
    dec_centers: np.ndarray,
    niter: int = 1,
    baseline: float | None = None,
    pixel_size_ra_deg: float | None = None,
    pixel_size_dec_deg: float | None = None,
) -> dict[str, Any]:
    """
    Keep pixel ``i`` and subtract Gaussian leakage from every other pixel.

    ``G`` is the fitted circular Gaussian (peak 1 for the beam shape). Mixing
    uses the same kernel **sum-normalized** so ``G ⊛ I`` stays in map units:

        ``S = 2 I - G_sum ⊛ I``

    on the baseline-subtracted map (pixel ``i`` is added back after the mix).
    Unobserved pixels stay NaN and are treated as the baseline during the FFT.
    """
    obs = np.asarray(observed, dtype=np.float64)
    hits = np.asarray(weight_sum, dtype=np.float64)
    if obs.shape != hits.shape:
        raise ValueError("observed and weight_sum must have the same 2D shape")
    n_loop = int(niter)
    if n_loop < 1:
        raise ValueError("niter must be >= 1")

    if baseline is None:
        if "baseline" in beam_params:
            baseline_val = float(beam_params["baseline"])
        elif "baseline_k" in beam_params:
            baseline_val = float(beam_params["baseline_k"])
        else:
            valid = (hits > 0) & np.isfinite(obs)
            baseline_val = float(np.nanmedian(obs[valid])) if np.any(valid) else 0.0
    else:
        baseline_val = float(baseline)
    if not np.isfinite(baseline_val):
        baseline_val = 0.0

    sigma_deg = _sigma_deg_from_beam_params(beam_params)
    g_peak = _gaussian_psf_2d_on_grid(
        obs.shape,
        float(pixel_size_deg),
        np.asarray(dec_centers, dtype=float),
        sigma_deg,
        normalize="peak",
        pixel_size_ra_deg=pixel_size_ra_deg,
        pixel_size_dec_deg=pixel_size_dec_deg,
    )
    g00_peak = float(g_peak[g_peak.shape[0] // 2, g_peak.shape[1] // 2])
    g_sum = g_peak / float(np.sum(g_peak))

    mask = (hits > 0) & np.isfinite(obs)
    excess = np.where(mask, obs - baseline_val, 0.0)
    sky = excess.copy()
    conv = np.zeros_like(sky)
    for _ in range(n_loop):
        conv = _fft_convolve_centered(sky, g_sum)
        sky = sky + (excess - conv)
        sky = np.where(mask, sky, 0.0)
    conv = _fft_convolve_centered(sky, g_sum)

    corrected = np.full(obs.shape, np.nan, dtype=np.float64)
    corrected[mask] = sky[mask] + baseline_val
    reconvolved = np.full(obs.shape, np.nan, dtype=np.float64)
    reconvolved[mask] = conv[mask] + baseline_val
    residual = np.full(obs.shape, np.nan, dtype=np.float64)
    residual[mask] = obs[mask] - reconvolved[mask]

    return {
        "corrected_map": corrected,
        "observed_map": obs,
        "psf": g_peak,
        "baseline": baseline_val,
        "niter": n_loop,
        "sigma_deg": float(sigma_deg),
        "g00": g00_peak,
        "reconvolved_map": reconvolved,
        "residual_map": residual,
    }


def _gridded_taper_weight(
    weight_sum: np.ndarray,
    pixel_size_deg: float,
    *,
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = None,
    taper_fwhm_deg: float | None = None,
    taper_kind: str = "cosine",
) -> np.ndarray:
    """
    Taper weight in [0, 1] for blending toward edge-extrapolated values at patch edges.

    If ``taper_width_deg`` is set, weight rises from 0 at the coverage boundary
    to 1 over that many degrees inward (distance transform on the grid).
    Otherwise falls back to normalized ``weight_sum`` (then optional Gaussian
    smoothing via ``apodize_fwhm_deg`` / ``taper_fwhm_deg``).
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    hits = np.asarray(weight_sum, dtype=np.float64)
    observed = hits > 0
    w = np.zeros_like(hits, dtype=np.float64)
    if not np.any(observed):
        return w

    pix = float(pixel_size_deg)
    if taper_width_deg is not None:
        tw = float(taper_width_deg)
        if not np.isfinite(tw) or tw <= 0:
            raise ValueError("taper_width_deg must be positive and finite when provided")
        dist_pix = distance_transform_edt(observed.astype(np.float64))
        dist_deg = dist_pix * pix
        u = np.clip(dist_deg / tw, 0.0, 1.0)
        kind = str(taper_kind).strip().lower()
        if kind == "cosine":
            w = np.where(observed, 0.5 * (1.0 - np.cos(np.pi * u)), 0.0)
        elif kind == "linear":
            w = np.where(observed, u, 0.0)
        else:
            raise ValueError("taper_kind must be 'cosine' or 'linear'")
    else:
        wmax = float(np.nanmax(hits))
        w = np.clip(hits / wmax, 0.0, 1.0) if wmax > 0 else observed.astype(np.float64)

    sqrt_2_ln2 = np.sqrt(2.0 * np.log(2.0))
    if apodize_fwhm_deg is not None:
        sig_pix = float(apodize_fwhm_deg) / (sqrt_2_ln2 * pix)
        if sig_pix > 0:
            w = gaussian_filter(w, sigma=sig_pix, mode="nearest")
            w = np.clip(w / max(float(np.nanmax(w)), 1e-30), 0.0, 1.0)

    if taper_fwhm_deg is not None:
        sig_pix = float(taper_fwhm_deg) / (sqrt_2_ln2 * pix)
        if sig_pix > 0:
            w = gaussian_filter(w, sigma=sig_pix, mode="nearest")
            w = np.clip(w / max(float(np.nanmax(w)), 1e-30), 0.0, 1.0)

    return np.clip(w, 0.0, 1.0)


def _apodize_gridded_map(
    observed: np.ndarray,
    weight_sum: np.ndarray,
    pad_value: float | str,
    *,
    pixel_size_deg: float,
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = None,
    taper_fwhm_deg: float | None = None,
    taper_kind: str = "cosine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Taper toward per-pixel edge extrapolation (nearest observed spectrum).

    ``padded = edge_target * (1 - w) + observed * w`` on the coverage mask.

    Returns ``(padded_map, taper_weight, edge_target, scalar_fallback)``.
    """
    obs = np.asarray(observed, dtype=np.float64)
    hits = np.asarray(weight_sum, dtype=np.float64)
    valid = (hits > 0) & np.isfinite(obs)
    fallback = _gridded_scalar_fallback(obs, hits, pad_value)
    edge_target = _gridded_edge_extrapolated_map(obs, hits, pad_value)
    w = _gridded_taper_weight(
        hits,
        pixel_size_deg,
        taper_width_deg=taper_width_deg,
        apodize_fwhm_deg=apodize_fwhm_deg,
        taper_fwhm_deg=taper_fwhm_deg,
        taper_kind=taper_kind,
    )
    if not np.any(valid):
        m = edge_target.copy()
        m[~np.isfinite(m)] = fallback
        return m, w, edge_target, fallback
    obs_safe = np.where(valid, obs, edge_target)
    padded = edge_target * (1.0 - w) + obs_safe * w
    padded[~np.isfinite(padded)] = edge_target[~np.isfinite(padded)]
    return padded, w, edge_target, fallback


def _fft_radial_mean(power: np.ndarray, n_bins: int = 48) -> np.ndarray:
    """Azimuthally average a 2D FFT-plane array onto the same grid."""
    p = np.asarray(power, dtype=np.float64)
    ny, nx = p.shape
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    kx2d, ky2d = np.meshgrid(kx, ky)
    k = np.sqrt(kx2d * kx2d + ky2d * ky2d)
    kmax = float(np.max(k)) if k.size else 1.0
    if kmax <= 0:
        return np.full(p.shape, float(np.nanmean(p)), dtype=np.float64)
    n_bins = int(max(8, min(n_bins, p.size // 4)))
    edges = np.linspace(0.0, kmax, n_bins + 1)
    idx = np.clip(np.digitize(k.ravel(), edges) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=p.ravel(), minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    means = np.divide(
        sums,
        np.maximum(counts, 1),
        dtype=np.float64,
    )
    # Fill empty bins from neighbors, then broadcast back to 2D.
    valid = counts > 0
    if not np.any(valid):
        return np.full(p.shape, float(np.nanmean(p)), dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means[~valid] = np.interp(
        centers[~valid],
        centers[valid],
        means[valid],
    )
    return means[idx].reshape(p.shape)


def _estimate_map_noise_rms(observed: np.ndarray, weight_sum: np.ndarray) -> float:
    """Robust noise RMS from faint / low-weight covered pixels."""
    obs = np.asarray(observed, dtype=np.float64)
    hits = np.asarray(weight_sum, dtype=np.float64)
    covered = (hits > 0) & np.isfinite(obs)
    if not np.any(covered):
        return 0.0
    vals = obs[covered]
    w = hits[covered]
    faint = vals <= np.nanpercentile(vals, 25.0)
    low_w = w <= np.nanpercentile(w, 30.0)
    sample = vals[faint | low_w]
    if sample.size < 8:
        sample = vals
    med = float(np.nanmedian(sample))
    mad = float(np.nanmedian(np.abs(sample - med)))
    return float(1.4826 * mad) if np.isfinite(mad) else 0.0


def _fft_deconvolve_gaussian_2d(
    observed: np.ndarray,
    weight_sum: np.ndarray,
    *,
    pixel_size_deg: float,
    dec_centers: np.ndarray,
    sigma_deg: float,
    method: str = "regularized",
    regularization_alpha: float = 1e-2,
    bl_floor: float = 1e-3,
    pad_value: float | str = "edge",
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = None,
    taper_fwhm_deg: float | None = None,
    taper_kind: str = "cosine",
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """
    Image-plane FFT deconvolution with a 2D Gaussian PSF (circular on the sky).

    Methods
    -------
    regularized
        ``F = H* / (|H|^2 + alpha |H|_max^2)`` with free ``regularization_alpha``.
    wiener / weiner
        Data-driven Wiener filter using radially averaged FFT power of the map:
        ``F = H* P_S / (|H|^2 P_S + P_N)``, with ``P_N`` from beam-suppressed modes
        and ``P_S = max(P_O - P_N, 0) / max(|H|^2, eps)``.
    residual_match
        Choose Tikhonov ``alpha`` so residual RMS after reconvolution matches the
        robust noise RMS estimated from faint/low-weight pixels.

    Returns ``(deconvolved_map, psf, fill_value, diagnostics)``.
    """
    obs = np.asarray(observed, dtype=np.float64)
    hits = np.asarray(weight_sum, dtype=np.float64)
    if obs.shape != hits.shape:
        raise ValueError("observed and weight_sum must have the same 2D shape")
    work, _, _, fill_value = _apodize_gridded_map(
        obs,
        hits,
        pad_value,
        pixel_size_deg=pixel_size_deg,
        taper_width_deg=taper_width_deg,
        apodize_fwhm_deg=apodize_fwhm_deg,
        taper_fwhm_deg=taper_fwhm_deg,
        taper_kind=taper_kind,
    )

    psf = _gaussian_psf_2d_on_grid(
        work.shape, pixel_size_deg, dec_centers, sigma_deg
    )
    H = np.fft.fft2(np.fft.fftshift(psf))
    O = np.fft.fft2(work)
    H_abs = np.abs(H)
    H_max = float(np.max(H_abs)) if H_abs.size else 0.0
    if H_max <= 0:
        raise ValueError("PSF FFT has zero peak response")
    thresh = float(bl_floor) * H_max
    P_O = _fft_radial_mean(np.abs(O) ** 2)

    mth = str(method).strip().lower()
    diagnostics: dict[str, Any] = {
        "method": mth,
        "bl_floor": float(bl_floor),
        "H_max": H_max,
        "noise_rms_map": _estimate_map_noise_rms(obs, hits),
    }

    def _apply_tikhonov(alpha: float) -> np.ndarray:
        alpha_eff = float(alpha) * (H_max**2)
        denom = H_abs * H_abs + alpha_eff
        filt_local = np.zeros_like(H, dtype=np.complex128)
        good = (H_abs > thresh) & (denom > 0)
        filt_local[good] = np.conj(H[good]) / denom[good]
        return filt_local

    filt = np.zeros_like(H, dtype=np.complex128)
    if mth == "regularized":
        alpha = float(regularization_alpha)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("regularization_alpha must be finite and >= 0")
        filt = _apply_tikhonov(alpha)
        diagnostics["regularization_alpha"] = alpha
    elif mth in {"wiener", "weiner"}:
        # Noise power from modes where the beam response is negligible.
        noise_mode = H_abs <= max(thresh, 0.05 * H_max)
        if np.count_nonzero(noise_mode) < 16:
            # Fallback: outer third of Fourier radius.
            ny, nx = H_abs.shape
            ky = np.fft.fftfreq(ny)
            kx = np.fft.fftfreq(nx)
            kx2d, ky2d = np.meshgrid(kx, ky)
            k = np.sqrt(kx2d * kx2d + ky2d * ky2d)
            noise_mode = k >= (2.0 / 3.0) * float(np.max(k))
        P_N = float(np.median(P_O[noise_mode])) if np.any(noise_mode) else float(np.median(P_O))
        P_N = max(P_N, 1e-30)
        eps_h = max((0.01 * H_max) ** 2, 1e-30)
        P_S = np.maximum(P_O - P_N, 0.0) / np.maximum(H_abs * H_abs, eps_h)
        denom = H_abs * H_abs * P_S + P_N
        good = (H_abs > thresh) & (denom > 0)
        filt[good] = (np.conj(H[good]) * P_S[good]) / denom[good]
        diagnostics.update(
            {
                "noise_power": P_N,
                "signal_power_median": float(np.median(P_S[good])) if np.any(good) else 0.0,
                "n_modes_used": int(np.count_nonzero(good)),
            }
        )
    elif mth in {"residual_match", "residual"}:
        noise_rms = float(diagnostics["noise_rms_map"])
        if not np.isfinite(noise_rms) or noise_rms <= 0:
            # Fall back to mild Tikhonov if no noise estimate.
            alpha = float(regularization_alpha)
            filt = _apply_tikhonov(alpha)
            diagnostics["regularization_alpha"] = alpha
            diagnostics["residual_match_status"] = "fallback_no_noise_rms"
        else:
            mask = hits > 0
            alphas = np.logspace(-6, 1, 25)
            best_alpha = float(alphas[-1])
            best_err = np.inf
            best_resid_rms = np.nan
            for alpha in alphas:
                filt_try = _apply_tikhonov(float(alpha))
                dec_try = np.real(np.fft.ifft2(O * filt_try))
                recon = np.real(np.fft.ifft2(np.fft.fft2(np.nan_to_num(dec_try, nan=0.0)) * H))
                resid = (obs - recon)[mask]
                resid = resid[np.isfinite(resid)]
                if resid.size == 0:
                    continue
                resid_rms = float(np.sqrt(np.mean(resid * resid)))
                err = abs(np.log((resid_rms + 1e-30) / (noise_rms + 1e-30)))
                if err < best_err:
                    best_err = err
                    best_alpha = float(alpha)
                    best_resid_rms = resid_rms
            filt = _apply_tikhonov(best_alpha)
            diagnostics.update(
                {
                    "regularization_alpha": best_alpha,
                    "residual_rms": best_resid_rms,
                    "residual_match_status": "ok",
                }
            )
    else:
        raise ValueError(
            "method must be 'regularized', 'wiener'/'weiner', or 'residual_match'"
        )

    dec = np.real(np.fft.ifft2(O * filt))
    mask = hits > 0
    dec[~mask] = np.nan
    return dec, psf, fill_value, diagnostics


def preview_healpix_padding(
    map_ring: np.ndarray,
    hit_count: np.ndarray,
    pad_value: float | str = "edge",
    *,
    taper_width_deg: float | None = 3.0,
    apodize_fwhm_deg: float | None = None,
    gaussian_taper_fwhm_deg: float | None = None,
    gaussian_taper_normalize: bool = True,
) -> dict[str, Any]:
    """
    Apply HealPix padding/taper (same as ``convolve_beam_with_fit``) for inspection.
    """
    raw = np.asarray(map_ring, dtype=float)
    padded, fallback, taper_w, edge_target = _pad_unobserved_healpix_map(
        raw,
        hit_count,
        pad_value=pad_value,
        taper_width_deg=taper_width_deg,
        apodize_fwhm_deg=apodize_fwhm_deg,
        taper_fwhm_deg=gaussian_taper_fwhm_deg,
        taper_normalize=gaussian_taper_normalize,
    )
    return {
        "scalar_fallback": float(fallback),
        "fill_value": float(fallback),
        "edge_target_map": edge_target,
        "observed_map": raw,
        "padded_map": padded,
        "taper_weight": taper_w,
        "hit_count": np.asarray(hit_count, dtype=float),
        "taper_width_deg": taper_width_deg,
        "gaussian_taper_fwhm_deg": gaussian_taper_fwhm_deg or taper_width_deg,
        "pad_value_requested": pad_value,
    }


def _interp_gridded_map_at_pointings(
    map2d: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of a gridded map at (ra, dec) pointings."""
    from scipy.interpolate import RegularGridInterpolator

    z = np.asarray(map2d, dtype=float)
    ra_c = _normalize_ra_deg(np.asarray(ra_centers, dtype=float))
    dec_c = np.asarray(dec_centers, dtype=float)
    ra_q = _normalize_ra_deg(np.asarray(ra_deg, dtype=float))
    dec_q = np.asarray(dec_deg, dtype=float)
    if dec_c.size >= 2 and dec_c[0] > dec_c[-1]:
        dec_c = dec_c[::-1]
        z = z[::-1, :]
    if ra_c.size >= 2 and ra_c[0] > ra_c[-1]:
        ra_c = ra_c[::-1]
        z = z[:, ::-1]
    interp = RegularGridInterpolator(
        (dec_c, ra_c),
        z,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    return np.asarray(interp(np.column_stack([dec_q, ra_q])), dtype=float)


def deconvolve_gridded_map(
    grid_result: dict[str, Any],
    beam_params: dict[str, Any],
    *,
    attribute: str = "AB_",
    ra_deg: np.ndarray | None = None,
    dec_deg: np.ndarray | None = None,
    method: str = "wiener",
    regularization_alpha: float = 1e-2,
    bl_floor: float = 1e-3,
    pad_value: float | str = "edge",
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = None,
    taper_fwhm_deg: float | None = None,
    taper_kind: str = "cosine",
) -> dict[str, Any]:
    """
    Image-plane FFT deconvolution of a beam-weighted gridded map.

    Parameters
    ----------
    grid_result : dict
        Output of ``grid_beam_obs_pointing``.
    beam_params : dict
        Fitted beam parameters (``FWHM_deg`` / ``sigma_deg``).
    attribute : str
        Pol channel to deconvolve.
    ra_deg, dec_deg : array or None
        If given, interpolate the deconvolved map to these pointings and return
        ``deconvolved[attribute]``. Otherwise that entry is omitted.
    method : str
        ``wiener`` (default, data-driven), ``residual_match``, or ``regularized``.
    regularization_alpha, bl_floor
        Used by ``regularized`` / ``residual_match`` fallback; ``bl_floor`` is a
        safety cut on weak PSF Fourier modes for all methods.
    pad_value : float or {"edge","median","mean","zero"}
        Default ``"edge"``: taper toward per-pixel nearest-observed spectrum.
        A float or ``"median"``/``"mean"`` is only a **fallback** far from data.
    taper_width_deg : float or None
        If set (e.g. 3), blend from ``pad_value`` at the coverage edge to full observed
        values over this many degrees inward (cosine rolloff by default).
    apodize_fwhm_deg, taper_fwhm_deg
        Optional extra Gaussian smoothing of the taper window (FWHM in degrees).
    taper_kind : str
        ``"cosine"`` or ``"linear"`` rolloff when ``taper_width_deg`` is set.

    Returns
    -------
    dict
        ``grid``, ``observed_map``, ``deconvolved_map``, ``psf``, ``psf_fwhm_deg``,
        ``fill_value``, ``deconvolved`` (if ra/dec given), ``ra``, ``dec``,
        ``method``, ``diagnostics``.
    """
    if attribute not in grid_result["maps"]:
        raise KeyError(
            f"Attribute {attribute!r} not in grid maps. "
            f"Available: {list(grid_result['maps'].keys())}"
        )
    observed = grid_result["maps"][attribute]
    weight = grid_result["weight_sum"][attribute]
    sigma_deg = _sigma_deg_from_beam_params(beam_params)
    fwhm_deg = _fwhm_deg_from_beam_params(beam_params)

    deconv, psf, fill_value, diagnostics = _fft_deconvolve_gaussian_2d(
        observed,
        weight,
        pixel_size_deg=float(grid_result["pixel_size_deg"]),
        dec_centers=grid_result["dec_centers"],
        sigma_deg=sigma_deg,
        method=method,
        regularization_alpha=regularization_alpha,
        bl_floor=bl_floor,
        pad_value=pad_value,
        taper_width_deg=taper_width_deg,
        apodize_fwhm_deg=apodize_fwhm_deg,
        taper_fwhm_deg=taper_fwhm_deg,
        taper_kind=taper_kind,
    )

    out: dict[str, Any] = {
        "grid": grid_result,
        "observed_map": np.asarray(observed, dtype=float),
        "deconvolved_map": deconv,
        "psf": psf,
        "psf_fwhm_deg": float(fwhm_deg),
        "fill_value": float(fill_value),
        "taper_width_deg": taper_width_deg,
        "method": str(method),
        "regularization_alpha": float(
            diagnostics.get("regularization_alpha", regularization_alpha)
        ),
        "diagnostics": diagnostics,
        "deconvolved": {},
        "noise_spectra": {},
        "healpix_map": None,
    }
    if ra_deg is not None and dec_deg is not None:
        ra_a = np.atleast_1d(np.asarray(ra_deg, dtype=float))
        dec_a = np.atleast_1d(np.asarray(dec_deg, dtype=float))
        out["ra"] = ra_a.copy()
        out["dec"] = dec_a.copy()
        out["deconvolved"][attribute] = _interp_gridded_map_at_pointings(
            deconv,
            grid_result["ra_centers"],
            grid_result["dec_centers"],
            ra_a,
            dec_a,
        )
    return out


def plot_gridded_map(
    grid_result: dict[str, Any],
    attribute: str,
    *,
    ax: Any = None,
    show_coverage: bool = True,
    overlay_pointings: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    source_ra_deg: float | None = None,
    source_dec_deg: float | None = None,
    mark_source: bool = True,
    cmap_brightness: str = "viridis",
    cmap_coverage: str = "Greys",
    brightness_label: str | None = None,
    coverage_label: str = "Kernel weight sum",
    clim1: tuple[float, float] | None = None,
    clim2: tuple[float, float] | None = None,
    scale2: str = "linear",
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Plot output of ``grid_beam_obs_pointing`` on RA/Dec with colorbars.

    Axes limits match the gridded extent so edge pointings are visible. Raw
    pointings are drawn with open red markers (not white) so they stand out on
    the colormap.

    ``clim1`` sets color limits for the brightness panel; ``clim2`` for the
    coverage panel. Legacy ``clim`` / ``vmin`` / ``vmax`` kwargs alias ``clim1``.
    ``scale2`` is ``'linear'`` (default) or ``'log'`` for the coverage color
    scale (alias kwarg: ``scale``).
    """
    if attribute not in grid_result["maps"]:
        raise KeyError(
            f"Attribute {attribute!r} not in grid_result['maps']. "
            f"Available: {list(grid_result['maps'].keys())}"
        )
    Z = grid_result["maps"][attribute]
    ra_edges = grid_result["ra_edges"]
    dec_edges = grid_result["dec_edges"]
    wsum = grid_result["weight_sum"][attribute]
    xlim = (float(ra_edges[0]), float(ra_edges[-1]))
    ylim = (float(dec_edges[0]), float(dec_edges[-1]))

    if ax is not None and show_coverage:
        raise ValueError("Pass ax only when show_coverage=False (single panel).")

    ncols = 2 if show_coverage else 1
    if ax is None:
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
        axes_flat = axes.ravel()
    else:
        fig = ax.figure
        axes_flat = [ax]

    def _parse_clim(clim: Any, name: str) -> tuple[float | None, float | None]:
        if clim is None:
            return None, None
        try:
            return float(clim[0]), float(clim[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"{name} must be a (vmin, vmax) pair") from exc

    # Brightness clim: clim1 preferred; clim / vmin / vmax kept as aliases.
    clim_legacy = kwargs.pop("clim", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if clim1 is not None and clim_legacy is not None:
        raise ValueError("Pass either clim1 or clim, not both")
    if clim1 is None:
        clim1 = clim_legacy
    vmin1, vmax1 = _parse_clim(clim1, "clim1")
    if clim1 is not None and (vmin is not None or vmax is not None):
        raise ValueError("Pass either clim1/clim or vmin/vmax, not both")
    if clim1 is None:
        vmin1, vmax1 = vmin, vmax

    vmin2, vmax2 = _parse_clim(clim2, "clim2")
    scale_alias = kwargs.pop("scale", None)
    if scale_alias is not None:
        if scale2 != "linear":
            raise ValueError("Pass either scale2 or scale, not both")
        scale2 = str(scale_alias)
    scale2_key = str(scale2).strip().lower()
    if scale2_key not in {"linear", "log"}:
        raise ValueError("scale2 must be 'linear' or 'log'")
    cmap = kwargs.pop("cmap", cmap_brightness)
    mesh_kw = {"shading": "flat", **kwargs}

    def _overlay(ax_plot: Any) -> None:
        if overlay_pointings is None:
            return
        if isinstance(overlay_pointings, tuple):
            ra_ov, dec_ov = overlay_pointings
        else:
            ra_ov, dec_ov = overlay_pointings, None
        if dec_ov is None:
            raise ValueError("overlay_pointings must be (ra, dec) when a tuple")
        ax_plot.scatter(
            _normalize_ra_deg(np.asarray(ra_ov, dtype=float)),
            dec_ov,
            s=18,
            facecolors="none",
            edgecolors="crimson",
            linewidths=0.8,
            zorder=5,
            label="Pointings",
        )

    ax0 = axes_flat[0]
    pc0 = ax0.pcolormesh(
        ra_edges,
        dec_edges,
        Z,
        cmap=cmap,
        vmin=vmin1,
        vmax=vmax1,
        **mesh_kw,
    )
    ax0.set_xlim(xlim)
    ax0.set_ylim(ylim)
    ax0.set_xlabel("RA (deg)")
    ax0.set_ylabel("Dec (deg)")
    ax0.set_aspect("equal")
    ax0.set_title("Gridded brightness")
    _overlay(ax0)
    if mark_source and source_ra_deg is not None and source_dec_deg is not None:
        ax0.plot(
            float(_normalize_ra_deg(np.array([source_ra_deg]))[0]),
            float(source_dec_deg),
            "k*",
            ms=12,
            zorder=6,
            label="Source",
        )
    if overlay_pointings is not None or (
        mark_source and source_ra_deg is not None
    ):
        ax0.legend(fontsize=8, loc="best")
    cb0 = fig.colorbar(pc0, ax=ax0)
    cb0.set_label(
        brightness_label if brightness_label is not None else f"{attribute} (K)"
    )

    if show_coverage:
        import matplotlib.colors as mcolors

        ax1 = axes_flat[1]
        cov_kw = dict(mesh_kw)
        if scale2_key == "log":
            # LogNorm needs strictly positive data/limits.
            wplot = np.asarray(wsum, dtype=float)
            positive = wplot[np.isfinite(wplot) & (wplot > 0)]
            if vmin2 is None:
                vmin2 = float(np.nanmin(positive)) if positive.size else 1.0
            if vmax2 is None:
                vmax2 = float(np.nanmax(positive)) if positive.size else 1.0
            if not (vmin2 > 0 and vmax2 > vmin2):
                raise ValueError(
                    "scale2='log' requires clim2 with 0 < vmin < vmax "
                    f"(got vmin={vmin2}, vmax={vmax2})"
                )
            cov_kw["norm"] = mcolors.LogNorm(vmin=vmin2, vmax=vmax2)
            pc1 = ax1.pcolormesh(
                ra_edges,
                dec_edges,
                wplot,
                cmap=cmap_coverage,
                **cov_kw,
            )
        else:
            pc1 = ax1.pcolormesh(
                ra_edges,
                dec_edges,
                wsum,
                cmap=cmap_coverage,
                vmin=vmin2,
                vmax=vmax2,
                **cov_kw,
            )
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_xlabel("RA (deg)")
        ax1.set_ylabel("Dec (deg)")
        ax1.set_aspect("equal")
        ax1.set_title("Coverage")
        _overlay(ax1)
        cb1 = fig.colorbar(pc1, ax=ax1)
        cb1.set_label(coverage_label)

    if show:
        plt.tight_layout()
        plt.show()
    return axes_flat if show_coverage else ax0


def map_source_axis_cuts(
    map2d: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    *,
    source_ra_deg: float | None = None,
    source_dec_deg: float | None = None,
) -> dict[str, Any]:
    """
    RA and Dec cuts of a gridded map through the source (or the map peak).

    Returns 1-D arrays ``ra`` / ``amp_vs_ra`` (row at the source Dec) and
    ``dec`` / ``amp_vs_dec`` (column at the source RA).
    """
    z = np.asarray(map2d, dtype=float)
    ra_c = np.asarray(ra_centers, dtype=float)
    dec_c = np.asarray(dec_centers, dtype=float)
    if z.ndim != 2 or z.shape != (dec_c.size, ra_c.size):
        raise ValueError(
            f"map shape {z.shape} does not match "
            f"(n_dec={dec_c.size}, n_ra={ra_c.size})"
        )
    if source_ra_deg is None or source_dec_deg is None:
        if not np.any(np.isfinite(z)):
            raise ValueError("map has no finite pixels to locate a source")
        i_src, j_src = np.unravel_index(int(np.nanargmax(z)), z.shape)
        src_ra = float(ra_c[j_src])
        src_dec = float(dec_c[i_src])
    else:
        src_ra = float(source_ra_deg)
        src_dec = float(source_dec_deg)
        j_src = int(np.argmin(np.abs(ra_c - src_ra)))
        i_src = int(np.argmin(np.abs(dec_c - src_dec)))
    return {
        "source_ra_deg": src_ra,
        "source_dec_deg": src_dec,
        "row": int(i_src),
        "col": int(j_src),
        "ra": ra_c,
        "amp_vs_ra": z[i_src, :].copy(),
        "dec": dec_c,
        "amp_vs_dec": z[:, j_src].copy(),
    }


def plot_map_source_axis_cuts(
    map2d: np.ndarray,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    *,
    source_ra_deg: float | None = None,
    source_dec_deg: float | None = None,
    overlay_ra: np.ndarray | None = None,
    overlay_dec: np.ndarray | None = None,
    overlay_amp: np.ndarray | None = None,
    strip_deg: float | None = None,
    pixel_size_deg: float | None = None,
    map_label: str = "Map cut",
    extra_maps: list[tuple[np.ndarray, str]] | None = None,
    mark_ra_deg: float | None = None,
    mark_dec_deg: float | None = None,
    ylabel: str = "Amplitude",
    show: bool = True,
) -> dict[str, Any]:
    """
    Plot RA vs amplitude and Dec vs amplitude through a source on a gridded map.

    Cuts go through ``source_ra_deg`` / ``source_dec_deg`` if given, otherwise
    through the brightest finite pixel so the profile is the actual peak.
    ``mark_ra_deg`` / ``mark_dec_deg`` (e.g. catalog 3C353) are extra vertical
    lines. Overlay samples are those that lie on the same RA or Dec line as
    the map cut, within a few pixels (or ``strip_deg``).
    """
    cuts = map_source_axis_cuts(
        map2d,
        ra_centers,
        dec_centers,
        source_ra_deg=source_ra_deg,
        source_dec_deg=source_dec_deg,
    )
    pix = float(pixel_size_deg) if pixel_size_deg is not None else None
    if pix is None and ra_centers.size > 1:
        pix = abs(float(ra_centers[1] - ra_centers[0]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_ra, ax_dec = axes
    ax_ra.plot(cuts["ra"], cuts["amp_vs_ra"], "-", lw=1.6, color="C0", label=map_label)
    ax_dec.plot(cuts["dec"], cuts["amp_vs_dec"], "-", lw=1.6, color="C0", label=map_label)
    if extra_maps:
        for k, (extra, lab) in enumerate(extra_maps, start=1):
            extra_cuts = map_source_axis_cuts(
                extra,
                ra_centers,
                dec_centers,
                source_ra_deg=cuts["source_ra_deg"],
                source_dec_deg=cuts["source_dec_deg"],
            )
            ax_ra.plot(
                extra_cuts["ra"],
                extra_cuts["amp_vs_ra"],
                "-",
                lw=1.4,
                color=f"C{k}",
                label=lab,
            )
            ax_dec.plot(
                extra_cuts["dec"],
                extra_cuts["amp_vs_dec"],
                "-",
                lw=1.4,
                color=f"C{k}",
                label=lab,
            )

    if overlay_ra is not None:
        if overlay_dec is None or overlay_amp is None:
            raise ValueError("overlay_ra, overlay_dec, and overlay_amp must be given together")
        ra_p = np.asarray(overlay_ra, dtype=float)
        dec_p = np.asarray(overlay_dec, dtype=float)
        amp_p = np.asarray(overlay_amp, dtype=float)
        src_ra = cuts["source_ra_deg"]
        src_dec = cuts["source_dec_deg"]
        finite = np.isfinite(ra_p) & np.isfinite(dec_p) & np.isfinite(amp_p)
        ra_f, dec_f, amp_f = ra_p[finite], dec_p[finite], amp_p[finite]
        cos_dec = max(abs(float(np.cos(np.radians(src_dec)))), 1e-6)
        if strip_deg is not None:
            hw_dec = hw_ra = float(strip_deg)
        else:
            # Daisy pointings are not constant-Dec rows. Along-track spacing
            # as a half-width keeps only the nearest sample.
            hw = max(3.0 * pix, 0.03) if pix is not None else 0.03
            hw_dec = hw_ra = hw
        m_ra = np.abs(dec_f - src_dec) <= hw_dec
        m_dec = np.abs(ra_f - src_ra) * cos_dec <= hw_ra
        ax_ra.scatter(
            ra_f[m_ra],
            amp_f[m_ra],
            s=10,
            c="0.35",
            alpha=0.45,
            zorder=3,
            label="Scan samples",
        )
        ax_dec.scatter(
            dec_f[m_dec],
            amp_f[m_dec],
            s=10,
            c="0.35",
            alpha=0.45,
            zorder=3,
            label="Scan samples",
        )

    ax_ra.axvline(cuts["source_ra_deg"], color="k", ls=":", lw=1.0, label="Map peak")
    ax_dec.axvline(cuts["source_dec_deg"], color="k", ls=":", lw=1.0, label="Map peak")
    if mark_ra_deg is not None:
        ax_ra.axvline(float(mark_ra_deg), color="C3", ls="--", lw=1.0, label="Catalog RA")
    if mark_dec_deg is not None:
        ax_dec.axvline(float(mark_dec_deg), color="C3", ls="--", lw=1.0, label="Catalog Dec")
    ax_ra.set_xlabel("RA (deg)")
    ax_dec.set_xlabel("Dec (deg)")
    ax_ra.set_ylabel(ylabel)
    ax_dec.set_ylabel(ylabel)
    ax_ra.set_title("Scan across source (RA)")
    ax_dec.set_title("Scan across source (Dec)")
    ax_ra.legend(fontsize=8, loc="best")
    ax_dec.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if show:
        plt.show()
    cuts["fig"] = fig
    cuts["axes"] = axes
    return cuts


def _beam_obs_pointing_ra_dec_attrs(
    beam_obs_pointing: object,
    attribute: str | None,
    attributes: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, Any, list[str]]:
    """Validate matched pointing object; return ``ra``, ``dec``, pol mean container, channel names."""
    ra = np.atleast_1d(np.asarray(getattr(beam_obs_pointing, "ra"), dtype=float))
    dec = np.atleast_1d(np.asarray(getattr(beam_obs_pointing, "dec"), dtype=float))
    if ra.size != dec.size:
        raise ValueError("beam_obs_pointing must have ra and dec of the same length")

    spec_source = get_pol_source(beam_obs_pointing, kind="mean")
    if spec_source is None:
        raise ValueError(
            "beam_obs_pointing must have calibrated_spec_mean or spec_mean "
            "(e.g. output of match_data_and_pointing)"
        )
    available = get_available_pol_names(beam_obs_pointing, kind="mean")
    if not available:
        raise ValueError("No polarization mean data on beam_obs_pointing")

    if attributes is not None:
        attrs = list(attributes)
        for name in attrs:
            if name not in available:
                raise ValueError(
                    f"Attribute {name!r} not available. Available: {available}"
                )
    elif attribute is not None:
        if attribute not in available:
            raise ValueError(
                f"Attribute {attribute!r} not available. Available: {available}"
            )
        attrs = [attribute]
    else:
        attrs = available

    return ra, dec, spec_source, attrs


def convolve_beam_with_fit(
    beam_obs_pointing: object,
    beam_params: dict[str, Any],
    *,
    nside: int,
    attribute: str | None = None,
    attributes: list[str] | None = None,
    freq_index: int | None = None,
    bl_floor: float = 1e-3,
    lmax: int | None = None,
    map2alm_iter: int = 3,
    method: str = "regularized",
    regularization_alpha: float = 1e-6,
    signal_cl: dict[str, np.ndarray] | np.ndarray | None = None,
    noise_cl: dict[str, np.ndarray] | np.ndarray | None = None,
    estimate_noise_spectrum: bool = True,
    pad_unobserved: bool = False,
    pad_value: float | str = "edge",  # per-pixel nearest-observed; float = far-field fallback only
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = None,
    gaussian_taper_fwhm_deg: float | None = None,
    gaussian_taper_normalize: bool = True,
) -> dict[str, Any]:
    """
    Spherical-harmonic deconvolution of the fitted circular Gaussian PSF.

    A ``radio_beam.Beam`` matching the fitted FWHM defines the PSF model (see
    https://radio-beam.readthedocs.io/en/latest/). Per channel, pointings are
    gridded with ``HealPixMap.fill_from_pointing`` (mean per pixel), then both sky
    map and beam are represented in spherical harmonics with healpy.

    Deconvolution filter:
      - ``method="regularized"``: ``F_l = B_l / (B_l^2 + alpha)``
      - ``method="wiener"`` (or ``"weiner"``): ``F_l = B_l C_l^S / (B_l^2 C_l^S + C_l^N)``

    Modes where ``B_l`` is below ``bl_floor`` times the peak of ``B_l``
    (excluding the monopole) are suppressed.

    This is a healpy ``map2alm`` / ``almxfl`` / ``alm2map`` inverse-beam step, not
    map-plane Richardson–Lucy or Wiener filtering with ``Beam.as_kernel`` (see
    radio-beam's convolution-kernel documentation for that style of kernel).

    Parameters
    ----------
    beam_obs_pointing
        Output of ``match_data_and_pointing`` with ``ra``, ``dec``, and
        ``calibrated_spec_mean`` or ``spec_mean``.
    beam_params : dict
        Return value of ``fit_beam_gaussian``; must include a width key
        (``FWHM_deg``, ``sigma_deg``, or ``half_power_radius_deg``).
    nside : int
        HealPix resolution for map-making and deconvolution.
    attribute : str or None
        Single pol channel (e.g. ``\"AB_\"``). Ignored if ``attributes`` is set.
    attributes : list of str or None
        If set, only these channels are processed. If both ``attribute`` and
        ``attributes`` are ``None``, all available channels are used.
    freq_index : int or None
        Per-channel frequency index when arrays are ``(n_pointing, n_freq)``;
        otherwise mean over frequency.
    bl_floor : float
        Modes with ``B_l <= bl_floor * max(B)`` (excluding the monopole) are not
        boosted; larger values give a smoother, less aggressive sharpen.
    lmax : int or None
        Band limit for ``map2alm`` / ``almxfl``; default ``3 * nside - 1``.
    map2alm_iter : int
        ``iter`` passed to ``hp.map2alm`` (see healpy).
    method : str
        Deconvolution method: ``"regularized"`` or ``"wiener"`` (``"weiner"`` alias).
    regularization_alpha : float
        Tikhonov regularization parameter for ``method="regularized"``.
    signal_cl, noise_cl : dict[str, np.ndarray] | np.ndarray | None
        Optional input spectra for Wiener filtering. If dicts are given, channel
        names are used as keys. If omitted in Wiener mode, spectra are estimated
        from observed map power and a high-``l`` white-noise tail.
    estimate_noise_spectrum : bool
        If True, estimate residual noise spectrum per channel by reconvolving the
        recovered sky map and analyzing residuals.
    pad_unobserved : bool
        If True, fill pixels with ``hit_count <= 0`` before the harmonic transform to
        reduce edge effects from partial-sky coverage (sinc/ringing in ell-space).
    pad_value : float | {"median","mean","zero"}
        Fill value for unobserved pixels when ``pad_unobserved=True``. If a string,
        the statistic is computed over observed pixels.
    apodize_fwhm_deg : float or None
        If set (degrees) and ``pad_unobserved=True``, apodize the observed/unobserved
        boundary by smoothing the hit-mask with this Gaussian FWHM, blending into the
        fill value outside the observed region.
    gaussian_taper_fwhm_deg : float or None
        If set (degrees) and ``pad_unobserved=True``, apply a Gaussian taper window
        derived from the smoothed hit-mask, blending the observed patch into the
        padding baseline before ``map2alm``. This directly suppresses sinc-like
        ringing from a hard-edged patch window.
    gaussian_taper_normalize : bool
        If True (default), normalize the taper window so its maximum is 1.

    Returns
    -------
    dict
        ``ra``, ``dec``, ``deconvolved`` (channel → array sampled from the
        deconvolved map at each pointing), ``psf_fwhm_deg``,
        ``psf_radio_beam`` (string form), ``healpix_map`` (``HealPixMap`` with
        deconvolved channel maps), and ``noise_spectra`` (per channel dictionary;
        empty if ``estimate_noise_spectrum=False``).
    """
    hp.check_max_nside(int(nside))
    psf_rb, fwhm_deg = _radio_beam_psf_from_fit(beam_params)
    fwhm_rad = np.radians(float(fwhm_deg))

    ra, dec, spec_source, attrs = _beam_obs_pointing_ra_dec_attrs(
        beam_obs_pointing, attribute, attributes
    )

    out = HealPixMap(int(nside))
    out._channel_maps.clear()
    out._channel_stds.clear()

    deconvolved: dict[str, np.ndarray] = {}
    noise_spectra: dict[str, dict[str, np.ndarray | float]] = {}
    combined_hits: np.ndarray | None = None
    for attr in attrs:
        vals = _pol_channel_values_1d(spec_source, attr, freq_index)
        if vals.size != ra.size:
            raise ValueError(
                f"Length mismatch: ra/dec length {ra.size}, {attr!r} length {vals.size}"
            )
        tmp = HealPixMap(int(nside))
        tmp.fill_from_pointing(ra, dec, values=vals)
        hits = tmp._hit_count.copy()
        raw_map = tmp.map.copy()

        map_for_harmonics = raw_map
        mask_unseen = True
        if pad_unobserved:
            map_for_harmonics, _, _, _ = _pad_unobserved_healpix_map(
                raw_map,
                hits,
                pad_value=pad_value,
                taper_width_deg=taper_width_deg,
                apodize_fwhm_deg=apodize_fwhm_deg,
                taper_fwhm_deg=gaussian_taper_fwhm_deg,
                taper_normalize=gaussian_taper_normalize,
            )
            mask_unseen = False
        elif gaussian_taper_fwhm_deg is not None:
            raise ValueError("gaussian_taper_fwhm_deg requires pad_unobserved=True")

        sig_cl_attr = signal_cl.get(attr) if isinstance(signal_cl, dict) else signal_cl
        noi_cl_attr = noise_cl.get(attr) if isinstance(noise_cl, dict) else noise_cl
        dec_map = _deconvolve_healpix_gaussian_psf(
            map_for_harmonics,
            hits,
            int(nside),
            fwhm_rad,
            bl_floor=bl_floor,
            lmax=lmax,
            map2alm_iter=map2alm_iter,
            mask_unseen=mask_unseen,
            method=method,
            regularization_alpha=regularization_alpha,
            signal_cl=sig_cl_attr,
            noise_cl=noi_cl_attr,
        )
        out._channel_maps[attr] = dec_map
        std = np.full(tmp.npix, np.nan, dtype=float)
        std[hits > 0] = 0.0
        out._channel_stds[attr] = std
        if estimate_noise_spectrum:
            noise_spectra[attr] = estimate_noise_spectrum_from_deconvolution(
                raw_map,
                dec_map,
                hits,
                fwhm_rad=fwhm_rad,
                lmax=lmax,
                map2alm_iter=map2alm_iter,
            )
        deconvolved[attr] = _interp_healpix_ring_to_radec_deg(dec_map, ra, dec)
        combined_hits = (
            hits.copy()
            if combined_hits is None
            else np.maximum(combined_hits, hits)
        )

    assert combined_hits is not None
    out._hit_count = combined_hits
    out.map = combined_hits.copy()
    out._map_std = np.full(out.npix, np.nan, dtype=float)
    out._map_std[combined_hits > 0] = 0.0

    return {
        "ra": ra.copy(),
        "dec": dec.copy(),
        "deconvolved": deconvolved,
        "psf_fwhm_deg": float(fwhm_deg),
        "psf_radio_beam": str(psf_rb),
        "healpix_map": out,
        "noise_spectra": noise_spectra,
    }
