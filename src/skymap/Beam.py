'''
Beam estimation for 310 MHz observations

- Get RA and Dec of the input source from calibrators.dat.
- Using matched pointing + spectrum data (e.g. ``match_data_and_pointing``), compute
  per-pointing radial offset from the source (same flat-sky convention as ``plot_offset_map``).
- Fit a Gaussian on a constant floor (mean brightness beyond ~1° offset) to estimate beam width.
- Optionally approximate removal of the fitted circular Gaussian PSF on a HealPix
  grid (``convolve_beam_with_fit``), using radio-beam (https://radio-beam.readthedocs.io/en/latest/)
  for the PSF model and healpy for regularized harmonic ``1 / B_\\ell``.
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


def _flat_sky_radial_offset_deg(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    source_ra_deg: float,
    source_dec_deg: float,
) -> np.ndarray:
    """
    Flat-sky radial offset in degrees: sqrt(dRA^2 + dDec^2).
    RA is normalized to [0, 360) and dRA uses shortest separation (same as
    ``plot_offset_map``).
    """
    ra_deg = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec_deg = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    ra_n = _normalize_ra_deg(ra_deg)
    src_ra = float(_normalize_ra_deg(np.array([float(source_ra_deg)]))[0])
    src_dec = float(source_dec_deg)
    dra = ((ra_n - src_ra + 180.0) % 360.0) - 180.0
    ddec = dec_deg - src_dec
    return np.sqrt(dra * dra + ddec * ddec)


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

    Offset convention matches ``plot_offset_map`` (flat-sky, RA wrapped).
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

    If `source_ra_deg` and `source_dec_deg` are provided, the offset is always
    computed from the source (so the color is guaranteed to represent radial
    offset, even if a caller accidentally passes some other array as offset_deg).
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
    ax.scatter(x, y, s=12, label="Data (K)", zorder=3, alpha=0.35)
    ax.axhline(
        baseline_k,
        color='k',
        ls=":",
        lw=1.0,
        label=rf"baseline (mean $r>{baseline_outer_deg:g}°$)",
    )
    ax.plot(
        x_fit,
        y_model,
        "k",
        label= rf"Model: $\sigma={sigma:.3f}$, HPBW={hpbw:.3f}$, FWHM={fwhm:.3f}$ ",
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


def _pad_unobserved_healpix_map(
    map_ring: np.ndarray,
    hit_count: np.ndarray,
    *,
    pad_value: float | str = "median",
    apodize_fwhm_deg: float | None = None,
    taper_fwhm_deg: float | None = None,
    taper_normalize: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Fill unobserved HealPix pixels (``hit_count <= 0``) with a constant, and optionally
    apodize/taper the hard edge by smoothing the observed/unobserved mask.

    This is primarily to reduce ringing/leakage in spherical-harmonic transforms when
    only a small sky patch is observed.

    Returns
    -------
    padded_map_ring, fill_value
    """
    m = np.asarray(map_ring, dtype=np.float64).copy()
    hits = np.asarray(hit_count, dtype=float)

    observed = hits > 0
    valid_obs = observed & np.isfinite(m) & (m != hp.UNSEEN)

    pv = pad_value
    if isinstance(pv, str):
        key = pv.strip().lower()
        if key == "median":
            fill_value = float(np.nanmedian(m[valid_obs])) if np.any(valid_obs) else 0.0
        elif key == "mean":
            fill_value = float(np.nanmean(m[valid_obs])) if np.any(valid_obs) else 0.0
        elif key in {"zero", "0"}:
            fill_value = 0.0
        else:
            raise ValueError("pad_value must be a float or one of {'median','mean','zero'}")
    else:
        fill_value = float(pv)

    m[~observed] = fill_value
    m[m == hp.UNSEEN] = fill_value
    m[~np.isfinite(m)] = fill_value

    if apodize_fwhm_deg is not None:
        fwhm_deg = float(apodize_fwhm_deg)
        if not np.isfinite(fwhm_deg) or fwhm_deg <= 0:
            raise ValueError("apodize_fwhm_deg must be positive and finite when provided")
        w = hp.smoothing(observed.astype(np.float64), fwhm=np.radians(fwhm_deg), verbose=False)
        w = np.asarray(w, dtype=np.float64)
        w = np.clip(w, 0.0, 1.0)
        m = fill_value * (1.0 - w) + m * w

    if taper_fwhm_deg is not None:
        tf = float(taper_fwhm_deg)
        if not np.isfinite(tf) or tf <= 0:
            raise ValueError("taper_fwhm_deg must be positive and finite when provided")
        w = hp.smoothing(observed.astype(np.float64), fwhm=np.radians(tf), verbose=False)
        w = np.asarray(w, dtype=np.float64)
        w = np.clip(w, 0.0, 1.0)
        if taper_normalize:
            wmax = float(np.nanmax(w)) if w.size else 1.0
            if np.isfinite(wmax) and wmax > 0:
                w = np.clip(w / wmax, 0.0, 1.0)
        # Gaussian taper: smoothly blend the observed patch into the padding baseline
        m = fill_value + (m - fill_value) * w

    return m, fill_value


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
) -> np.ndarray:
    """One row per pointing for pol channel ``attribute`` (same rules as ``HealPixMap.fill_from_pointing_data``)."""
    arr = getattr(spec_source, attribute)
    if getattr(arr, "unit", None) is not None:
        arr = getattr(arr, "value", arr)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        if freq_index is not None:
            return arr[:, int(freq_index)]
        return np.nanmean(arr, axis=1)
    return arr.ravel()


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
    pad_value: float | str = "median",
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
            map_for_harmonics, _ = _pad_unobserved_healpix_map(
                raw_map,
                hits,
                pad_value=pad_value,
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
