'''
Beam estimation for 310 MHz observations

- Get RA and Dec of the input source from calibrators.dat.
- Using a HealPixMap where mean is mapped in RA and Dec, compute a histogram
  with x-axis = radial pointing offset in degrees from the source RA/Dec and
  y-axis = mean in K (binned by offset).
- Optionally fit a Gaussian to estimate beam width (full beam of feed+dish).
'''

from __future__ import annotations

from pathlib import Path
from typing import Any

import healpy as hp
import matplotlib.pyplot as plt
import warnings

import numpy as np
from scipy.optimize import curve_fit
from skymap.healmap import HealPixMap

# Default catalogue next to this module
_CALIBRATORS_PATH = Path(__file__).resolve().parent / "calibrators.dat"


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


def compute_beam_histogram(
    healmap_or_ra_deg: HealPixMap | np.ndarray,
    dec_deg: np.ndarray | None = None,
    values: np.ndarray | None = None,
    source_name: str | None = None,
    source_ra_deg: float | None = None,
    source_dec_deg: float | None = None,
    attribute: str | None = None,
    bin_size_deg: float = 0.1,
    calibrators_path: Path | str | None = None,
    max_offset_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Beam histogram: x=radial pointing offset (deg), y=mean(values) per bin.

    Pass either:
    - a `HealPixMap` as first arg (set `attribute` if needed), OR
    - arrays: (ra_deg, dec_deg, values).

    Source is specified by either `source_name` (looked up in calibrators.dat)
    or explicit `source_ra_deg` + `source_dec_deg`.
    """
    # Input normalization: HealPixMap or arrays
    if hasattr(healmap_or_ra_deg, "get_map_radec_values"):
        healmap = healmap_or_ra_deg  # type: ignore[assignment]
        ra_deg, dec_deg2, values2 = healmap.get_map_radec_values(attribute)
    else:
        ra_deg = np.asarray(healmap_or_ra_deg)  # type: ignore[assignment]
        if dec_deg is None or values is None:
            raise ValueError("When passing arrays, provide dec_deg and values")
        dec_deg2 = dec_deg
        values2 = values

    if source_ra_deg is not None and source_dec_deg is not None:
        src_ra, src_dec = source_ra_deg, source_dec_deg
    elif source_name is not None:
        src_ra, src_dec = get_source_radec(source_name, path=calibrators_path)
    else:
        raise ValueError("Provide either source_name or (source_ra_deg, source_dec_deg)")

    ra_deg = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec_deg2 = np.atleast_1d(np.asarray(dec_deg2, dtype=float))
    values2 = np.atleast_1d(np.asarray(values2, dtype=float))
    if ra_deg.size != dec_deg2.size or ra_deg.size != values2.size:
        raise ValueError("ra_deg, dec_deg, and values must have the same length")
    if ra_deg.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Flat-sky approximation (treat RA/Dec as Cartesian axes in degrees).
    # Use shortest RA separation to avoid 0/360 wrap artifacts.
    dra = ((ra_deg - float(src_ra) + 180.0) % 360.0) - 180.0
    ddec = dec_deg2 - float(src_dec)
    offset_deg = np.sqrt(dra * dra + ddec * ddec)

    if max_offset_deg is not None:
        mask = offset_deg <= max_offset_deg
        offset_deg = offset_deg[mask]
        values2 = values2[mask]

    if offset_deg.size == 0:
        return np.array([]), np.array([]), np.array([])

    max_off = float(np.max(offset_deg))
    bin_edges = np.arange(0, max_off + bin_size_deg, bin_size_deg)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    # Assign bins
    bin_idx = np.digitize(offset_deg, bin_edges) - 1
    bin_idx[bin_idx == n_bins] = n_bins - 1

    mean_k = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = bin_idx == i
        if np.any(mask):
            mean_k[i] = np.nanmean(values2[mask])
            counts[i] = mask.sum()

    return bin_centers, mean_k, counts




def _normalize_ra_deg(ra: np.ndarray) -> np.ndarray:
    """Put RA in [0, 360) to avoid -180/360 wrap issues."""
    ra = np.asarray(ra, dtype=float)
    return ra % 360.0


def plot_offset_map(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    offset_deg: np.ndarray | None = None,
    *,
    source_ra_deg: float | None = None,
    source_dec_deg: float | None = None,
    relative_frame: bool = False,
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
    Plot offset from source as color on RA/Dec: x=RA, y=Dec, color=offset_deg.
    All RA/Dec in degrees (RA 0--360 or -180--180; Dec -90--90). RA is normalized
    to [0, 360) so offset is consistent.

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
        # Flat-sky offset to match compute_beam_histogram
        dra = ((ra_deg - src_ra + 180.0) % 360.0) - 180.0
        ddec = dec_deg - src_dec
        offset_deg = np.sqrt(dra * dra + ddec * ddec)
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

    # Choose plotting coordinates:
    # - absolute frame: x=RA, y=Dec
    # - relative frame: x=ΔRA, y=ΔDec with the source at (0,0) (matches plot_relative_frame)
    if relative_frame:
        if src_ra is None or src_dec is None:
            raise ValueError("relative_frame=True requires source_ra_deg and source_dec_deg")
        x = ((ra_deg - src_ra + 180.0) % 360.0) - 180.0
        y = dec_deg - src_dec
        x_label = "ΔRA (deg)"
        y_label = "ΔDec (deg)"
        mark_x, mark_y = 0.0, 0.0
    else:
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
    Axisymmetric beam model used for convolution (radial offset in **degrees**).

    - ``x ≤ r_poly_deg``: core Gaussian ``A exp(-x²/(2σ²))`` (same ``A``, ``σ`` as fit).
    - ``x > r_poly_deg``: polynomial ``P(x)`` from ``poly_coeffs`` (low-to-high),
      clipped to be non-negative for a physical beam response.

    ``beam_params`` should be the return value of ``fit_beam_gaussian`` (needs
    ``A``, ``poly_coeffs``, ``r_poly_deg``, and a width key).
    """
    if "A" not in beam_params:
        raise ValueError("beam_params must include 'A' (output of fit_beam_gaussian).")
    if "poly_coeffs" not in beam_params:
        raise ValueError("beam_params must include 'poly_coeffs'.")
    x = np.asarray(radial_offset_deg, dtype=float)
    A = float(beam_params["A"])
    sigma_deg = _sigma_deg_from_beam_params(beam_params)
    r_poly = float(beam_params.get("r_poly_deg", 1.2))

    g = _gaussian_only(x, A, sigma_deg)
    proc = np.asarray(beam_params["poly_coeffs"], dtype=float).ravel()
    if proc.size == 0:
        p = np.zeros_like(x, dtype=float)
    else:
        p = np.polyval(proc[::-1], x)
    p = np.clip(p, 0.0, np.inf)

    return np.where(x <= r_poly, g, p)


# Radial edge (deg) for [0, 1] beam normalization: 1 at r=0, 0 at this radius.
_BEAM_NORM_R_MAX_DEG = 5.0


def _beam_approx_endpoint_affine(
    beam_params: dict[str, Any],
    r_edge_deg: float = _BEAM_NORM_R_MAX_DEG,
) -> tuple[float, float, float]:
    """
    Return ``(B(0), B(r_edge), B(0) - B(r_edge))`` from ``beam_approximation``.

    Used to affine-map the fitted radial profile to 1 at the origin and 0 at
    ``r_edge_deg`` (requires ``B(0) > B(r_edge)``).
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
    **W(5°)=0** (default ``r_edge_deg``), matching the convolution kernel in
    ``convolve_beam_with_fit``. Values are forced to 0 for ``r >= r_edge_deg``;
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
    r_poly_line: bool = True,
    ax: Any = None,
    show: bool = True,
    title: str | None = "Piecewise beam approximation",
) -> Any:
    """
    Plot ``beam_approximation`` vs radial offset (degrees).

    Parameters
    ----------
    beam_params : dict
        Same dictionary passed to ``beam_approximation`` / ``convolve_beam_with_fit``
        (e.g. return value of ``fit_beam_gaussian``).
    x_max_deg : float
        Upper limit of the horizontal axis (deg); capped at 5° to match the
        domain used for convolution.
    npts : int
        Number of samples along the radius.
    normalize : bool
        If True (default), scale the profile to ``[0, 1]`` with amplitude 1 at
        0° and 0 at 5° (affine map of ``beam_approximation``, same rule as
        ``convolve_beam_with_fit``).
    show_components : bool
        If True, also draw the full Gaussian and full polynomial (dotted) behind
        the piecewise composite for comparison.
    r_poly_line : bool
        If True, mark ``r_poly_deg`` with a vertical line.
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
    r_poly = float(beam_params.get("r_poly_deg", 1.2))

    if normalize:
        affine = _beam_approx_endpoint_affine(beam_params, _BEAM_NORM_R_MAX_DEG)
        _, B5, denom = affine
        y = _beam_approx_radial_endpoint_normalized(
            x, beam_params, endpoint_affine=affine
        )
    else:
        y = np.asarray(beam_approximation(x, beam_params), dtype=float)
        B5 = float("nan")
        denom = float("nan")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if show_components:
        A = float(beam_params["A"])
        sigma_deg = _sigma_deg_from_beam_params(beam_params)
        proc = np.asarray(beam_params["poly_coeffs"], dtype=float).ravel()
        g_full = _gaussian_only(x, A, sigma_deg)
        if proc.size == 0:
            p_full = np.zeros_like(x, dtype=float)
        else:
            p_full = np.clip(np.polyval(proc[::-1], x), 0.0, np.inf)
        if normalize:
            g_full = np.clip((g_full - B5) / denom, 0.0, 1.0)
            p_full = np.clip((p_full - B5) / denom, 0.0, 1.0)
        ax.plot(x, g_full, ":", color="0.6", lw=1.2, alpha=0.85, label="Gaussian G(x)")
        ax.plot(x, p_full, ":", color="0.4", lw=1.2, alpha=0.85, label="Polynomial P(x)")

    ax.plot(
        x,
        y,
        "b-",
        lw=2,
        label="Beam B(x)" if not normalize else r"$B(r)$: 1 at $0°$, 0 at $5°$",
    )

    if r_poly_line:
        ax.axvline(r_poly, color="0.45", ls="--", lw=1.0, alpha=0.85, label=r"$r_\mathrm{poly}$")

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


def _beam_approximation_beam_bl(
    lmax: int,
    beam_params: dict[str, Any],
    *,
    theta_max_deg: float,
    ntheta: int,
    normalize: bool = False,
) -> np.ndarray:
    """
    Transfer function B_ell / B_0 from ``beam_approximation``, with ``B_0`` set to 1.

    If ``normalize`` is True, the radial profile is affine-mapped so
    ``beam_approximation`` has amplitude **1 at 0°** and **0 at 5°** (values in
    ``[0, 1]``, same rule as ``plot_beam_approximation`` / ``convolve_beam_with_fit``).
    If False, amplitudes from the fit are kept.
    """
    theta_rad = np.linspace(0.0, np.radians(float(theta_max_deg)), int(ntheta))
    x_deg = np.degrees(theta_rad)

    if normalize:
        beam_prof = _beam_approx_radial_endpoint_normalized(
            x_deg, beam_params, r_edge_deg=_BEAM_NORM_R_MAX_DEG
        )
    else:
        beam_prof = beam_approximation(x_deg, beam_params)
        beam_prof = np.clip(beam_prof, 0.0, np.inf)
        peak = float(beam_prof[0])
        if not np.isfinite(peak) or peak <= 0.0:
            peak = float(np.nanmax(beam_prof))
        if not np.isfinite(peak) or peak <= 0.0:
            raise ValueError("Beam approximation has no positive peak; check the fit.")
    bl = hp.beam2bl(beam_prof, theta_rad, int(lmax))
    b0 = float(bl[0])
    if not np.isfinite(b0) or b0 == 0.0:
        raise ValueError(
            "beam2bl produced invalid monopole; try increasing theta_max_deg or ntheta."
        )
    return bl / b0


def fit_beam_gaussian(
    bin_centers: np.ndarray,
    mean_vals: np.ndarray,
    counts: np.ndarray | None = None,
    *,
    poly_order: int = 4,
    knee_k: float = 120.0,
    r_poly_deg: float = 1.2,
    refine_joint: bool = False,
    ax: Any = None,
    show: bool = True,
    title: str = "Beam profile",
) -> dict[str, Any]:
    """
    Piecewise fit: **Gaussian (core) and polynomial (outer)** in radial offset (deg).

    Separate models (fitted independently on disjoint bin sets):

        G(x) = A exp(-x²/(2σ²))
        P(x) = Σ_k c_k x^k

    **Fitting strategy**

    1. **Gaussian** — ``curve_fit`` on bins with ``y > knee_k`` only. Then
       **raise** ``A`` so the tallest data bin lies on or under the curve:
       ``A ≥ yₘₐₓ / exp(-xₘₐₓ²/(2σ²))`` at the global argmax of ``y``.
    2. **Polynomial** — bins with ``x > r_poly_deg``; ``polyfit`` directly on ``y``.

    **FWHM vs half-power radius:** for an isolated on-axis Gaussian,
    ``FWHM_deg = 2·√(2·ln 2)·σ ≈ 2.355·σ`` is the *full width* at half maximum.
    The *radius* from the beam centre to the half-power point is
    ``half_power_radius_deg = √(2·ln 2)·σ ≈ 1.18·σ`` (often what people read off a plot).

    ``refine_joint`` is kept for backward compatibility but ignored because the
    two fits are intentionally independent.

    Parameters
    ----------
    poly_order : int
        Polynomial degree for the outer radial region. Must be >= 0.
    knee_k : float
        Core vs shoulder split in mean K; default 120 K.
    r_poly_deg : float
        Radial split (deg): polynomial uses **x > r_poly_deg**. Default 1.2°.
    refine_joint : bool
        If True, joint ``curve_fit`` from sequential ``p0``. Default False.
    """

    mask = np.isfinite(mean_vals) & np.isfinite(bin_centers)
    x = np.asarray(bin_centers[mask], dtype=float)
    y = np.asarray(mean_vals[mask], dtype=float)

    if poly_order < 0:
        raise ValueError("poly_order must be >= 0")
    if r_poly_deg <= 0:
        raise ValueError("r_poly_deg must be positive")
    if x.size < max(3, poly_order + 2):
        raise ValueError(
            f"Not enough valid data points ({x.size}) for poly_order={poly_order}"
        )

    if counts is not None:
        c = np.asarray(counts[mask], dtype=float)
        sigma_weights = 1.0 / np.sqrt(np.maximum(c, 0.0) + 1e-6)
    else:
        sigma_weights = None

    x_fit = np.linspace(0.0, float(np.max(x)), 500)

    mask_core = y > knee_k
    mask_poly = x > r_poly_deg
    n_core = int(np.sum(mask_core))
    n_poly = int(np.sum(mask_poly))

    # --- (1) First Gaussian: y > knee_k ---
    x_c = x[mask_core]
    y_c = y[mask_core]
    if sigma_weights is not None:
        sw_c = sigma_weights[mask_core]
    else:
        sw_c = None

    if n_core >= 2:
        A0 = float(max(np.nanmax(y_c), 1e-6))
        i_pk = int(np.nanargmax(y_c))
        sigma0 = max(float(x_c[i_pk]) / 2.0, 1e-4)
        bounds_g = ([0.0, 1e-6], [np.inf, 180.0])
        try:
            (A, sigma), _ = curve_fit(
                _gaussian_only,
                x_c,
                y_c,
                p0=[A0, sigma0],
                sigma=sw_c,
                absolute_sigma=False,
                bounds=bounds_g,
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            (A, sigma), _ = curve_fit(
                _gaussian_only,
                x_c,
                y_c,
                p0=[A0, sigma0],
                sigma=sw_c,
                absolute_sigma=False,
                maxfev=20000,
            )
        A, sigma = float(A), float(sigma)
        idx_max = int(np.nanargmax(y))
        phi_pk = float(np.exp(-(float(x[idx_max]) ** 2) / (2.0 * sigma**2)))
        A = max(A, float(y[idx_max]) / max(phi_pk, 1e-15))
    else:
        warnings.warn(
            f"Fewer than 2 core points (y > {knee_k} K); fitting first Gaussian to full (x, y).",
            UserWarning,
            stacklevel=2,
        )
        A0 = float(max(np.nanmax(y), 1e-6))
        i_pk = int(np.nanargmax(y))
        sigma0 = max(float(x[i_pk]) / 2.0, 1e-4)
        try:
            (A, sigma), _ = curve_fit(
                _gaussian_only,
                x,
                y,
                p0=[A0, sigma0],
                sigma=sigma_weights,
                absolute_sigma=False,
                bounds=([0.0, 1e-6], [np.inf, 180.0]),
                maxfev=20000,
            )
        except (RuntimeError, ValueError):
            (A, sigma), _ = curve_fit(
                _gaussian_only,
                x,
                y,
                p0=[A0, sigma0],
                sigma=sigma_weights,
                absolute_sigma=False,
                maxfev=20000,
            )
        A, sigma = float(A), float(sigma)
        idx_max = int(np.nanargmax(y))
        phi_pk = float(np.exp(-(float(x[idx_max]) ** 2) / (2.0 * sigma**2)))
        A = max(A, float(y[idx_max]) / max(phi_pk, 1e-15))

    g1_x = _gaussian_only(x, A, sigma)

    # --- (2) Polynomial: x > r_poly_deg only, fitted independently on y ---
    if n_poly == 0:
        p_high = np.zeros(poly_order + 1, dtype=float)
        warnings.warn(
            f"No bins with x > {r_poly_deg}°; polynomial coefficients set to zero.",
            UserWarning,
            stacklevel=2,
        )
    else:
        x_p = x[mask_poly]
        y_p = y[mask_poly]
        eff_deg = min(poly_order, max(0, n_poly - 1))
        if eff_deg < poly_order:
            warnings.warn(
                f"Only {n_poly} outer points (x > {r_poly_deg}°); "
                f"fitting polynomial degree {eff_deg} (requested {poly_order}).",
                UserWarning,
                stacklevel=2,
            )
        if sigma_weights is not None:
            w_p = 1.0 / np.maximum(sigma_weights[mask_poly], 1e-12) ** 2
            p_high = np.polyfit(x_p, y_p, eff_deg, w=w_p)
        else:
            p_high = np.polyfit(x_p, y_p, eff_deg)
        if eff_deg < poly_order:
            p_high = np.concatenate(
                [np.zeros(poly_order - eff_deg, dtype=float), p_high]
            )

    if refine_joint:
        warnings.warn(
            "refine_joint=True ignored: Gaussian and polynomial are fit independently.",
            UserWarning,
            stacklevel=2,
        )

    poly_coeffs = p_high[::-1].astype(float)

    y_gauss = _gaussian_only(x_fit, A, sigma)
    y_poly = np.polyval(poly_coeffs[::-1], x_fit)

    sqrt_2_ln2 = np.sqrt(2.0 * np.log(2.0))
    hpbw = float(sqrt_2_ln2 * sigma)
    fwhm = float(2.0 * hpbw)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, s=12, label="Data (mean K)", zorder=3)
    ax.axvline(r_poly_deg, color="0.5", ls="-", lw=0.8, alpha=0.6)
    ax.plot(
        x_fit,
        y_gauss,
        "--",
        alpha=0.85,
        label=f"Gaussian: σ={sigma:.3f}°, HPBW={hpbw:.3f}°, FWHM={fwhm:.3f}°",
    )
    ax.plot(x_fit, y_poly, ":", alpha=0.85, label=f"Polynomial (deg {poly_order})")
    ax.set_xlabel("Radial offset (deg)")
    ax.set_ylabel("Mean (K)")
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
        "poly_coeffs": poly_coeffs,
        "poly_order": poly_order,
        "knee_k": knee_k,
        "r_poly_deg": r_poly_deg,
        "refine_joint": False,
        "covariance": None,
        "y_fit_gaussian": y_gauss,
        "y_fit_polynomial": y_poly,
        "x_fit": x_fit,
    }


def convolve_beam_with_fit(
    healmap: HealPixMap,
    beam_params: dict[str, Any],
    *,
    attribute: str | None = None,
    mask_uncovered: bool = True,
    normalize: bool = True,
    theta_max_deg: float = 5.0,
    lmax: int | None = None,
    ntheta: int | None = None,
) -> HealPixMap:
    """
    Harmonic smoothing with a **single** composite beam from ``beam_approximation``:

    - radial offset ``≤ r_poly_deg``: Gaussian (core),
    - radial offset ``> r_poly_deg``: polynomial (outer).

    The radial profile is turned into ``B_\\ell`` with ``healpy.beam2bl``; the
    window is then scaled so ``B_0 = 1`` for ``healpy.smoothing``.
    By default the radial beam is affine-mapped to **1 at 0°** and **0 at 5°**
    (same ``[0, 1]`` rule as ``plot_beam_approximation``); set ``normalize=False``
    to keep physical amplitudes from the fit.
    Width for the Gaussian uses ``FWHM_deg`` (preferred), ``half_power_radius_deg``
    (as FWHM in degrees), or ``sigma_deg``; ``σ = \\mathrm{FWHM} / (2\\sqrt{2\\ln 2})``.

    Parameters
    ----------
    healmap : HealPixMap
        Map with channel(s) to smooth.
    beam_params : dict
        Return value of ``fit_beam_gaussian``; must include ``A``, ``poly_coeffs``,
        ``r_poly_deg``, and one of ``FWHM_deg``, ``half_power_radius_deg``, or
        ``sigma_deg``.
    attribute : str or None
        Pol channel to smooth (e.g. ``\"AA_\"``). If ``None`` and there is
        exactly one entry in ``_channel_maps``, that channel is used; if there
        are multiple, you must pass ``attribute``.
    mask_uncovered : bool
        If True (default), pixels with no hits are set to ``UNSEEN`` before
        smoothing, then restored to 0 so they stay masked in ``plot``.
    normalize : bool
        If True (default), map ``beam_approximation`` to amplitude 1 at 0° and 0
        at 5° (values clipped to ``[0, 1]``) before ``beam2bl``, matching
        ``plot_beam_approximation``.
    theta_max_deg : float
        Upper limit (degrees) for discretizing the radial beam before ``beam2bl``.
        Capped at 5° to match the plot; default 5°.
    lmax : int or None
        Multipole cutoff for the beam window. Default ``3 * nside - 1``.
    ntheta : int or None
        Number of radial samples for ``beam2bl``. Default ``max(512, 2 * (lmax + 1))``.

    Returns
    -------
    HealPixMap
        New map with smoothed data; same ``nside`` and ``_hit_count`` as input.
        Use ``.plot(attribute=..., ra_dec_map=True, region=...)`` as with the
        original map.
    """
    hp.check_max_nside(healmap.nside)
    if lmax is None:
        lmax = 3 * healmap.nside - 1
    if ntheta is None:
        ntheta = max(512, 2 * (lmax + 1))

    theta_cap = min(float(theta_max_deg), 5.0)
    beam_bl = _beam_approximation_beam_bl(
        lmax,
        beam_params,
        theta_max_deg=theta_cap,
        ntheta=ntheta,
        normalize=normalize,
    )
    hit = np.asarray(healmap.get_hit_count(), dtype=float)

    attrs: list[str]
    if healmap._channel_maps:
        if attribute is not None:
            if attribute not in healmap._channel_maps:
                raise KeyError(
                    f"No map for {attribute!r}. Available: {list(healmap._channel_maps.keys())}"
                )
            attrs = [attribute]
        elif len(healmap._channel_maps) == 1:
            attrs = [next(iter(healmap._channel_maps.keys()))]
        else:
            raise ValueError(
                "Several channel maps exist; pass attribute= e.g. 'AA_' or 'BB_'."
            )
    else:
        if attribute is not None:
            raise ValueError("No per-channel maps on healmap; attribute must be None")
        attrs = []

    out = HealPixMap(healmap.nside)
    out.map = hit.copy()
    out._hit_count = hit.copy()
    out._map_std = np.full(out.npix, np.nan, dtype=float)

    def _smooth_one(arr: np.ndarray) -> np.ndarray:
        work = np.asarray(arr, dtype=float).copy()
        if mask_uncovered:
            work[hit <= 0] = hp.UNSEEN
        sm = hp.smoothing(
            work,
            beam_window=beam_bl,
            pol=False,
            lmax=lmax,
            verbose=False,
        )
        if mask_uncovered:
            sm = np.asarray(sm, dtype=float)
            sm[hit <= 0] = 0.0
        return sm

    if attrs:
        for attr in attrs:
            raw = healmap.get_map(attr)
            out._channel_maps[attr] = _smooth_one(raw)
            out._channel_stds[attr] = np.full(out.npix, np.nan, dtype=float)
    else:
        raw = healmap.get_map(None)
        out.map = _smooth_one(raw)
        out._map_std = np.full(out.npix, np.nan, dtype=float)

    return out
