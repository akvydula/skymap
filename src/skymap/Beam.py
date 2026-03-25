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


def _gaussian_peak(theta: np.ndarray, A: float, sigma: float) -> np.ndarray:
    """Gaussian bump only (no constant): peak A at theta=0."""
    return A * np.exp(-(theta**2) / (2.0 * sigma**2))


def plot_beam_gaussian(
    bin_centers: np.ndarray,
    mean_vals: np.ndarray,
    counts: np.ndarray | None = None,
    *,
    baseline_k: float | None = None,
    ax: Any = None,
    show: bool = True,
    title: str = "Beam profile",
) -> dict[str, Any]:
    """
    Fit a Gaussian bump to radial-offset bins vs mean (K), always relative to the
    mean of ``mean_vals``: fit ``A*exp(-x**2/(2*sigma**2))`` to ``mean_k - mean(mean_k)``,
    then plot ``bump + baseline``.

    Parameters
    ----------
    baseline_k : float or None
        If set, use this as the baseline instead of ``mean(mean_vals)``.
    """
    mask = np.isfinite(mean_vals) & np.isfinite(bin_centers)
    x = np.asarray(bin_centers[mask], dtype=float)
    y = np.asarray(mean_vals[mask], dtype=float)

    if x.size < 5:
        raise ValueError("Not enough valid data points to fit")

    if counts is not None:
        c = np.asarray(counts[mask], dtype=float)
        sigma_weights = 1.0 / np.sqrt(np.maximum(c, 0.0) + 1e-6)
    else:
        sigma_weights = None

    x_fit = np.linspace(0.0, float(np.max(x)), 500)

    baseline = float(baseline_k) if baseline_k is not None else float(np.nanmean(y))
    y_rel = y - baseline
    A0 = float(np.nanmax(y_rel) - np.nanmin(y_rel))
    if A0 <= 0:
        A0 = float(np.nanmax(y_rel))
    sigma0 = max(float(x[np.argmax(y_rel)]) / 2.0, 1e-3)
    p0 = [A0, sigma0]
    popt, pcov = curve_fit(
        _gaussian_peak,
        x,
        y_rel,
        p0=p0,
        sigma=sigma_weights,
        absolute_sigma=False,
        maxfev=10000,
    )
    A, sigma = float(popt[0]), float(popt[1])
    y_fit = _gaussian_peak(x_fit, A, sigma) + baseline
    baseline_used = baseline

    fwhm = 2.355 * sigma

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=10, label="Data (mean per pixel [K])")
    ax.plot(
        x_fit,
        y_fit,
        label=f"Gaussian fit \nσ={sigma:.3f}°, FWHM={fwhm:.3f}°",
    )
    ax.axhline(baseline_used, color="gray", ls="--", lw=1, alpha=0.7, label="Baseline")
    ax.set_xlabel("Radial offset (deg)")
    ax.set_ylabel("Mean (K)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    if show:
        plt.show()

    return {
        "A": A,
        "sigma_deg": sigma,
        "FWHM_deg": fwhm,
        "baseline_K": baseline_used,
        "covariance": pcov,
    }


def convolve_beam_gaussian(
    healmap: HealPixMap,
    gaussian_params: dict[str, Any],
    *,
    attribute: str | None = None,
    mask_uncovered: bool = True,
) -> HealPixMap:
    """
    Smooth a healpix beam map with a Gaussian beam
    Convolution is done with ``healpy.smoothing`` (Gaussian beam in harmonic
    space, symmetric on the sphere).

    Parameters
    ----------
    healmap : HealPixMap
        Map with channel(s) to smooth
    gaussian_params : dict
        Must include ``FWHM_deg`` and/or ``sigma_deg`` (from ``plot_beam_gaussian``).
    attribute : str or None
        Pol channel to smooth (e.g. ``\"AA_\"``). If ``None`` and there is
        exactly one entry in ``_channel_maps``, that channel is used; if there
        are multiple, you must pass ``attribute``.
    mask_uncovered : bool
        If True (default), pixels with no hits are set to ``UNSEEN`` before
        smoothing, then restored to 0 so they stay masked in ``plot``.

    Returns
    -------
    HealPixMap
        New map with smoothed data; same ``nside`` and ``_hit_count`` as input.
        Use ``.plot(attribute=..., ra_dec_map=True, region=...)`` as with the
        original map.
    """
    if "FWHM_deg" in gaussian_params:
        fwhm_deg = float(gaussian_params["FWHM_deg"])
    elif "sigma_deg" in gaussian_params:
        fwhm_deg = 2.355 * float(gaussian_params["sigma_deg"])
    else:
        raise ValueError("gaussian_params must contain 'FWHM_deg' or 'sigma_deg'")

    fwhm_rad = np.radians(fwhm_deg)
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
        sm = hp.smoothing(work, fwhm=fwhm_rad, verbose=False)
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
