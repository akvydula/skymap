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
from typing import TYPE_CHECKING, Any

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
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


def radial_offset_histogram(
    healmap: HealPixMap,
    source_ra_deg: float,
    source_dec_deg: float,
    attribute: str | None = None,
    bin_size_deg: float = 0.1,
    max_offset_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a radial profile from a HealPixMap: for each pixel with valid mean,
    compute angular offset (deg) from the source (RA, Dec), then bin by offset
    and average the mean (K) in each bin.

    Parameters
    ----------
    healmap : HealPixMap
        Map filled with mean in K (e.g. from fill_from_pointing_data).
    source_ra_deg, source_dec_deg : float
        Source position in degrees (e.g. from get_source_radec).
    attribute : str or None
        PolChannel to use (e.g. 'AA_', 'BB_'). If None and map has no channel
        maps, the main map is used; if channel maps exist, one must be specified.
    bin_size_deg : float
        Bin width for radial offset in degrees (default 0.1).
    max_offset_deg : float or None
        If set, only include pixels with offset <= this (degrees).

    Returns
    -------
    bin_centers : np.ndarray
        Radial offset in degrees (bin centers).
    mean_k : np.ndarray
        Mean temperature in K in each radial bin.
    counts : np.ndarray
        Number of pixels in each bin.
    """
    nside = healmap.nside
    npix = healmap.npix
    map_vals = healmap.get_map(attribute)
    hit_count = healmap.get_hit_count()

    # Pixels with at least one hit and finite mean
    valid = (hit_count > 0) & np.isfinite(map_vals)
    ipix = np.where(valid)[0]
    if ipix.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Angular distance from source (degrees)
    vec_src = hp.ang2vec(
        np.radians(90.0 - source_dec_deg),
        np.radians(source_ra_deg),
    )
    thetas, phis = hp.pix2ang(nside, ipix)
    vecs = hp.ang2vec(thetas, phis)
    offset_rad = hp.rotator.angdist(vec_src, vecs)
    offset_deg = np.degrees(offset_rad)
    values = np.asarray(map_vals[ipix], dtype=float)

    if max_offset_deg is not None:
        mask = offset_deg <= max_offset_deg
        offset_deg = offset_deg[mask]
        values = values[mask]

    # Bin by offset_deg
    max_off = float(np.max(offset_deg)) if offset_deg.size else 0.0
    n_bins = max(1, int(np.ceil(max_off / bin_size_deg)))
    bin_edges = np.linspace(0, n_bins * bin_size_deg, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_k = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (offset_deg >= lo) & (offset_deg < hi)
        if np.any(in_bin):
            mean_k[i] = np.nanmean(values[in_bin])
            counts[i] = int(np.sum(in_bin))

    return bin_centers, mean_k, counts


def radial_offset_histogram_from_arrays(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    values: np.ndarray,
    source_ra_deg: float,
    source_dec_deg: float,
    bin_size_deg: float = 0.1,
    max_offset_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the radial histogram from pre-extracted (ra, dec, values), e.g. from
    HealPixMap.get_map_radec_values(). x-axis = radial offset (deg) from source,
    y-axis = mean value (K) per bin.

    Parameters
    ----------
    ra_deg, dec_deg : np.ndarray
        RA and Dec in degrees for each point (same length).
    values : np.ndarray
        Value (e.g. mean in K) for each point.
    source_ra_deg, source_dec_deg : float
        Source position in degrees.
    bin_size_deg : float
        Bin width for radial offset (default 0.1).
    max_offset_deg : float or None
        If set, only include points with offset <= this (degrees).

    Returns
    -------
    bin_centers : np.ndarray
        Radial offset in degrees (bin centers).
    mean_k : np.ndarray
        Mean value in each radial bin.
    counts : np.ndarray
        Number of points per bin.
    """
    ra_deg = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec_deg = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if ra_deg.size != dec_deg.size or ra_deg.size != values.size:
        raise ValueError("ra_deg, dec_deg, and values must have the same length")
    if ra_deg.size == 0:
        return np.array([]), np.array([]), np.array([])

    vec_src = hp.ang2vec(
        np.radians(90.0 - source_dec_deg),
        np.radians(source_ra_deg),
    )
    thetas = np.radians(90.0 - dec_deg)
    phis = np.radians(ra_deg)
    vecs = hp.ang2vec(thetas, phis)
    offset_rad = hp.rotator.angdist(vec_src, vecs)
    offset_deg = np.degrees(offset_rad)

    if max_offset_deg is not None:
        mask = offset_deg <= max_offset_deg
        offset_deg = offset_deg[mask]
        values = values[mask]

    if offset_deg.size == 0:
        return np.array([]), np.array([]), np.array([])

    max_off = float(np.max(offset_deg))
    n_bins = max(1, int(np.ceil(max_off / bin_size_deg)))
    bin_edges = np.linspace(0, n_bins * bin_size_deg, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_k = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (offset_deg >= lo) & (offset_deg < hi)
        if np.any(in_bin):
            mean_k[i] = np.nanmean(values[in_bin])
            counts[i] = int(np.sum(in_bin))

    return bin_centers, mean_k, counts


def compute_beam_histogram_from_arrays(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    values: np.ndarray,
    source_name: str | None = None,
    source_ra_deg: float | None = None,
    source_dec_deg: float | None = None,
    bin_size_deg: float = 0.1,
    calibrators_path: Path | str | None = None,
    max_offset_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the radial beam histogram from pre-extracted map data (e.g. from
    HealPixMap.get_map_radec_values()). Source position is either from
    source_name (calibrators.dat) or from (source_ra_deg, source_dec_deg).

    Parameters
    ----------
    ra_deg, dec_deg, values : np.ndarray
        Map data as returned by healmap.get_map_radec_values(attribute).
    source_name : str or None
        Calibrator name to look up (e.g. '3C286'). Ignored if source_ra_deg/source_dec_deg given.
    source_ra_deg, source_dec_deg : float or None
        Source position in degrees. If both given, used instead of source_name.
    bin_size_deg : float
        Bin width in degrees (default 0.1).
    calibrators_path : path or None
        Path to calibrator catalogue when using source_name.
    max_offset_deg : float or None
        If set, only include points within this radius (deg).

    Returns
    -------
    bin_centers, mean_k, counts
        Same as radial_offset_histogram_from_arrays.
    """
    if source_ra_deg is not None and source_dec_deg is not None:
        src_ra, src_dec = source_ra_deg, source_dec_deg
    elif source_name is not None:
        src_ra, src_dec = get_source_radec(source_name, path=calibrators_path)
    else:
        raise ValueError("Provide either source_name or (source_ra_deg, source_dec_deg)")
    return radial_offset_histogram_from_arrays(
        ra_deg,
        dec_deg,
        values,
        src_ra,
        src_dec,
        bin_size_deg=bin_size_deg,
        max_offset_deg=max_offset_deg,
    )


def compute_beam_histogram(
    healmap: HealPixMap,
    source_name: str,
    attribute: str | None = None,
    bin_size_deg: float = 0.1,
    calibrators_path: Path | str | None = None,
    max_offset_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience: get source RA/Dec from calibrators.dat, then compute
    radial offset histogram (x = offset deg, y = mean K).

    Returns
    -------
    bin_centers, mean_k, counts
        Same as radial_offset_histogram.
    """
    ra_deg, dec_deg = get_source_radec(source_name, path=calibrators_path)
    return radial_offset_histogram(
        healmap,
        ra_deg,
        dec_deg,
        attribute=attribute,
        bin_size_deg=bin_size_deg,
        max_offset_deg=max_offset_deg,
    )


def plot_radial_histogram(
    bin_centers: np.ndarray,
    mean_k: np.ndarray,
    counts: np.ndarray | None = None,
    *,
    title: str | None = "Radial profile",
    xlabel: str = "Radial offset (deg)",
    ylabel: str = "Mean (K)",
    mask_nan: bool = True,
    ax: Any = None,
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Plot the radial histogram: x = radial offset (deg), y = mean temperature (K).

    Parameters
    ----------
    bin_centers : np.ndarray
        Radial offset bin centers in degrees.
    mean_k : np.ndarray
        Mean temperature in K per bin.
    counts : np.ndarray or None
        Optional pixel count per bin (not plotted by default).
    title : str or None
        Axes title.
    xlabel, ylabel : str
        Axis labels.
    mask_nan : bool
        If True (default), skip bins with NaN mean when drawing the histogram.
    ax : matplotlib axes or None
        If given, plot into this axes; otherwise create a new figure.
    show : bool
        If True (default), call plt.show().
    **kwargs
        Passed to plt.hist() (e.g. color, edgecolor, label, alpha).

    Returns
    -------
    ax
        The matplotlib axes used.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if len(bin_centers) == 0:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
        return ax
    if mask_nan:
        valid = np.isfinite(mean_k)
        x, w = bin_centers[valid], mean_k[valid]
    else:
        x, w = bin_centers, mean_k
    # Reconstruct bin edges (uniform spacing) for plt.hist
    if len(bin_centers) > 1:
        width = float(bin_centers[1] - bin_centers[0])
    else:
        width = 0.1
    bin_edges = np.concatenate(
        [[bin_centers[0] - width / 2], bin_centers + width / 2]
    )
    ax.hist(x, bins=bin_edges, weights=w, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if show:
        plt.show()
    return ax


def plot_beam_histogram(
    healmap: HealPixMap,
    source_name: str,
    attribute: str | None = None,
    bin_size_deg: float = 0.1,
    calibrators_path: Path | str | None = None,
    max_offset_deg: float | None = None,
    title: str | None = None,
    ax: Any = None,
    show: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """
    Compute the radial beam histogram (source from calibrators.dat) and plot it.

    Parameters
    ----------
    healmap : HealPixMap
        Map filled with mean in K.
    source_name : str
        Calibrator name (e.g. '3C286', '3C48').
    attribute : str or None
        PolChannel to use (e.g. 'AA_', 'BB_').
    bin_size_deg : float
        Bin width in degrees (default 0.1).
    calibrators_path : path or None
        Path to calibrator catalogue; default is calibrators.dat next to this module.
    max_offset_deg : float or None
        If set, only include pixels within this radius (deg).
    title : str or None
        Plot title; default is "{source_name} radial profile".
    ax : matplotlib axes or None
        If given, plot into this axes.
    show : bool
        If True (default), call plt.show().
    **plot_kwargs
        Passed to plot_radial_histogram / plt.hist().

    Returns
    -------
    ax
        The matplotlib axes used.
    """
    bin_centers, mean_k, counts = compute_beam_histogram(
        healmap,
        source_name,
        attribute=attribute,
        bin_size_deg=bin_size_deg,
        calibrators_path=calibrators_path,
        max_offset_deg=max_offset_deg,
    )
    if title is None:
        title = f"{source_name} radial profile"
    return plot_radial_histogram(
        bin_centers,
        mean_k,
        counts,
        title=title,
        ax=ax,
        show=show,
        **plot_kwargs,
    )


def plot_beam_histogram_from_arrays(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    values: np.ndarray,
    source_name: str | None = None,
    source_ra_deg: float | None = None,
    source_dec_deg: float | None = None,
    bin_size_deg: float = 0.1,
    calibrators_path: Path | str | None = None,
    max_offset_deg: float | None = None,
    title: str | None = None,
    ax: Any = None,
    show: bool = True,
    **plot_kwargs: Any,
) -> Any:
    """
    Compute the radial beam histogram from pre-extracted map data and plot it.
    Map data can be from HealPixMap.get_map_radec_values(attribute).

    Parameters
    ----------
    ra_deg, dec_deg, values : np.ndarray
        Map data (e.g. from healmap.get_map_radec_values("AA_")).
    source_name : str or None
        Calibrator name (e.g. '3C286'). Ignored if source_ra_deg/source_dec_deg given.
    source_ra_deg, source_dec_deg : float or None
        Source position in degrees. If both given, used instead of source_name.
    bin_size_deg, calibrators_path, max_offset_deg
        Passed to compute_beam_histogram_from_arrays.
    title : str or None
        Plot title; default is "{source_name} radial profile" or "Radial profile".
    ax, show, **plot_kwargs
        Passed to plot_radial_histogram.

    Returns
    -------
    ax
        The matplotlib axes used.
    """
    bin_centers, mean_k, counts = compute_beam_histogram_from_arrays(
        ra_deg,
        dec_deg,
        values,
        source_name=source_name,
        source_ra_deg=source_ra_deg,
        source_dec_deg=source_dec_deg,
        bin_size_deg=bin_size_deg,
        calibrators_path=calibrators_path,
        max_offset_deg=max_offset_deg,
    )
    if title is None and source_name is not None:
        title = f"{source_name} radial profile"
    return plot_radial_histogram(
        bin_centers,
        mean_k,
        counts,
        title=title,
        ax=ax,
        show=show,
        **plot_kwargs,
    )
