'''
ploting utils functions
'''

from skymap.io import CalData, HDF5Data, CAL_POL_NAMES, PointingData
from skymap import io

from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
from datetime import datetime, timedelta, timezone as dt_timezone, tzinfo
from zoneinfo import ZoneInfo
import matplotlib.dates as mdates
import astropy.units as u
from astropy.time import Time
from typing import Literal


def plot_cal_data(cal_data: CalData, attribute: str = 'gain', save_path: str | Path | None = None) -> None:
    """
    Plot the calibration data.

    Parameters
    ----------
    cal_data : CalData
        CalData object containing frequency, te, and gain data.
 
    attribute : str, default='gain'
        Attribute to plot. Options:
        - 'gain': Plot gain
        - 'te': Plot Te. Default is 'gain'.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    """
    if attribute not in ['gain', 'te']:
        raise ValueError(f"Invalid attribute: {attribute}. Must be 'gain' or 'te'.")
    

    channels = getattr(cal_data, attribute)  # PolChannels (gain or te)
    labels = ['AA*', 'BB*', 'CC*', 'DD*', 'AB*', 'BC*', 'CD*', 'AC*', 'BD*', 'AD*']
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()
    for idx, name in enumerate(CAL_POL_NAMES):
        ax = axes[idx]
        ax.plot(cal_data.freq / 1e6, getattr(channels, name), label=labels[idx])
        ax.set_xlabel('Frequency (MHz)', fontsize=10)
        ax.legend(fontsize=8)
        
    fig.suptitle(f'{attribute.title()}', fontsize=10)
        
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()


def get_plot_data(
    data: HDF5Data,
    calibrated: bool = False,
    data_type: str = "mag",
    attribute: str | None = None,
    spec_attr: str | None = None,
) -> dict:
    """
    Select and process plot data from HDF5Data. Handles calibrated/spec selection,
    data_type (real, imag, mag, phase), units (K for calibrated), and optional pointing.

    Parameters
    ----------
    data : HDF5Data
        HDF5Data object containing frequency, time, and spec data.
    calibrated : bool, default=False
        If True, use calibrated_spec; otherwise use spec.
    data_type : str, default="mag"
        One of "real", "imag", "mag", "phase".
    attribute : str or None, default=None
        Single correlation to plot (e.g. "AA_"); if None, all available.
    pointing_data : PointingData or None, default=None
        If provided, match data with pointing and include ra, dec, el, az.

    Returns
    -------
    dict
        Keys: plot_data, to_plot, freq, time, calibrated, cbar_label, plot_order.
        If pointing was matched: ra, dec, el, az.
    """
    if spec_attr is not None:
        spec_source = getattr(data, spec_attr, None)
        if spec_source is None:
            raise ValueError(f"data has no {spec_attr!r} to plot")
        # If user explicitly asked for calibrated mean/std, label as calibrated.
        calibrated = calibrated or spec_attr.startswith("calibrated_")
    else:
        if data.spec is None:
            candidates = [n for n in ("calibrated_spec_mean", "calibrated_spec_std", "spec_mean", "spec_std") if getattr(data, n, None) is not None]
            hint = f" (did you mean spec_attr={candidates[0]!r}?)" if candidates else ""
            raise ValueError("HDF5Data object must have spec data to plot" + hint)
        if calibrated and getattr(data, "calibrated_spec", None) is None:
            candidates = [n for n in ("calibrated_spec_mean", "calibrated_spec_std") if getattr(data, n, None) is not None]
            hint = f" (did you mean spec_attr={candidates[0]!r}?)" if candidates else ""
            raise ValueError("HDF5Data must have calibrated_spec to plot with calibrated=True" + hint)
        spec_source = data.calibrated_spec if calibrated else data.spec
    correlations = {name: getattr(spec_source, name) for name in CAL_POL_NAMES}
    plot_order = CAL_POL_NAMES

    if attribute is not None and attribute not in plot_order:
        raise ValueError(f"attribute must be None or one of {plot_order}, got {attribute!r}")

    data_type_lower = data_type.lower()
    if data_type_lower not in ("real", "imag", "mag", "magnitude", "phase"):
        raise ValueError(f"data_type must be one of 'real', 'imag', 'mag', 'phase', got {data_type!r}")

    plot_data = {}
    for name, corr_data in correlations.items():
        if corr_data is None:
            continue
        if getattr(corr_data, "unit", None) is not None:
            corr_data = corr_data.to(u.K).value
        if np.iscomplexobj(corr_data):
            if data_type_lower in ("mag", "magnitude"):
                plot_data[name] = np.abs(corr_data)
            elif data_type_lower == "real":
                plot_data[name] = np.real(corr_data)
            elif data_type_lower == "imag":
                plot_data[name] = np.imag(corr_data)
            elif data_type_lower == "phase":
                plot_data[name] = np.angle(corr_data)
        else:
            plot_data[name] = corr_data

    if attribute is not None:
        if attribute not in plot_data:
            raise ValueError(f"No data for attribute {attribute!r} (may be uncalibrated)")
        to_plot = [attribute]
    else:
        to_plot = [n for n in plot_order if n in plot_data]
        if not to_plot:
            raise ValueError("No correlation data to plot (calibrated_spec may have no channels set)")

    if spec_attr in ("calibrated_spec_mean", "calibrated_spec_std"):
        cbar_label = "Temperature (K) mean" if spec_attr.endswith("_mean") else "Temperature (K) std"
    elif spec_attr in ("spec_mean", "spec_std"):
        cbar_label = "Raw measurements mean" if spec_attr.endswith("_mean") else "Raw measurements std"
    else:
        cbar_label = "Temperature (K)" if calibrated else "Raw measurements"
    result = {
        "plot_data": plot_data,
        "to_plot": to_plot,
        "freq": data.freq,
        "time": data.time,
        "calibrated": calibrated,
        "cbar_label": cbar_label,
        "plot_order": plot_order,
    }
    if getattr(data, "ra", None) is not None:
        result["ra"] = data.ra
    if getattr(data, "dec", None) is not None:
        result["dec"] = data.dec
    if getattr(data, "el", None) is not None:
        result["el"] = data.el
    if getattr(data, "az", None) is not None:
        result["az"] = data.az
    return result


def _parse_display_timezone(time_zone: str) -> tuple[tzinfo, str]:
    """
    Map user string to a tzinfo and a short axis label.

    ``EST`` is fixed UTC−5 (no daylight saving). For US Eastern with DST, use
    ``America/New_York``.
    """
    s = time_zone.strip()
    if s.upper() == "EST":
        return dt_timezone(timedelta(hours=-5)), "EST (UTC−5)"
    try:
        tz = ZoneInfo(s)
        return tz, s
    except Exception as e:
        raise ValueError(
            f"time_zone={time_zone!r} is invalid. Use 'EST' for fixed UTC−5, or an IANA name "
            f"such as 'America/New_York' for US Eastern with DST."
        ) from e


def _waterfall_times_in_zone(time: np.ndarray, time_zone: str) -> tuple[np.ndarray, str]:
    """Interpret ``time`` as UTC and return 1-D object array of timezone-aware datetimes."""
    tz, tz_label = _parse_display_timezone(time_zone)
    dt_utc = Time(time).to_datetime(timezone=dt_timezone.utc)
    dt_arr = np.atleast_1d(np.asarray(dt_utc))
    out = np.empty(dt_arr.shape, dtype=object)
    for idx in np.ndindex(dt_arr.shape):
        out[idx] = dt_arr[idx].astimezone(tz)
    return out.reshape(time.shape), tz_label


def plot_waterfall(
    data: HDF5Data,
    calibrated: bool = False,
    figsize: tuple[int, int] = (15, 5),
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    clim_per_plot: dict[str, tuple[float, float]] | None = None,
    data_type: str = "mag",
    attribute: str | None = None,
    spec_attr: str | None = None,
    save_path: str | Path | None = None,
    time_zone: str | None = None,
) -> None:
    """
    Create a waterfall plot of spectrograph data: either a 2x5 grid of all ten
    channel pairs or a single plot for one attribute.

    Usage:
    # 2x5 grid of magnitude for all 10 correlations
    plot_waterfall(data, data_type="mag", attribute=None, calibrated=False)

    # Single plot of AA_ phase
    plot_waterfall(data, data_type="phase", attribute="AA_")

    # 2x5 grid of real part
    plot_waterfall(data, data_type="real")

    Parameters
    ----------
    data : HDF5Data
        HDF5Data object containing frequency, time, and spec data.
    calibrated: bool, default=False
        If True, plot the calibrated data.
    figsize : tuple[int, int], default=(15, 5)
        Figure size (width, height) in inches. Used when attribute is None (2x5 grid).
    cmap : str, default="viridis"
        Colormap to use for the plots.
    clim : tuple[float, float], optional
        Global color limits (vmin, vmax) applied to all plots.
        If None, computed from the plotted data. Overridden by clim_per_plot when set.
    clim_per_plot : dict[str, tuple[float, float]], optional
        Per-plot color limits. Keys are attribute names ('AA_', 'BB_', ...),
        values are (vmin, vmax) tuples. Overrides clim for those plots.
    data_type : str, default="mag"
        Quantity to plot from the complex data. One of:
        - "real": Re(data)
        - "imag": Im(data)
        - "mag": |data| (magnitude)
        - "phase": phase in radians
    attribute : str or None, default=None
        Which of the 10 spectrograph measurements to plot. Must be one of
        plot_order: 'AA_', 'BB_', 'CC_', 'DD_', 'AB_', 'BC_', 'CD_', 'AC_', 'BD_', 'AD_'.
        If None, plot all ten in a 2x5 grid. If a string matching one of these,
        plot only that attribute as a single figure.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    time_zone : str or None, default=None
        If None, the y-axis uses ``data.time`` as stored. Typical HDF5 data from
        :func:`skymap.io.read_obs_hdf5` uses **naive UTC** datetimes; tick labels are
        then UTC wall time (matplotlib has no zone suffix). If set, times are interpreted
        as UTC and converted for display: use ``\"EST\"`` for fixed UTC−5 (no DST), or an
        IANA name such as ``\"America/New_York\"`` for US Eastern with daylight saving.
    """
    d = get_plot_data(data, calibrated=calibrated, data_type=data_type, attribute=attribute, spec_attr=spec_attr)
    plot_data = d["plot_data"]
    to_plot = d["to_plot"]
    freq = d["freq"]
    time = d["time"]
    cbar_label = d["cbar_label"]

    if attribute is not None:
        fig, axes_flat = plt.subplots(1, 1, figsize=(5, 5))
        axes_flat = np.atleast_1d(axes_flat)
    else:
        fig, axes_2d = plt.subplots(2, 5, figsize=figsize)
        axes_flat = axes_2d.flatten()

    # Global color limits from all data being plotted
    if clim is None and clim_per_plot is None:
        all_vals = np.concatenate([plot_data[name].flatten() for name in to_plot])
        clim = (np.nanmin(all_vals), np.nanmax(all_vals))

    is_datetime = (
        isinstance(time, np.ndarray)
        and len(time) > 0
        and (isinstance(time[0], datetime) or isinstance(time[0], np.datetime64))
    )
    time_y_label = "Time"
    tz_for_formatter: tzinfo | None = None
    if is_datetime and time_zone is not None and len(time) > 0:
        time, tz_axis_label = _waterfall_times_in_zone(time, time_zone)
        time_y_label = f"Time ({tz_axis_label})"
        tz_for_formatter, _ = _parse_display_timezone(time_zone)
    # Same x/y setup for all subplots: frequency vs time
    F, T = np.meshgrid(freq, time, indexing="ij")
    # With indexing='ij', F and T are (len(freq), len(time)); data is (n_time, n_freq), so use Z.T
    for idx, name in enumerate(to_plot):
        arr = plot_data[name]
        if arr.shape != (len(time), len(freq)):
            raise ValueError(
                f"Correlation '{name}' shape {arr.shape} does not match "
                f"expected ({len(time)}, {len(freq)})"
            )
        Z = arr.T  # (n_freq, n_time) to match F, T from meshgrid(..., indexing='ij')
        ax = axes_flat[idx]
        if clim_per_plot is not None and name in clim_per_plot:
            vmin, vmax = clim_per_plot[name]
        elif clim is not None:
            vmin, vmax = clim
        else:
            vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        im = ax.pcolormesh(F, T, Z, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Frequency (MHz)", fontsize=10)
        if is_datetime:
            ax.set_ylabel(time_y_label, fontsize=10)
            # Axis values are UTC instants; DateFormatter defaults to rcParams timezone (often UTC),
            # so tick text ignores wall-clock conversion unless *tz* is set.
            if tz_for_formatter is not None:
                y_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M:%S", tz=tz_for_formatter)
                ax.yaxis.set_major_locator(mdates.AutoDateLocator(tz=tz_for_formatter))
            else:
                y_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M:%S")
                ax.yaxis.set_major_locator(mdates.AutoDateLocator())
            ax.yaxis.set_major_formatter(y_fmt)
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            ax.set_ylabel("Time (s)", fontsize=10)
        ax.set_title(f"Correlation: {name}", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label=cbar_label).ax.tick_params(labelsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def _freq_avg_pointing_scatter_context(
    data: HDF5Data,
    pointing_data: PointingData | None,
    calibrated: bool,
    data_type: str,
    attribute: str | None,
    use_std: bool,
    AzEL_map: bool,
    freq_avg: bool,
) -> dict:
    """
    Shared setup for sky plots: match pointing, pick pol channel, frequency-average (or
    center channel), return axis arrays and labels.
    """
    if pointing_data is None:
        if getattr(data, "ra", None) is None or getattr(data, "dec", None) is None:
            raise ValueError("Pointing (ra, dec, az, el) required; provide pointing_data or pre-matched data")
        ra = data.ra
        dec = data.dec
        az = data.az
        el = data.el
    else:
        data = io.match_data_and_pointing(data, pointing_data)
        ra = data.ra
        dec = data.dec
        az = data.az
        el = data.el

    spec_mean = io.get_pol_source(data, kind="mean")
    spec_std = io.get_pol_source(data, kind="std")
    if spec_mean is not None and spec_std is not None:
        source = spec_std if use_std else spec_mean
        plot_data = {
            name: getattr(source, name)
            for name in CAL_POL_NAMES
            if getattr(source, name, None) is not None
        }
        if not plot_data:
            raise ValueError("No mean/std channels available on data")
        to_plot = [attribute] if attribute and attribute in plot_data else list(plot_data.keys())
        if attribute is not None and attribute not in plot_data:
            raise ValueError(f"attribute {attribute!r} not in mean/std data (available: {list(plot_data.keys())})")
        freq = data.freq
        time = data.time
        value_label = "Temperature (K) std" if use_std else "Temperature (K) mean"
    else:
        d = get_plot_data(data, calibrated=calibrated, data_type=data_type, attribute=attribute)
        plot_data = d["plot_data"]
        to_plot = d["to_plot"]
        freq = d["freq"]
        time = d["time"]
        value_label = d["cbar_label"]

    if AzEL_map:
        x_axis, y_axis = az, el
        x_label, y_label = "Azimuth (deg)", "Elevation (deg)"
    else:
        x_axis, y_axis = ra, dec
        x_label, y_label = "RA (J2000)", "Dec (J2000)"

    pol_name = to_plot[0] if to_plot else list(plot_data.keys())[0]
    arr = plot_data[pol_name]
    if freq_avg:
        arr_plot = np.nanmean(arr, axis=1)
    else:
        arr_plot = arr[:, len(freq) // 2]

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_label": x_label,
        "y_label": y_label,
        "arr_plot": arr_plot,
        "pol_name": pol_name,
        "value_label": value_label,
        "data_type": data_type,
        "spec_mean": spec_mean,
        "use_std": use_std,
    }


def _scatter_sizes_from_values(z: np.ndarray, s_max: float, s_min_frac: float = 0.2) -> np.ndarray:
    """Map scalar values to marker areas for scatter (linear, nan-safe)."""
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)
    if not np.any(finite):
        return np.full(z.shape, s_max, dtype=float)
    lo = np.nanmin(z)
    hi = np.nanmax(z)
    span = hi - lo
    if span <= 0:
        return np.full(z.shape, s_max, dtype=float)
    u = (z - lo) / span
    u = np.clip(np.where(finite, u, 0.0), 0.0, 1.0)
    return s_max * (s_min_frac + (1.0 - s_min_frac) * u)


def plot_waterfall_with_pointing(
    data: HDF5Data,
    pointing_data: PointingData | None = None,
    calibrated: bool = False,
    data_type: str = "mag",
    attribute: str | None = None,
    use_std: bool = False,
    AzEL_map: bool = False,
    RADEC_map: bool = True,
    freq_avg: bool = True,
    save_path: str | Path | None = None,
    size: float = 1,
    clim: tuple[float, float] = [50, 200],
) -> None:
    """
    Plot the data with pointing (ra/dec or az/el).

    Uses the same data selection as plot_waterfall: calibrated, data_type, attribute.
    When data has calibrated_spec_mean/calibrated_spec_std (from match_data_and_pointing),
    use_std selects which to plot for each of the 10 pols.
    Parameters
    ----------
    data : HDF5Data
        HDF5Data object containing frequency, time, and spec data (or matched data with
        calibrated_spec_mean / calibrated_spec_std).
    pointing_data : PointingData or None, default=None
        Pointing data. If None, ra/dec/az/el must already be on data.
    calibrated : bool, default=False
        If True, plot calibrated data (K).
    data_type : str, default="mag"
        One of "real", "imag", "mag", "phase" (ignored when using mean/std).
    attribute : str or None, default=None
        Which correlation to plot (e.g. "AA_", "AB_"). If None, uses first available.
    use_std : bool, default=False
        If True and data has calibrated_spec_mean/std (or spec_mean/spec_std), plot std
        instead of mean for each pol.
    AzEL_map : bool, default=False
        If True, use az/el for axes; else use ra/dec.
    RADEC_map : bool, default=True
        Unused; kept for backwards compatibility.
    freq_avg : bool, default=True
        If True, average over frequency.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    size : float, default=1
        Size of the scatter points.
    clim : tuple[float, float], optional
        Color limits for the scatter plot.
    """
    ctx = _freq_avg_pointing_scatter_context(
        data,
        pointing_data,
        calibrated,
        data_type,
        attribute,
        use_std,
        AzEL_map,
        freq_avg,
    )
    plt.figure(figsize=(8, 6))
    plt.scatter(
        ctx["x_axis"],
        ctx["y_axis"],
        c=ctx["arr_plot"],
        cmap="viridis",
        s=size,
        vmin=clim[0],
        vmax=clim[1],
    )
    plt.colorbar(label=ctx["value_label"])
    plt.xlabel(ctx["x_label"])
    plt.ylabel(ctx["y_label"])
    spec_mean = ctx["spec_mean"]
    stat_suffix = (
        " (std)"
        if (ctx["use_std"] and spec_mean is not None)
        else (" (mean)" if (spec_mean is not None) else "")
    )
    plt.title(f"{ctx['pol_name']} ({ctx['data_type']}){stat_suffix}".strip())
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_freq_avg_on_sky(
    data: HDF5Data,
    pointing_data: PointingData | None = None,
    calibrated: bool = False,
    data_type: str = "mag",
    attribute: str | None = None,
    use_std: bool = False,
    AzEL_map: bool = False,
    freq_avg: bool = True,
    save_path: str | Path | None = None,
    size: float = 36.0,
    marker_color: str = "C0",
    edgecolors: str | None = "k",
    linewidths: float = 0.35,
    encode_value: str = "size",
) -> None:
    """
    Frequency-averaged spectrum (or single channel) vs pointing, **without a colorbar**.

    Same data path as :func:`plot_waterfall_with_pointing` (mean/std from
    ``match_data_and_pointing`` when present, else ``get_plot_data``). Positions are
    RA/Dec (or Az/El if ``AzEL_map``). The scalar per point is shown by **marker area**
    when ``encode_value="size"`` (larger = larger value), or omitted when
    ``encode_value="none"`` (uniform markers, sky track only).

    Parameters
    ----------
    data : HDF5Data
        Spec data or pointing-matched data with mean/std channels.
    pointing_data : PointingData or None, default=None
        If None, ``data`` must already carry ra/dec/az/el.
    calibrated, data_type, attribute, use_std, AzEL_map, freq_avg
        As in :func:`plot_waterfall_with_pointing`.
    save_path : str or Path, optional
        If set, save the figure.
    size : float, default=36
        Scatter marker area (``s`` in matplotlib). When ``encode_value="size"``, this is
        the **maximum** area; minimum scales with ``size`` (20% of max by default).
    marker_color : str, default="C0"
        Face color for markers (no colormap).
    edgecolors : str or None, default="k"
        Edge color for markers; None for no edge.
    linewidths : float, default=0.35
        Edge width when ``edgecolors`` is set.
    encode_value : {"size", "none"}, default="size"
        How to represent the frequency-averaged value: marker size, or not at all.
    """
    if encode_value not in ("size", "none"):
        raise ValueError("encode_value must be 'size' or 'none'")

    ctx = _freq_avg_pointing_scatter_context(
        data,
        pointing_data,
        calibrated,
        data_type,
        attribute,
        use_std,
        AzEL_map,
        freq_avg,
    )
    arr_plot = ctx["arr_plot"]
    s = (
        _scatter_sizes_from_values(arr_plot, size)
        if encode_value == "size"
        else np.full_like(arr_plot, size, dtype=float)
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(
        ctx["x_axis"],
        ctx["y_axis"],
        s=s,
        c=marker_color,
        edgecolors=edgecolors,
        linewidths=linewidths if edgecolors is not None else 0,
    )
    plt.xlabel(ctx["x_label"])
    plt.ylabel(ctx["y_label"])
    spec_mean = ctx["spec_mean"]
    stat_suffix = (
        " (std)"
        if (ctx["use_std"] and spec_mean is not None)
        else (" (mean)" if (spec_mean is not None) else "")
    )
    enc_note = (
        f"; marker size scales with {ctx['value_label']}"
        if encode_value == "size"
        else ""
    )
    plt.title(f"{ctx['pol_name']} ({ctx['data_type']}){stat_suffix}{enc_note}".strip())
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_freq_avg_vs_pointing(
    data: HDF5Data,
    *,
    source_name: str | None = None,
    pointing_data: PointingData | None = None,
    calibrated: bool = False,
    data_type: str = "mag",
    attribute: str | None = None,
    use_std: bool = False,
    freq_avg: bool = True,
    sort_by_x: bool = True,
    plot: Literal["scatter", "line"] = "scatter",
    save_path: str | Path | None = None,
    size: float = 10.0,
    color: str = "C0",
    alpha: float = 0.9,
    clim: tuple[float, float] | None = None,
    sky_point_size: float | None = None,
    calibrators_path: Path | str | None = None,
) -> None:
    """
    Cross-pattern pointing: RA/Dec on-sky path plus marginals for each scan leg.

    Uses the same data path as :func:`plot_waterfall_with_pointing` (mean/std from
    matching when present). Splits samples **in time order** into two halves: the
    first half is treated as the scan used for the **RA** cut (amplitude vs RA),
    the second half for the **DEC** cut (amplitude vs Dec). The main panel shows
    the full track in RA/Dec, colored by frequency-averaged amplitude.

    Parameters
    ----------
    source_name : str or None, optional
        Calibrator name in ``calibrators.dat``. When set, overlays the catalog
        position (same lookup as :func:`skymap.io.get_pointing_offset`).
    clim : tuple of float or None, optional
        ``(vmin, vmax)`` for the main-panel color scale; if None, autoscale.
    sky_point_size : float or None, optional
        Marker size for the main RA/Dec scatter; defaults to ``size``.
    calibrators_path : path-like or None, optional
        Passed to :func:`skymap.Beam.get_source_radec` when ``source_name`` is set.
    """
    if plot not in ("scatter", "line"):
        raise ValueError("plot must be 'scatter' or 'line'")

    ctx = _freq_avg_pointing_scatter_context(
        data,
        pointing_data,
        calibrated,
        data_type,
        attribute,
        use_std,
        AzEL_map=False,
        freq_avg=freq_avg,
    )

    ra = np.asarray(ctx["x_axis"], dtype=float)
    dec = np.asarray(ctx["y_axis"], dtype=float)
    amp = np.asarray(ctx["arr_plot"], dtype=float)

    mask = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(amp)
    ra = ra[mask]
    dec = dec[mask]
    amp = amp[mask]
    if ra.size == 0:
        raise ValueError("No finite RA/Dec/amplitude samples to plot")

    n_mid = len(ra) // 2
    ra_1, dec_1, amp_1 = ra[:n_mid], dec[:n_mid], amp[:n_mid]
    ra_2, dec_2, amp_2 = ra[n_mid:], dec[n_mid:], amp[n_mid:]

    if sort_by_x:
        if ra_1.size:
            o = np.argsort(ra_1)
            ra_1, dec_1, amp_1 = ra_1[o], dec_1[o], amp_1[o]
        if ra_2.size:
            o = np.argsort(dec_2)
            ra_2, dec_2, amp_2 = ra_2[o], dec_2[o], amp_2[o]

    spec_mean = ctx["spec_mean"]
    stat_suffix = (
        " (std)"
        if (ctx["use_std"] and spec_mean is not None)
        else (" (mean)" if (spec_mean is not None) else "")
    )
    title_core = f"{ctx['pol_name']}{stat_suffix}".strip()

    src_ra = src_dec = None
    if source_name is not None:
        from skymap.Beam import get_source_radec

        src_ra, src_dec = get_source_radec(source_name, path=calibrators_path)

    s_sky = float(size) if sky_point_size is None else float(sky_point_size)
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.2], hspace=0.28, wspace=0.3)
    ax_amp_ra = fig.add_subplot(gs[0, 0])
    ax_amp_dec = fig.add_subplot(gs[0, 1])
    ax_radec = fig.add_subplot(gs[1, :])

    kw = dict(color=color, alpha=alpha)
    if plot == "line":
        if ra_1.size:
            ax_amp_ra.plot(ra_1, amp_1, **kw, linewidth=1.0)
        if ra_2.size:
            ax_amp_dec.plot(dec_2, amp_2, **kw, linewidth=1.0)
    else:
        if ra_1.size:
            ax_amp_ra.scatter(ra_1, amp_1, s=size, c=color, alpha=alpha, edgecolors="none")
        if ra_2.size:
            ax_amp_dec.scatter(dec_2, amp_2, s=size, c=color, alpha=alpha, edgecolors="none")

    ax_amp_ra.set_xlabel("RA (J2000) (deg)")
    ax_amp_ra.set_ylabel(ctx["value_label"])
    ax_amp_ra.set_title("Amplitude vs RA")

    ax_amp_dec.set_xlabel("Dec (J2000) (deg)")
    ax_amp_dec.set_ylabel(ctx["value_label"])
    ax_amp_dec.set_title("Amplitude vs Dec")

    if clim is not None:
        vmin, vmax = float(clim[0]), float(clim[1])
    else:
        vmin = float(np.nanmin(amp))
        vmax = float(np.nanmax(amp))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0 if np.isfinite(vmin) else 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    if ra_1.size:
        ax_radec.scatter(
            ra_1,
            dec_1,
            c=amp_1,
            cmap=cmap,
            norm=norm,
            s=s_sky,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.35,
            label="Scan 1",
        )
    if ra_2.size:
        ax_radec.scatter(
            ra_2,
            dec_2,
            c=amp_2,
            cmap=cmap,
            norm=norm,
            s=s_sky,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.35,
            label="Scan 2",
        )
    fig.colorbar(sm, ax=ax_radec, label=ctx["value_label"])
    if src_ra is not None:
        ax_radec.scatter(
            [src_ra],
            [src_dec],
            marker="*",
            s=85,
            c="red",
            edgecolors="k",
            linewidths=0.4,
            zorder=5,
            label=f"{source_name} (catalog)",
        )
    h, lab = ax_radec.get_legend_handles_labels()
    if lab:
        ax_radec.legend(loc="best", fontsize=9)
    ax_radec.set_xlabel("RA (J2000) (deg)")
    ax_radec.set_ylabel("Dec (J2000) (deg)")
    ax_radec.set_title(f"{title_core} — on-sky path")

    if src_ra is not None:
        ax_amp_ra.axvline(src_ra, color="red", linestyle="--", linewidth=1.0, alpha=0.85)
        ax_amp_dec.axvline(src_dec, color="red", linestyle="--", linewidth=1.0, alpha=0.85)

    fig.suptitle(title_core, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()