"""
Figure builders for the dashboard.

Most skymap plotting helpers draw with ``pyplot`` and then call ``plt.show()`` (and
``HealPixMap.plot`` also calls ``plt.close()``). :func:`capture` runs such a helper with ``show`` /
``close`` neutralised (under a lock, since pyplot is global state) and returns the resulting
``Figure``. Custom plots here use standalone ``matplotlib.figure.Figure`` objects instead.
"""

from __future__ import annotations

import math
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")

import healpy as hp  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.time import Time  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from skymap.dashboard import analysis  # noqa: E402
from skymap.dashboard.discovery import pol_label  # noqa: E402
from skymap.dashboard.loaders import PartialSpecData  # noqa: E402
from skymap import Beam, io, plots  # noqa: E402
from skymap.healmap import HealPixMap  # noqa: E402
from skymap.io import CAL_POL_NAMES, CalData, HDF5Data, PointingData  # noqa: E402
from skymap.utils import _time_to_mjd  # noqa: E402

_PYPLOT_LOCK = threading.RLock()


# --------------------------------------------------------------------------- pyplot capture
@contextmanager
def _pyplot_capture():
    """Neutralise ``plt.show``/``plt.close``; yield a list that receives the new figures."""
    new_figs: list[Figure] = []
    with _PYPLOT_LOCK:
        before = set(plt.get_fignums())
        real_show, real_close = plt.show, plt.close
        plt.show = lambda *a, **k: None
        plt.close = lambda *a, **k: None
        try:
            yield new_figs
        finally:
            plt.show, plt.close = real_show, real_close
            for num in plt.get_fignums():
                if num not in before:
                    new_figs.append(plt.figure(num))


def capture(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Figure:
    """
    Run a pyplot-based helper and return the figure it drew (the most recently created one).

    All figures created by the call are detached from pyplot (closed) so nothing leaks; the
    returned figure can still be rendered/saved.
    """
    with _pyplot_capture() as figs:
        fn(*args, **kwargs)
    if not figs:
        raise RuntimeError(f"{getattr(fn, '__name__', fn)} did not create a figure")
    with _PYPLOT_LOCK:
        for f in figs:
            plt.close(f)
    return figs[-1]


# --------------------------------------------------------------------------- time helpers
def _in_zone(time: np.ndarray, tz: str):
    """Naive-UTC datetimes -> (tz-aware datetimes, axis label, tzinfo)."""
    arr, label = plots._waterfall_times_in_zone(np.asarray(time), tz)
    tzinfo, _ = plots._parse_display_timezone(tz)
    return arr, label, tzinfo


def _mjd_in_zone(mjd: np.ndarray, tz: str):
    dts = np.asarray(Time(np.asarray(mjd, dtype=float), format="mjd", scale="utc").to_datetime())
    return _in_zone(dts, tz)


def _format_time_axis(ax, which: str, tzinfo, rotate: bool = True) -> None:
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(mdates.AutoDateLocator(tz=tzinfo))
    axis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=tzinfo))
    if rotate:
        labels = ax.get_xticklabels() if which == "x" else ax.get_yticklabels()
        for lab in labels:
            lab.set_rotation(30)
            lab.set_ha("right")


def _robust_clim(values: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float] | None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    a, b = np.percentile(v, [lo, hi])
    if a == b:
        b = a + 1.0
    return float(a), float(b)


# --------------------------------------------------------------------------- decimation
def decimate_obs(obs: HDF5Data, max_rows: int = 1500, *, magnitude: bool = True) -> HDF5Data:
    """
    Block-average ``obs`` in time to at most ``max_rows`` rows (display only).

    Calibrated spectra become plain ``float`` Kelvin arrays; raw spectra become the mean
    magnitude (``magnitude=True``) or the complex mean. Block start times are kept.
    """
    n = len(obs.time)
    block = max(1, math.ceil(n / max(1, int(max_rows))))
    if block == 1 and not _has_quantity(obs):
        return obs
    m = n // block
    if m == 0:
        return obs

    def reduce(arr, mag: bool):
        if getattr(arr, "unit", None) is not None:  # Quantity -> plain Kelvin values
            arr = arr.to_value(u.K)
        arr = np.asarray(arr)
        if mag and np.iscomplexobj(arr):
            arr = np.abs(arr)
        blk = arr[: m * block].reshape(m, block, arr.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(blk, axis=1)

    kwargs: dict = {}
    spec = None
    if obs.spec is not None:
        spec = PartialSpecData(
            **{
                n_: (reduce(getattr(obs.spec, n_), magnitude) if getattr(obs.spec, n_, None) is not None else None)
                for n_ in CAL_POL_NAMES
            },
            freq=obs.freq,
            time=obs.time[: m * block : block],
        )
    cal = getattr(obs, "calibrated_spec", None)
    if cal is not None:
        new = io.CalibratedSpec()
        for name in CAL_POL_NAMES:
            v = getattr(cal, name, None)
            setattr(new, name, reduce(v, False) if v is not None else None)
        kwargs["calibrated_spec"] = new
    return HDF5Data(freq=obs.freq, time=obs.time[: m * block : block], spec=spec, **kwargs)


def _has_quantity(obs: HDF5Data) -> bool:
    cal = getattr(obs, "calibrated_spec", None)
    if cal is None:
        return False
    return any(getattr(getattr(cal, n, None), "unit", None) is not None for n in CAL_POL_NAMES)


# --------------------------------------------------------------------------- calibration tab
def cal_figure(cal: CalData, pols: Sequence[str]) -> Figure:
    """Gain and Te vs frequency for the selected correlations (notebook ``beam_jan_2026`` cell 5)."""
    fig = Figure(figsize=(12, 4))
    ax_g, ax_t = fig.subplots(1, 2, sharex=True)
    f = cal.freq / 1e6
    for pol in pols:
        ax_g.plot(f, getattr(cal.gain, pol), label=pol_label(pol))
        ax_t.plot(f, getattr(cal.te, pol), label=pol_label(pol))
    ax_g.set_ylabel("Gain")
    ax_t.set_ylabel(r"$T_e$ (K)")
    for ax in (ax_g, ax_t):
        ax.set_xlabel("Frequency (MHz)")
        ax.grid(True, alpha=0.3)
    ax_g.legend(fontsize=8, ncol=2)
    fig.suptitle("Laboratory calibration")
    fig.tight_layout()
    return fig


def cal_grid_figure(cal: CalData, attribute: str) -> Figure:
    """The 2x5 per-correlation grid from :func:`skymap.plots.plot_cal_data`."""
    return capture(plots.plot_cal_data, cal, attribute=attribute)


def raw_vs_calibrated_figure(
    obs: HDF5Data,
    pol: str,
    tz: str,
    *,
    max_rows: int = 1500,
    clim_cal: tuple[float, float] | None = None,
) -> Figure:
    """Raw magnitude vs calibrated sky-temperature waterfall (notebook cell 18)."""
    dec = decimate_obs(obs, max_rows, magnitude=True)
    d_raw = plots.get_plot_data(dec, calibrated=False, data_type="magnitude", attribute=pol)
    d_cal = plots.get_plot_data(dec, calibrated=True, data_type="magnitude", attribute=pol)
    time, tz_label, tzinfo = _in_zone(d_raw["time"], tz)
    freq = d_raw["freq"]
    if clim_cal is None:
        clim_cal = _robust_clim(d_cal["plot_data"][pol])

    fig = Figure(figsize=(12, 5))
    ax0, ax1 = fig.subplots(1, 2, sharey=True)
    im0 = ax0.pcolormesh(freq, time, d_raw["plot_data"][pol], cmap="viridis", shading="auto")
    ax0.set_title(f"{pol_label(pol)} (uncalibrated, |V²|)")
    im1 = ax1.pcolormesh(
        freq, time, d_cal["plot_data"][pol], cmap="viridis", shading="auto",
        vmin=None if clim_cal is None else clim_cal[0],
        vmax=None if clim_cal is None else clim_cal[1],
    )
    ax1.set_title(f"{pol_label(pol)} (calibrated, T_sky)")
    for ax, im, lab in ((ax0, im0, d_raw["cbar_label"]), (ax1, im1, d_cal["cbar_label"])):
        ax.set_xlabel("Frequency (MHz)")
        fig.colorbar(im, ax=ax, label=lab)
    ax0.set_ylabel(f"Time ({tz_label})")
    _format_time_axis(ax0, "y", tzinfo)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- single-scan tab
def waterfall_figure(
    obs: HDF5Data,
    pols: Sequence[str],
    *,
    calibrated: bool,
    tz: str,
    data_type: str = "mag",
    clim: tuple[float, float] | None = None,
    max_rows: int = 1500,
) -> Figure:
    """
    :func:`skymap.plots.plot_waterfall` of a (block-averaged) window; 2x5 grid for ALL.

    With ``clim=None`` each panel gets its own 1-99 percentile range.
    """
    dec = decimate_obs(obs, max_rows, magnitude=data_type in ("mag", "magnitude"))
    single = len(pols) == 1
    attribute = pols[0] if single else None
    kwargs: dict = {}
    if clim is not None:
        kwargs["clim"] = clim
    else:
        d = plots.get_plot_data(dec, calibrated=calibrated, data_type=data_type, attribute=attribute)
        per = {n: _robust_clim(d["plot_data"][n]) for n in d["to_plot"]}
        kwargs["clim_per_plot"] = {k: v for k, v in per.items() if v is not None}
    fig = capture(
        plots.plot_waterfall,
        dec,
        calibrated=calibrated,
        data_type=data_type,
        attribute=attribute,
        time_zone=tz,
        figsize=(22, 8),
        **kwargs,
    )
    if single:
        fig.set_size_inches(8, 5.5)
        fig.tight_layout()
    return fig


def amplitude_time_figure(obs: HDF5Data, pols: Sequence[str], *, calibrated: bool, tz: str) -> Figure:
    """Frequency-averaged brightness vs time, one line per selected correlation."""
    dec = decimate_obs(obs, 4000, magnitude=True)
    time, tz_label, tzinfo = _in_zone(dec.time, tz)
    fig = Figure(figsize=(11, 3.8))
    ax = fig.add_subplot()
    for pol in pols:
        try:
            y = analysis.amplitude_series(dec, [pol], calibrated=calibrated)
        except ValueError:
            continue
        ax.plot(time, y, lw=1.0, label=pol_label(pol))
    ax.set_xlabel(f"Time ({tz_label})")
    ax.set_ylabel("Frequency-averaged " + ("T (K)" if calibrated else "|V²|"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5)
    _format_time_axis(ax, "x", tzinfo)
    fig.tight_layout()
    return fig


def pointing_time_figure(pointing: PointingData, tz: str) -> Figure:
    """Mount azimuth and elevation vs time."""
    time, tz_label, tzinfo = _mjd_in_zone(pointing.dmjd, tz)
    fig = Figure(figsize=(11, 5))
    ax_az, ax_el = fig.subplots(2, 1, sharex=True)
    ax_az.plot(time, pointing.az, ".", ms=2, color="C0")
    ax_az.set_ylabel("Az (deg)")
    ax_el.plot(time, pointing.el, ".", ms=2, color="C1")
    ax_el.set_ylabel("El (deg)")
    ax_el.set_xlabel(f"Time ({tz_label})")
    for ax in (ax_az, ax_el):
        ax.grid(alpha=0.3)
    _format_time_axis(ax_el, "x", tzinfo)
    fig.tight_layout()
    return fig


def radec_track_figure(
    matched: HDF5Data,
    pols: Sequence[str],
    *,
    source: str | None = None,
    use_std: bool = False,
    size: float = 30.0,
    clim: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """RA/Dec pointing track coloured by the frequency-averaged amplitude of the matched data."""
    amp = analysis.amplitude_series(matched, pols, kind="std" if use_std else "mean")
    ra = np.asarray(matched.ra, dtype=float)
    dec = np.asarray(matched.dec, dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(amp)
    if clim is None:
        clim = _robust_clim(amp[ok], 1, 99)
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot()
    sc = ax.scatter(
        ra[ok], dec[ok], c=amp[ok], s=size, cmap="viridis",
        vmin=None if clim is None else clim[0], vmax=None if clim is None else clim[1],
    )
    fig.colorbar(sc, ax=ax, label="Temperature (K) " + ("std" if use_std else "mean"))
    if source is not None:
        s_ra, s_dec = Beam.get_source_radec(source)
        ax.scatter([s_ra], [s_dec], marker="*", s=140, c="red", edgecolors="k", zorder=5, label=source)
        ax.legend(fontsize=8)
    ax.set_xlabel("RA (J2000) (deg)")
    ax.set_ylabel("Dec (J2000) (deg)")
    ax.grid(alpha=0.3)
    ax.set_title(title or "/".join(pol_label(p) for p in pols[:3]) + ("..." if len(pols) > 3 else ""))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- multi-scan tab
def _safe_axis_fit(xs, ys, *, x_name: str, ax, labels):
    try:
        res = plots.fit_axis_beam(xs, ys, x_name=x_name, ax=ax, labels=labels)
        res = {k: v for k, v in res.items() if k != "covariance"}
        return res, None
    except Exception as exc:  # noqa: BLE001 - shown on the axes and in the UI
        ax.text(0.5, 0.5, f"{x_name} fit failed:\n{exc}", transform=ax.transAxes, ha="center", va="center", fontsize=8)
        return None, f"{type(exc).__name__}: {exc}"


def peak_scan_figure(
    tc: HDF5Data,
    source: str,
    pols: Sequence[str],
    n_scans: int,
    n_el_scans: int,
    *,
    clim: tuple[float, float] | None = None,
    size: float = 50.0,
    value_label: str = "Temperature (K)",
) -> tuple[Figure, dict]:
    """
    Five-panel peak-scan summary (``pointing_offset.ipynb`` cell 25).

    ``tc`` is spectrum-cadence data with ``ra/dec/az/el`` and ``calibrated_spec_mean`` (e.g. from
    :func:`analysis.timing_corrected`) or pointing-cadence matched data. RA/Dec are shifted by the
    mean per-leg sky offsets before the Dec/RA Gaussian fits. Returns ``(figure, info)``.
    """
    kw = analysis.scan_leg_args(n_scans, n_el_scans)
    amp_all = analysis.amplitude_series(tc, pols)
    pt_off = io.get_pointing_offset(tc, source, **kw)
    ra_shift = float(np.mean(pt_off["ra_offsets"]))
    dec_shift = float(np.mean(pt_off["dec_offsets"]))

    az = np.asarray(tc.az, dtype=float)
    el = np.asarray(tc.el, dtype=float)
    ra = np.asarray(tc.ra, dtype=float) - ra_shift
    dec = np.asarray(tc.dec, dtype=float) - dec_shift
    time = np.asarray(tc.time)
    mask = np.isfinite(az) & np.isfinite(el) & np.isfinite(amp_all) & np.isfinite(ra) & np.isfinite(dec)
    az, el, ra, dec, amp, time = az[mask], el[mask], ra[mask], dec[mask], amp_all[mask], time[mask]
    if amp.size < n_scans:
        raise ValueError("Not enough finite samples in the window for the requested number of scans")

    legs, split_mode, _ = io.resolve_scan_legs(
        _time_to_mjd(time), n_scans=n_scans, valid_mask=np.isfinite(az) & np.isfinite(el)
    )
    el_idx, az_idx = legs[:n_el_scans], legs[n_el_scans:]
    colors = [f"C{i}" for i in range(n_scans)]

    src_ra, src_dec = Beam.get_source_radec(source)
    src_az, src_el = io.expected_source_altaz_deg(src_ra, src_dec, time)
    n_el_end = int(el_idx[-1][-1]) + 1
    src_el_ref = float(src_el[int(np.nanargmax(amp[:n_el_end]))]) if n_el_end > 0 else None
    src_az_ref = float(src_az[n_el_end + int(np.nanargmax(amp[n_el_end:]))]) if n_el_end < len(amp) else None

    fig = Figure(figsize=(12, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.35], hspace=0.35, wspace=0.28)
    ax_el = fig.add_subplot(gs[0, 0])
    ax_az = fig.add_subplot(gs[0, 1])
    ax_dec = fig.add_subplot(gs[1, 0])
    ax_ra = fig.add_subplot(gs[1, 1])
    ax_sky = fig.add_subplot(gs[2, :])

    def marginal(ax, coord, a, color, label):
        o = np.argsort(coord)
        ax.scatter(coord[o], a[o], s=10, c=color, alpha=0.9, edgecolors="none", label=label)

    for i, idx in enumerate(el_idx):
        marginal(ax_el, el[idx], amp[idx], colors[i], f"Scan {i + 1}")
    for j, idx in enumerate(az_idx):
        marginal(ax_az, az[idx], amp[idx], colors[n_el_scans + j], f"Scan {n_el_scans + j + 1}")
    for ax, xl, ttl, ref in (
        (ax_el, "Elevation (deg)", "Peak scan: Elevation", src_el_ref),
        (ax_az, "Azimuth (deg)", "Peak scan: Azimuth", src_az_ref),
    ):
        ax.set_xlabel(xl)
        ax.set_ylabel(value_label)
        ax.set_title(ttl)
        ax.grid(True)
        ax.legend(loc="best", fontsize=8)
        if ref is not None:
            ax.axvline(ref, color="red", ls="--", lw=1.0, alpha=0.85)

    el_labels = [f"Scan {i + 1}" for i in range(n_el_scans)]
    az_labels = [f"Scan {n_el_scans + j + 1}" for j in range(len(az_idx))]
    fit_dec, err_dec = _safe_axis_fit(
        [dec[i] for i in el_idx], [amp[i] for i in el_idx], x_name="Dec", ax=ax_dec, labels=el_labels
    )
    fit_ra, err_ra = _safe_axis_fit(
        [ra[i] for i in az_idx], [amp[i] for i in az_idx], x_name="RA", ax=ax_ra, labels=az_labels
    )
    ax_dec.axvline(src_dec, color="red", ls="--", lw=1.0, alpha=0.85, label=f"{source} Dec")
    ax_ra.axvline(src_ra, color="red", ls="--", lw=1.0, alpha=0.85, label=f"{source} RA")
    for ax, ttl in ((ax_dec, "Freq-avg: Gaussian fit across Dec"), (ax_ra, "Freq-avg: Gaussian fit across RA")):
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(ttl)
        ax.set_ylabel(value_label)

    if clim is None:
        clim = _robust_clim(amp, 1, 99)
    sc = ax_sky.scatter(
        ra, dec, c=amp, cmap="viridis", s=size,
        vmin=None if clim is None else clim[0], vmax=None if clim is None else clim[1],
    )
    fig.colorbar(sc, ax=ax_sky, label=value_label)
    ax_sky.scatter([src_ra], [src_dec], marker="*", s=140, c="red", edgecolors="k", zorder=5, label=source)
    ax_sky.legend(fontsize=8)
    ax_sky.set_xlabel("RA (J2000) (deg)")
    ax_sky.set_ylabel("Dec (J2000) (deg)")
    ax_sky.grid(True)
    fig.tight_layout()

    info = {
        "scan_split": split_mode,
        "n_samples": int(amp.size),
        "ra_shift_deg": ra_shift,
        "dec_shift_deg": dec_shift,
        "pointing_offset": pt_off,
        "fit_dec": fit_dec,
        "fit_ra": fit_ra,
        "fit_dec_error": err_dec,
        "fit_ra_error": err_ra,
    }
    return fig, info


# --------------------------------------------------------------------------- offsets tab
def timing_diagnostic_figure(
    obs: HDF5Data,
    pointing: PointingData,
    timing_info: dict,
    source: str,
) -> Figure:
    """Per scan leg: pointing closest approach vs spectrum brightness peak (notebook diagnostic)."""
    spec_mjd = _time_to_mjd(obs.time)
    p_mjd_all = _time_to_mjd(np.asarray(pointing.dmjd, dtype=float))
    order = np.argsort(p_mjd_all)
    p_mjd = p_mjd_all[order]
    p_az = np.asarray(pointing.az, dtype=float)[order]
    p_el = np.asarray(pointing.el, dtype=float)[order]
    t0 = float(p_mjd[0])
    p_t = (p_mjd - t0) * 86400.0
    s_t = (spec_mjd - t0) * 86400.0

    src_ra, src_dec = Beam.get_source_radec(source)
    src_az_cat, src_el_cat = io.expected_source_altaz_deg(src_ra, src_dec, p_mjd)
    src_az = src_az_cat + timing_info["az_offset_deg"]
    src_el = src_el_cat + timing_info["el_offset_deg"]

    panels = [("el", leg, j) for j, leg in enumerate(timing_info["el_scans"])]
    panels += [("az", leg, j) for j, leg in enumerate(timing_info["az_scans"])]
    ncols = 2
    nrows = math.ceil(len(panels) / ncols)
    fig = Figure(figsize=(12, 4 * nrows))
    axes = np.atleast_1d(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    for ax, (kind, leg, j) in zip(axes, panels):
        key = "el" if kind == "el" else "az"
        i_point = timing_info[f"{key}_pointing_closest_indices"][j]
        i_spec = timing_info[f"{key}_spectrum_peak_indices"][j]
        dt_ms = timing_info[f"{key}_timing_offsets"][j] * 1e3
        if kind == "el":
            ax.plot(p_t, p_el, ".", ms=2, alpha=0.4, label="pointing el")
            ax.axhline(float(src_el_cat[i_point]), color="0.5", ls=":", label="catalog el")
            ax.axhline(float(src_el[i_point]), color="C1", ls="--", label="catalog+spatial el")
            ax.set_ylabel("El (deg)")
        else:
            ax.plot(p_t, p_az, ".", ms=2, alpha=0.4, label="pointing az")
            ax.axhline(float(src_az_cat[i_point]), color="0.5", ls=":", label="catalog az")
            ax.axhline(float(src_az[i_point]), color="C1", ls="--", label="catalog+spatial az")
            ax.set_ylabel("Az (deg)")
        ax.axvline(p_t[i_point], color="C2", label="pointing closest")
        ax.axvline(s_t[i_spec], color="C3", ls=":", label="spectrum peak")
        ax.set_title(f"{kind} scan leg {leg}: Δt = {dt_ms:.1f} ms")
        ax.set_xlabel("Time (s from first pointing sample)")
        ax.legend(fontsize=7, loc="best")
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(
        f"Spatial then timing: az_off={timing_info['az_offset_deg']:.4f}°, "
        f"el_off={timing_info['el_offset_deg']:.4f}°, "
        f"lag={timing_info['timing_offset_seconds'] * 1e3:.1f} ms"
    )
    fig.tight_layout()
    return fig


def timing_compare_figure(
    window_matched: HDF5Data,
    timing_corrected: HDF5Data,
    pols: Sequence[str],
    *,
    source: str | None = None,
    max_points: int = 4000,
) -> Figure:
    """Window-matched vs timing-corrected sky track, coloured by amplitude."""
    a0 = analysis.amplitude_series(window_matched, pols)
    a1 = analysis.amplitude_series(timing_corrected, pols)
    clim = _robust_clim(np.concatenate([a0[np.isfinite(a0)], a1[np.isfinite(a1)]]), 1, 99)
    step = max(1, len(timing_corrected.time) // max_points)
    fig = Figure(figsize=(12, 4.5))
    axes = fig.subplots(1, 2)
    for ax, ra, dec, a, ttl, s in (
        (axes[0], window_matched.ra, window_matched.dec, a0, "Window match (no clock correction)", 12),
        (
            axes[1],
            np.asarray(timing_corrected.ra)[::step],
            np.asarray(timing_corrected.dec)[::step],
            a1[::step],
            f"Timing-corrected ({getattr(timing_corrected, 'timing_offset_seconds', 0.0) * 1e3:.1f} ms)",
            5,
        ),
    ):
        sc = ax.scatter(ra, dec, c=a, s=s, cmap="viridis", vmin=None if clim is None else clim[0],
                        vmax=None if clim is None else clim[1])
        fig.colorbar(sc, ax=ax, label="Temperature (K)")
        ax.set_title(ttl)
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        if source is not None:
            s_ra, s_dec = Beam.get_source_radec(source)
            ax.scatter([s_ra], [s_dec], marker="*", s=120, c="red", edgecolors="k", zorder=5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- HEALPix tab
def _pol_title(pols: Sequence[str]) -> str:
    return pol_label(pols[0]) if len(pols) == 1 else "all correlations"


def healpix_full_figure(
    hmap: HealPixMap,
    pols: Sequence[str],
    *,
    stat: str = "mean",
    clim: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Full-sky Mollweide via :meth:`HealPixMap.plot` (grid of maps for ALL)."""
    kwargs: dict = {"stat": stat, "cmap": cmap, "title": f"HealPix {_pol_title(pols)} (nside={hmap.nside})"}
    if clim is not None:
        kwargs["clim"] = clim
    if len(pols) == 1:
        kwargs["attribute"] = pols[0]
        kwargs["figsize"] = (9, 5)
    return capture(hmap.plot, **kwargs)


def healpix_patch_figure(
    hmap: HealPixMap,
    pols: Sequence[str],
    *,
    center_ra: float,
    center_dec: float,
    patch_deg: float = 10.0,
    stat: str = "mean",
    clim: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> Figure:
    """RA/Dec patch via :meth:`HealPixMap.plot` with ``ra_dec_map=True`` and a region."""
    kwargs: dict = {
        "stat": stat,
        "cmap": cmap,
        "ra_dec_map": True,
        "region": (float(center_ra), float(center_dec), float(patch_deg) ** 2),
        "title": f"HealPix {_pol_title(pols)} ({patch_deg:g}° × {patch_deg:g}°)",
    }
    if clim is not None:
        kwargs["clim"] = clim
    if len(pols) == 1:
        kwargs["attribute"] = pols[0]
        kwargs["figsize"] = (7, 6)
    return capture(hmap.plot, **kwargs)


def healpix_gnomview_figure(
    hmap: HealPixMap,
    pol: str,
    *,
    center_ra: float,
    center_dec: float,
    patch_deg: float = 10.0,
    reso_arcmin: float = 2.0,
    clim: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> Figure:
    """``healpy.gnomview`` of one correlation's map centred on (RA, Dec)."""
    m = np.asarray(hmap.get_map(pol), dtype=float).copy()
    m[np.asarray(hmap.get_hit_count(), dtype=float) <= 0] = hp.UNSEEN
    xsize = int(round(patch_deg * 60.0 / reso_arcmin))
    kwargs: dict = {}
    if clim is not None:
        kwargs["min"], kwargs["max"] = clim
    return capture(
        hp.gnomview,
        m,
        rot=(float(center_ra), float(center_dec), 0.0),
        coord="C",
        reso=reso_arcmin,
        xsize=xsize,
        title=f"HealPix gnomview {pol_label(pol)} ({patch_deg:g}° × {patch_deg:g}°)",
        unit=f"{pol_label(pol)} (K)",
        cmap=cmap,
        notext=True,
        **kwargs,
    )


# --------------------------------------------------------------------------- beam tab
def beam_fit_figure(
    offset_deg: np.ndarray,
    values_k: np.ndarray,
    *,
    baseline_outer_deg: float = 1.0,
    x_max_deg: float = 5.0,
    normalize: bool = True,
    show_components: bool = True,
    title: str = "Beam profile",
) -> tuple[Figure, dict]:
    """Radial profile with the Gaussian fit (left) and the fitted beam approximation (right)."""
    fig = Figure(figsize=(12, 4.2))
    ax_fit, ax_beam = fig.subplots(1, 2)
    params = Beam.fit_beam_gaussian(
        offset_deg, values_k, baseline_outer_deg=baseline_outer_deg, ax=ax_fit, show=False, title=title
    )
    Beam.plot_beam_approximation(
        params, x_max_deg=x_max_deg, normalize=normalize, show_components=show_components,
        ax=ax_beam, show=False,
    )
    fig.tight_layout()
    return fig, params


def beam_approximation_figure(
    params: dict, *, x_max_deg: float = 5.0, normalize: bool = True, show_components: bool = True,
    title: str = "Gaussian beam",
) -> Figure:
    """Beam approximation for an already-known parameter dict (e.g. loaded JSON)."""
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot()
    Beam.plot_beam_approximation(
        params, x_max_deg=x_max_deg, normalize=normalize, show_components=show_components,
        ax=ax, show=False, title=title,
    )
    fig.tight_layout()
    return fig
