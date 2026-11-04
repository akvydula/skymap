"""
Skymap beam-analysis dashboard (Streamlit).

Run in the environment where ``skymap`` is installed (from the folder that holds ``data_310/``)::

    skymap-dashboard                      # or: python -m skymap.dashboard
    skymap-dashboard --data-root /path/to/data_310

Pick a session, calibration file, pointing directory, spectrometer file and correlation in the
sidebar, then select the time windows to open. Only those windows are read from the (large)
files. All analysis is done by the existing ``skymap`` package; see ``skymap/dashboard/analysis.py``.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple
from zoneinfo import ZoneInfo

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from astropy.time import Time  # noqa: E402

from skymap.dashboard import analysis, discovery, figures, loaders  # noqa: E402
from skymap.dashboard.discovery import pol_label  # noqa: E402
from skymap import Beam, io  # noqa: E402

TZ_CHOICES = ["America/New_York", "EST", "UTC"]
VIEWS = [
    "Calibration",
    "Single scan",
    "Peak scans (az/el)",
    "Pointing & timing offsets",
    "HEALPix",
    "Beam fit",
]
# Defaults for the Jan-16 test observation (UTC), used only when they fall inside the chosen file.
DEFAULT_WINDOWS_UTC = {
    "peak": (datetime(2026, 1, 16, 18, 4), datetime(2026, 1, 16, 18, 9)),
    "beam": (datetime(2026, 1, 16, 18, 9), datetime(2026, 1, 16, 18, 52)),
}
DEFAULT_DATA_FILE = "GBTOBS_2026-01-16T112750.hdf5"
FALLBACK_WINDOW_S = 300.0

st.set_page_config(page_title="Skymap beam dashboard", layout="wide")


# =========================================================================== cached loading
class WindowKey(NamedTuple):
    """Hashable description of one loaded window (cache key for everything downstream)."""

    data_path: str
    data_mtime: float
    cal_path: str
    cal_mtime: float
    pointing_dir: str
    i0: int
    i1: int
    pols: tuple[str, ...]
    add_offset: bool


@dataclass
class Window:
    obs: object
    pointing: object | None
    pointing_error: str | None
    start_utc: datetime
    end_utc: datetime

    @property
    def n_samples(self) -> int:
        return len(self.obs.time)

    def require_pointing(self):
        if self.pointing is None:
            raise RuntimeError(f"No pointing data for this window: {self.pointing_error}")
        return self.pointing


@st.cache_data(show_spinner=False, max_entries=8)
def cached_peek(path: str, mtime: float) -> loaders.ObsPeek:
    return loaders.peek_obs_hdf5(path)


@st.cache_resource(show_spinner=False, max_entries=4)
def cached_cal(path: str, mtime: float):
    return loaders.load_cal(path)


@st.cache_resource(show_spinner="Reading spectra and pointing for the selected window...", max_entries=4)
def load_window(key: WindowKey) -> Window:
    peek = cached_peek(key.data_path, key.data_mtime)
    sl = slice(key.i0, key.i1)
    obs = loaders.read_obs_window(key.data_path, sl, list(key.pols))
    loaders.calibrate_window(obs, cached_cal(key.cal_path, key.cal_mtime), list(key.pols))
    start_utc, end_utc = loaders.window_utc_bounds(peek, sl)
    pointing, err = None, None
    try:
        pointing = loaders.load_pointing_window(
            key.pointing_dir, start_utc, end_utc, add_pointing_offset=key.add_offset
        )
    except Exception as exc:  # noqa: BLE001 - surfaced where pointing is needed
        err = f"{type(exc).__name__}: {exc}"
    return Window(obs, pointing, err, start_utc, end_utc)


@st.cache_resource(show_spinner="Matching spectra to pointing...", max_entries=6)
def get_matched(key: WindowKey):
    w = load_window(key)
    return analysis.match_window(w.obs, w.require_pointing(), key.pols)


@st.cache_resource(show_spinner="Estimating timing offset...", max_entries=6)
def get_timing(key: WindowKey, source: str, n_scans: int, n_el: int, max_offset: float):
    w = load_window(key)
    return analysis.timing_estimate(
        w.obs, w.require_pointing(), source, key.pols, n_scans, n_el, max_offset
    )


@st.cache_resource(show_spinner="Applying timing correction...", max_entries=6)
def get_timing_corrected(key: WindowKey, offset_s: float, max_offset: float):
    w = load_window(key)
    return analysis.timing_corrected(
        w.obs, w.require_pointing(), key.pols,
        timing_offset_seconds=offset_s, max_offset_seconds=max_offset,
    )


@st.cache_resource(show_spinner="Filling HEALPix map...", max_entries=3)
def get_healpix(key: WindowKey, nside: int, freq_index: int | None):
    return analysis.build_healpix(get_matched(key), key.pols, nside, freq_index)


# =========================================================================== small helpers
def show(fig) -> None:
    st.pyplot(fig, width="stretch")


def run(fn, *args, **kwargs) -> None:
    """Run a view; report errors in the page instead of crashing the app."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - user-facing error report
        st.error(f"{type(exc).__name__}: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())


def clim_inputs(key: str, default: tuple[float, float] = (30.0, 150.0)) -> tuple[float, float] | None:
    if st.checkbox("Auto colour limits", value=True, key=f"{key}_auto"):
        return None
    c1, c2 = st.columns(2)
    lo = c1.number_input("Min", value=float(default[0]), key=f"{key}_lo")
    hi = c2.number_input("Max", value=float(default[1]), key=f"{key}_hi")
    return (float(lo), float(hi))


def _utc_to_wall(dt_utc: datetime, tz: str) -> datetime:
    return dt_utc.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz)).replace(tzinfo=None)


def _wall_to_utc(dt_wall: datetime, tz: str) -> datetime:
    return dt_wall.replace(tzinfo=ZoneInfo(tz)).astimezone(timezone.utc).replace(tzinfo=None)


def _floor_s(dt: datetime) -> datetime:
    return dt.replace(microsecond=0)


def _ceil_s(dt: datetime) -> datetime:
    return dt if dt.microsecond == 0 else dt.replace(microsecond=0) + timedelta(seconds=1)


def default_window_mjd(peek: loaders.ObsPeek, slot: str) -> tuple[float, float]:
    """Notebook windows if they lie inside the file; else the first five minutes."""
    d0, d1 = DEFAULT_WINDOWS_UTC[slot]
    m0, m1 = float(Time(d0).mjd), float(Time(d1).mjd)
    if peek.start_mjd <= m0 < peek.end_mjd:
        return m0, min(m1, peek.end_mjd)
    return peek.start_mjd, min(peek.end_mjd, peek.start_mjd + FALLBACK_WINDOW_S / 86400.0)


@dataclass
class WindowSel:
    sl: slice | None
    summary: str = ""


def select_window(
    slot: str, title: str, peek: loaders.ObsPeek, tz: str, n_pols: int, max_samples: int
) -> WindowSel:
    """Sidebar controls that turn a wall-time range / index range into a ``slice``."""
    tag = f"{slot}|{Path(peek.path).name}|{tz}"
    m0, m1 = default_window_mjd(peek, slot)
    w_start = _ceil_s(loaders.mjd_to_wall(peek.start_mjd, tz))
    w_end = _floor_s(loaders.mjd_to_wall(peek.end_mjd, tz))
    d_start = min(max(_floor_s(loaders.mjd_to_wall(m0, tz)), w_start), w_end)
    d_end = min(max(_floor_s(loaders.mjd_to_wall(m1, tz)), d_start), w_end)

    st.markdown(f"**{title}**")
    mode = st.radio(
        "Select by", ["Time slider", "Type times", "Sample index"],
        horizontal=True, key=f"{slot}_mode", label_visibility="collapsed",
    )
    sl: slice | None = None
    try:
        if mode == "Time slider" and w_end > w_start:
            a, b = st.slider(
                f"Window ({tz})", min_value=w_start, max_value=w_end, value=(d_start, d_end),
                step=timedelta(seconds=1), format="MM/DD HH:mm:ss", key=f"{slot}_slider|{tag}",
            )
            sl = loaders.slice_from_wall_times(peek, a, b, tz)
        elif mode == "Type times":
            s_txt = st.text_input(f"Start ({tz})", d_start.isoformat(sep=" "), key=f"{slot}_s|{tag}")
            e_txt = st.text_input(f"End ({tz})", d_end.isoformat(sep=" "), key=f"{slot}_e|{tag}")
            sl = loaders.slice_from_wall_times(
                peek, datetime.fromisoformat(s_txt.strip()), datetime.fromisoformat(e_txt.strip()), tz
            )
        else:
            i0d = int(np.searchsorted(peek.mjd, m0, side="left"))
            i1d = int(np.searchsorted(peek.mjd, m1, side="left"))
            c1, c2 = st.columns(2)
            i0 = c1.number_input("Start index", 0, peek.n_time, i0d, key=f"{slot}_i0|{tag}")
            i1 = c2.number_input("End index", 0, peek.n_time, i1d, key=f"{slot}_i1|{tag}")
            sl = loaders.slice_from_indices(peek, int(i0), int(i1))
    except ValueError as exc:
        st.error(f"Invalid window: {exc}")
        return WindowSel(None)

    n = sl.stop - sl.start
    if n <= 0:
        st.warning("The selected window contains no samples.")
        return WindowSel(None)
    t0, t1 = loaders.mjd_to_naive_utc(peek.mjd[sl.start]), loaders.mjd_to_naive_utc(peek.mjd[sl.stop - 1])
    mem_mb = loaders.estimate_window_memory_mb(n, n_pols, peek.n_freq)
    summary = (
        f"{n:,} samples, {(peek.mjd[sl.stop - 1] - peek.mjd[sl.start]) * 86400:.0f} s, "
        f"{t0:%H:%M:%S}-{t1:%H:%M:%S} UTC, ~{mem_mb:,.0f} MB"
    )
    st.caption(summary)
    if n > max_samples:
        st.warning(f"Window exceeds the {max_samples:,}-sample guard.")
        if not st.checkbox("Load anyway", key=f"{slot}_force|{tag}"):
            return WindowSel(None, summary)
    return WindowSel(sl, summary)


# =========================================================================== sidebar
@dataclass
class Ctx:
    session: Path
    cal_path: str
    pointing_dir: str
    data_path: str
    choice: str
    pols: tuple[str, ...]
    source: str
    tz: str
    add_offset: bool
    n_scans: int
    n_el: int
    max_rows: int
    override_ms: float | None
    max_offset_s: float
    peak: WindowSel
    beam: WindowSel

    def key(self, sel: WindowSel) -> WindowKey:
        assert sel.sl is not None
        return WindowKey(
            self.data_path, os.path.getmtime(self.data_path),
            self.cal_path, os.path.getmtime(self.cal_path),
            self.pointing_dir, int(sel.sl.start), int(sel.sl.stop),
            tuple(self.pols), bool(self.add_offset),
        )


def build_sidebar() -> Ctx | None:
    sb = st.sidebar
    sb.title("Skymap beam dashboard")
    root = Path(sb.text_input("Data root", str(discovery.DEFAULT_DATA_ROOT))).expanduser()
    sessions = discovery.find_sessions(root)
    if not sessions:
        sb.error(f"No session folders (e.g. TGBT25B_614_01) found in {root}")
        return None
    names = [s.name for s in sessions]
    default_i = names.index(discovery.DEFAULT_SESSION) if discovery.DEFAULT_SESSION in names else len(names) - 1
    session = sessions[names.index(sb.selectbox("Session", names, index=default_i))]

    cals = discovery.find_cal_files(session)
    pdirs = discovery.find_pointing_dirs(session)
    include_derived = sb.checkbox("Include derived cuts (beam_obs_*, *_source_cut_*)", value=False)
    datas = discovery.find_data_files(session, include_derived=include_derived)
    for label, items in (("calibration files", cals), ("pointing directory", pdirs), ("spectrometer HDF5 files", datas)):
        if not items:
            sb.error(f"No {label} found in {session.name}")
    if not (cals and pdirs and datas):
        return None

    cal = sb.selectbox("Calibration file", cals, format_func=lambda p: p.name)
    pdir = sb.selectbox("Pointing directory", pdirs, format_func=lambda p: p.name)
    cov = discovery.pointing_coverage(pdir)
    if cov.n_files:
        sb.caption(f"{cov.n_files} pointing files, {cov.first:%Y-%m-%d %H:%M:%S} - {cov.last:%H:%M:%S} UTC (file starts)")
    data_names = [p.name for p in datas]
    data_i = data_names.index(DEFAULT_DATA_FILE) if DEFAULT_DATA_FILE in data_names else 0
    data = sb.selectbox("Receiver data (spectrometer)", datas, index=data_i, format_func=lambda p: p.name)
    choice = sb.selectbox(
        "Cross-correlation", discovery.CORRELATION_CHOICES,
        index=discovery.CORRELATION_CHOICES.index("AB*"),
    )
    pols = tuple(discovery.pol_attributes(choice))

    sources = analysis.source_names()
    source = sb.selectbox("Source", sources, index=sources.index(analysis.DEFAULT_SOURCE) if analysis.DEFAULT_SOURCE in sources else 0)
    tz = sb.selectbox("Display / input time zone", TZ_CHOICES)

    try:
        peek = cached_peek(str(data), os.path.getmtime(data))
    except Exception as exc:  # noqa: BLE001
        sb.error(f"Could not read {data.name}: {exc}")
        return None
    sb.caption(
        f"{peek.n_time:,} samples x {peek.n_pol} pols x {peek.n_freq} chan, cadence {peek.cadence_s:.3f} s\n\n"
        f"{loaders.mjd_to_wall(peek.start_mjd, tz):%Y-%m-%d %H:%M:%S} - "
        f"{loaders.mjd_to_wall(peek.end_mjd, tz):%H:%M:%S} {tz}"
    )

    with sb.expander("Options"):
        max_samples = int(st.number_input("Window size guard (samples)", 1000, 500_000, 40_000, step=5000))
        add_offset = st.checkbox(
            f"Add fixed telescope offsets (+{io._POINTING_EL_OFFSET_DEG} el, +{io._POINTING_AZ_OFFSET_DEG} az deg)",
            value=True,
        )
        n_scans = int(st.selectbox("Scans in peak-scan window", [2, 3, 4], index=2))
        n_el = int(st.selectbox("...of which elevation scans (first)", list(range(1, n_scans)), index=min(n_scans // 2, n_scans - 1) - 1))
        max_rows = int(st.slider("Max waterfall rows (display only)", 300, 4000, 1500, step=100))
        max_offset_s = float(st.number_input("Max timing offset (s)", 0.01, 5.0, io._MAX_SPECTRA_TIMING_OFFSET_S, step=0.05))
        override = st.checkbox("Override timing offset", value=False)
        override_ms = float(st.number_input("Timing offset (ms)", -5000.0, 5000.0, 0.0, step=10.0)) if override else None

    sb.divider()
    sb.subheader("Time windows")
    with sb:
        peak = select_window("peak", "Peak-scan window (source cut)", peek, tz, len(pols), max_samples)
        beam = select_window("beam", "Beam-map window", peek, tz, len(pols), max_samples)

    return Ctx(
        session, str(cal), str(pdir), str(data), choice, pols, source, tz, add_offset,
        n_scans, n_el, max_rows, override_ms, max_offset_s, peak, beam,
    )


# =========================================================================== views
def _need(sel: WindowSel, name: str) -> bool:
    if sel.sl is None:
        st.info(f"Choose a valid {name} in the sidebar.")
        return False
    return True


def _window_info(w: Window) -> None:
    ptxt = (
        f"{len(w.pointing.dmjd):,} pointing samples" if w.pointing is not None else f"pointing unavailable ({w.pointing_error})"
    )
    st.caption(f"{w.n_samples:,} spectra, {w.start_utc:%Y-%m-%d %H:%M:%S} - {w.end_utc:%H:%M:%S} UTC; {ptxt}")


def view_calibration(ctx: Ctx) -> None:
    cal = cached_cal(ctx.cal_path, os.path.getmtime(ctx.cal_path))
    st.subheader("Laboratory calibration")
    show(figures.cal_figure(cal, ctx.pols))
    with st.expander("Per-correlation grids"):
        attr = st.radio("Quantity", ["gain", "te"], horizontal=True)
        show(figures.cal_grid_figure(cal, attr))

    st.subheader("Raw vs calibrated")
    slot = st.radio("Window", ["Peak-scan window", "Beam-map window"], horizontal=True)
    sel = ctx.peak if slot.startswith("Peak") else ctx.beam
    if not _need(sel, slot.lower()):
        return
    pol = ctx.pols[0] if len(ctx.pols) == 1 else ctx.pols[
        [pol_label(p) for p in ctx.pols].index(st.selectbox("Correlation", [pol_label(p) for p in ctx.pols]))
    ]
    w = load_window(ctx.key(sel))
    _window_info(w)
    show(figures.raw_vs_calibrated_figure(w.obs, pol, ctx.tz, max_rows=ctx.max_rows))


def view_single_scan(ctx: Ctx) -> None:
    if not _need(ctx.peak, "peak-scan window"):
        return
    key = ctx.key(ctx.peak)
    w = load_window(key)
    _window_info(w)
    c1, c2 = st.columns([1, 2])
    calibrated = c1.checkbox("Calibrated (K)", value=True)
    data_type = "mag" if calibrated else c2.selectbox("Raw quantity", ["mag", "real", "imag", "phase"])
    with st.expander("Waterfall colour limits"):
        clim = clim_inputs("wf", (30.0, 150.0))
    st.subheader("Waterfall")
    show(figures.waterfall_figure(w.obs, ctx.pols, calibrated=calibrated, tz=ctx.tz, data_type=data_type, clim=clim, max_rows=ctx.max_rows))
    st.subheader("Frequency-averaged amplitude vs time")
    show(figures.amplitude_time_figure(w.obs, ctx.pols, calibrated=calibrated, tz=ctx.tz))
    st.subheader("Pointing")
    pointing = w.require_pointing()
    show(figures.pointing_time_figure(pointing, ctx.tz))
    st.subheader("Sky track (RA/Dec)")
    c1, c2, c3 = st.columns(3)
    use_std = c1.checkbox("Show std instead of mean", value=False)
    size = c2.slider("Marker size", 2, 100, 30)
    with c3:
        track_clim = clim_inputs("track", (50.0, 150.0))
    show(figures.radec_track_figure(get_matched(key), ctx.pols, source=ctx.source, use_std=use_std, size=size, clim=track_clim))


def _timing_for_window(ctx: Ctx, key: WindowKey):
    """Return ``(timing_info | None, applied_offset_s | None, error | None)``."""
    info, err = None, None
    try:
        info = get_timing(key, ctx.source, ctx.n_scans, ctx.n_el, ctx.max_offset_s)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    if ctx.override_ms is not None:
        return info, ctx.override_ms / 1e3, err
    if info is not None:
        return info, float(info["timing_offset_seconds"]), err
    return None, None, err


def view_peak_scans(ctx: Ctx) -> None:
    if not _need(ctx.peak, "peak-scan window"):
        return
    key = ctx.key(ctx.peak)
    w = load_window(key)
    _window_info(w)
    st.caption(f"{ctx.n_scans} scans: first {ctx.n_el} elevation, remaining {ctx.n_scans - ctx.n_el} azimuth (change under Options).")
    mode = st.radio(
        "Positions", ["Timing-corrected (spectrum cadence)", "Window-matched (pointing cadence)"], horizontal=True
    )
    c1, c2 = st.columns(2)
    size = c1.slider("Sky marker size", 5, 150, 50)
    with c2:
        clim = clim_inputs("peak", (30.0, 150.0))

    data = None
    if mode.startswith("Timing"):
        _, offset, err = _timing_for_window(ctx, key)
        if offset is None:
            st.warning(f"Timing offset unavailable ({err}); falling back to window-matched positions.")
        else:
            data = get_timing_corrected(key, offset, ctx.max_offset_s)
            st.caption(f"Applied timing offset: {offset * 1e3:.1f} ms")
    if data is None:
        data = get_matched(key)

    fig, info = figures.peak_scan_figure(
        data, ctx.source, ctx.pols, ctx.n_scans, ctx.n_el, clim=clim, size=size
    )
    show(fig)
    st.caption(f"Scan legs found by: {info['scan_split']} ({info['n_samples']:,} finite samples)")
    rows = []
    for name, fit, err in (("Dec (el scans)", info["fit_dec"], info["fit_dec_error"]), ("RA (az scans)", info["fit_ra"], info["fit_ra_error"])):
        if fit is None:
            rows.append({"axis": name, "error": err})
        else:
            rows.append({"axis": name, "A (K)": fit["A"], "mu (deg)": fit["mu_deg"], "sigma (deg)": fit["sigma_deg"], "FWHM (deg)": fit["FWHM_deg"], "baseline (K)": fit["baseline"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"RA/Dec were shifted by the mean per-leg peak offsets: dRA={info['ra_shift_deg']:.5f} deg, dDec={info['dec_shift_deg']:.5f} deg.")


def view_offsets(ctx: Ctx) -> None:
    if not _need(ctx.peak, "peak-scan window"):
        return
    key = ctx.key(ctx.peak)
    w = load_window(key)
    _window_info(w)
    matched = get_matched(key)

    st.subheader("Pointing offsets (peak per scan leg vs catalogue)")
    off = analysis.pointing_offsets(matched, ctx.source, ctx.n_scans, ctx.n_el)
    cols = st.columns(4)
    for col, (lab, k) in zip(cols, (("Az", "az_offset"), ("El", "el_offset"), ("RA", "ra_offset"), ("Dec", "dec_offset"))):
        col.metric(f"{lab} offset (mean |.|)", f"{off[k]:.4f} deg", help=f"{off[k] * 60:.2f} arcmin")
    st.caption(
        f"Scan legs by: {off.get('scan_split', 'n/a')}. Fixed telescope offsets "
        + ("were added to the mount az/el on load." if ctx.add_offset else "were NOT added on load.")
        + f" Reference values: +{io._POINTING_EL_OFFSET_DEG} deg el, +{io._POINTING_AZ_OFFSET_DEG} deg az."
    )
    c1, c2 = st.columns(2)
    c1.markdown("**Elevation legs**")
    c1.dataframe(
        pd.DataFrame({"leg": list(off["el_scans"]), "el offset (deg)": off["el_offsets"], "dec offset (deg)": off["dec_offsets"]}),
        hide_index=True, width="stretch",
    )
    c2.markdown("**Azimuth legs**")
    c2.dataframe(
        pd.DataFrame({"leg": list(off["az_scans"]), "az offset (deg)": off["az_offsets"], "ra offset (deg)": off["ra_offsets"]}),
        hide_index=True, width="stretch",
    )

    st.subheader("Timing offset (spectra clock vs pointing clock)")
    info, offset, err = _timing_for_window(ctx, key)
    if info is None:
        st.error(f"Timing estimate failed: {err}")
    else:
        cols = st.columns(5)
        cols[0].metric("Combined", f"{info['timing_offset_seconds'] * 1e3:.1f} ms")
        cols[1].metric("El legs", f"{info['el_timing_offset_seconds'] * 1e3:.1f} ms")
        cols[2].metric("Az legs", f"{info['az_timing_offset_seconds'] * 1e3:.1f} ms")
        cols[3].metric("Spatial az offset", f"{info['az_offset_deg']:.4f} deg")
        cols[4].metric("Spatial el offset", f"{info['el_offset_deg']:.4f} deg")
        st.caption("Positive offset: spectra clock is behind the pointing clock.")
        show(figures.timing_diagnostic_figure(w.obs, w.require_pointing(), info, ctx.source))
    if offset is not None:
        if ctx.override_ms is not None:
            st.info(f"Using manual timing offset {ctx.override_ms:.1f} ms for the comparison below.")
        tc = get_timing_corrected(key, offset, ctx.max_offset_s)
        st.subheader("Window-matched vs timing-corrected sky track")
        show(figures.timing_compare_figure(matched, tc, ctx.pols, source=ctx.source))


def _sky_center(ctx: Ctx, key: str) -> tuple[float, float]:
    s_ra, s_dec = Beam.get_source_radec(ctx.source)
    if st.radio("Patch centre", [f"Source ({ctx.source})", "Custom"], horizontal=True, key=f"{key}_c").startswith("Source"):
        return float(s_ra), float(s_dec)
    c1, c2 = st.columns(2)
    return (
        float(c1.number_input("Centre RA (deg)", 0.0, 360.0, float(s_ra), key=f"{key}_ra")),
        float(c2.number_input("Centre Dec (deg)", -90.0, 90.0, float(s_dec), key=f"{key}_dec")),
    )


def view_healpix(ctx: Ctx) -> None:
    if not _need(ctx.beam, "beam-map window"):
        return
    key = ctx.key(ctx.beam)
    w = load_window(key)
    _window_info(w)

    c1, c2, c3 = st.columns(3)
    nside = int(c1.selectbox("nside", [64, 128, 256, 512, 1024], index=3))
    stat = c2.radio("Statistic", ["mean", "std"], horizontal=True)
    freq_mode = c3.radio("Frequency", ["Band average", "Single channel"], horizontal=True)
    freq_index = None
    if freq_mode == "Single channel":
        freq_index = int(st.slider("Channel", 0, len(w.obs.freq) - 1, len(w.obs.freq) // 2))
        st.caption(f"{w.obs.freq[freq_index]:.3f} MHz")
    approx_gb = 12 * nside**2 * 8 * (3 + 2 * len(ctx.pols)) / 1e9
    if approx_gb > 1.5:
        st.warning(f"This map needs roughly {approx_gb:.1f} GB of RAM; consider a smaller nside or a single correlation.")
    c1, c2 = st.columns(2)
    patch_deg = float(c1.number_input("Patch size (deg, square)", 1.0, 60.0, 10.0))
    with c2:
        clim = clim_inputs("hp", (30.0, 110.0))
    center_ra, center_dec = _sky_center(ctx, "hp")

    hmap = get_healpix(key, nside, freq_index)
    st.caption(
        f"nside={nside}: {hmap.npix:,} pixels, pixel ~ {np.degrees(hp_resol(nside)):.3f} deg, "
        f"{int(np.count_nonzero(hmap.get_hit_count() > 0)):,} pixels hit"
    )
    st.subheader("Patch in RA/Dec")
    show(figures.healpix_patch_figure(hmap, ctx.pols, center_ra=center_ra, center_dec=center_dec, patch_deg=patch_deg, stat=stat, clim=clim))
    st.subheader("Full sky")
    show(figures.healpix_full_figure(hmap, ctx.pols, stat=stat, clim=clim))
    with st.expander("Gnomonic view (healpy.gnomview)"):
        if st.checkbox("Draw gnomview", value=False):
            labels = [pol_label(p) for p in ctx.pols]
            pol = ctx.pols[labels.index(st.selectbox("Correlation", labels, key="gnom_pol"))]
            reso = float(st.number_input("Resolution (arcmin/pixel)", 0.5, 20.0, 2.0))
            show(figures.healpix_gnomview_figure(hmap, pol, center_ra=center_ra, center_dec=center_dec, patch_deg=patch_deg, reso_arcmin=reso, clim=clim))


def hp_resol(nside: int) -> float:
    import healpy as hp

    return float(hp.nside2resol(nside))


def _fit_row(pol: str, res: dict) -> dict:
    if "error" in res:
        return {"correlation": pol_label(pol), "error": res["error"]}
    p = res["params"]
    return {
        "correlation": pol_label(pol), "n": res["n"], "A (K)": p["A"], "sigma (deg)": p["sigma_deg"],
        "FWHM (deg)": p["FWHM_deg"], "baseline (K)": p["baseline_k"],
    }


def view_beam(ctx: Ctx) -> None:
    if not _need(ctx.beam, "beam-map window"):
        return
    key = ctx.key(ctx.beam)
    w = load_window(key)
    _window_info(w)
    matched = get_matched(key)

    c1, c2, c3, c4 = st.columns(4)
    baseline = float(c1.slider("Baseline radius (deg)", 0.2, 4.0, 1.0, step=0.1, help="Mean of samples beyond this radius sets the floor"))
    x_max = float(c2.slider("Beam plot x-max (deg)", 1.0, 5.0, 5.0, step=0.5))
    normalize = c3.checkbox("Normalise beam (1 at 0, 0 at 5 deg)", value=True)
    components = c4.checkbox("Show Gaussian component", value=True)

    results = analysis.fit_beam_all(matched, ctx.pols, ctx.source, baseline)
    if len(ctx.pols) > 1:
        st.subheader("Fits for all correlations")
        st.dataframe(pd.DataFrame([_fit_row(p, results[p]) for p in ctx.pols]), hide_index=True, width="stretch")
        labels = [pol_label(p) for p in ctx.pols]
        pol = ctx.pols[labels.index(st.selectbox("Show detail for", labels))]
    else:
        pol = ctx.pols[0]

    if "error" in results[pol]:
        st.error(f"{pol_label(pol)}: {results[pol]['error']}")
        return
    off, vals = analysis.radial_profile(matched, pol, ctx.source)
    fig, params = figures.beam_fit_figure(
        off, vals, baseline_outer_deg=baseline, x_max_deg=x_max, normalize=normalize,
        show_components=components, title=f"{pol_label(pol)} beam profile ({ctx.source})",
    )
    for msg in results[pol].get("warnings", []):
        st.warning(msg)
    show(fig)
    cols = st.columns(5)
    cols[0].metric("Amplitude A", f"{params['A']:.1f} K")
    cols[1].metric("sigma", f"{params['sigma_deg']:.4f} deg")
    cols[2].metric("FWHM", f"{params['FWHM_deg']:.4f} deg")
    cols[3].metric("Half-power radius", f"{params['half_power_radius_deg']:.4f} deg")
    cols[4].metric("Baseline", f"{params['baseline_k']:.1f} K")

    st.subheader("Save / load fits")
    default_path = discovery.REPO_ROOT / "outputs" / f"beam_fit_{ctx.session.name}_{pol.rstrip('_')}.json"
    path = st.text_input("Save fit to", str(default_path))
    if st.button("Save fit"):
        payload = {k: v for k, v in params.items() if k not in ("x_fit", "y_fit_gaussian", "covariance")}
        payload.update(source=ctx.source, correlation=pol, session=ctx.session.name, data_file=Path(ctx.data_path).name)
        st.success(f"Saved {Beam.save_beam_fit(payload, Path(path))}")
    saved = sorted((discovery.REPO_ROOT / "outputs").glob("*.json"))
    if saved:
        pick = st.selectbox("Load saved fit", saved, format_func=lambda p: p.name)
        loaded = Beam.load_beam_fit(Path(pick))
        c1, c2 = st.columns([1, 1])
        c1.json({k: v for k, v in loaded.items() if k != "covariance"})
        try:
            with c2:
                show(figures.beam_approximation_figure(loaded, x_max_deg=x_max, normalize=normalize, show_components=components, title=Path(pick).name))
        except Exception as exc:  # noqa: BLE001
            c2.info(f"Cannot draw this file as a beam approximation: {exc}")


# =========================================================================== main
def main() -> None:
    ctx = build_sidebar()
    if ctx is None:
        st.info("Choose a data folder with at least one session, calibration file, pointing directory and spectrometer file.")
        return
    st.title("Skymap beam dashboard")
    st.caption(
        f"{ctx.session.name} | data: {Path(ctx.data_path).name} | cal: {Path(ctx.cal_path).name} | "
        f"correlation: {ctx.choice} | source: {ctx.source}"
    )
    view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view")
    fn = {
        VIEWS[0]: view_calibration,
        VIEWS[1]: view_single_scan,
        VIEWS[2]: view_peak_scans,
        VIEWS[3]: view_offsets,
        VIEWS[4]: view_healpix,
        VIEWS[5]: view_beam,
    }[view]
    run(fn, ctx)


main()
