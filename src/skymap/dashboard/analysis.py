"""
Thin, UI-independent wrappers around the existing skymap analysis functions.

Nothing here re-implements analysis: every function delegates to :mod:`skymap.io`,
:mod:`skymap.Beam` or :mod:`skymap.healmap`, adding only the bookkeeping the dashboard needs
(restricting results to the selected correlations, dropping unmatched pointings, ...).
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Sequence

import numpy as np
from matplotlib.figure import Figure

from skymap import Beam, io
from skymap.healmap import HealPixMap
from skymap.io import CAL_POL_NAMES, CalibratedSpec, HDF5Data, PointingData

DEFAULT_SOURCE = "3C353"


# --------------------------------------------------------------------------- helpers
def source_names() -> list[str]:
    """Unique calibrator names from ``calibrators.dat`` (catalogue order)."""
    seen: dict[str, None] = {}
    for name, _, _ in Beam.load_calibrators():
        seen.setdefault(name.strip(), None)
    return list(seen)


def timing_attribute(pols: Sequence[str]) -> str | None:
    """Brightness channel used for peak finding: the single selected pol, else all (None)."""
    return pols[0] if len(pols) == 1 else None


def scan_leg_args(n_scans: int, n_el_scans: int) -> dict:
    """``n_scans`` / ``el_scans`` / ``az_scans`` keyword arguments for the io functions."""
    if not 1 <= n_el_scans < n_scans:
        raise ValueError(f"n_el_scans must be in [1, {n_scans - 1}], got {n_el_scans}")
    return {
        "n_scans": int(n_scans),
        "el_scans": tuple(range(n_el_scans)),
        "az_scans": tuple(range(n_el_scans, n_scans)),
    }


def _values(arr) -> np.ndarray:
    arr = getattr(arr, "value", arr)
    return np.asarray(arr)


def amplitude_series(
    data: object,
    pols: Sequence[str],
    *,
    calibrated: bool = True,
    kind: str = "mean",
) -> np.ndarray:
    """
    Frequency-averaged brightness per time/pointing sample, averaged over ``pols``.

    With ``calibrated=True`` uses the first available of ``calibrated_spec_mean``,
    ``calibrated_spec``, ``spec_mean``, ``spec`` on ``data``; with ``calibrated=False`` only the
    raw ones. Complex (raw) values are converted to magnitude. ``kind="std"`` selects the
    per-pointing standard deviation containers produced by ``match_data_and_pointing``.
    """
    if kind == "std":
        candidates = ("calibrated_spec_std", "spec_std")
    elif calibrated:
        candidates = ("calibrated_spec_mean", "calibrated_spec", "spec_mean", "spec")
    else:
        candidates = ("spec_mean", "spec")
    source = None
    for name in candidates:
        source = getattr(data, name, None)
        if source is not None:
            break
    if source is None:
        raise ValueError("data has no spectra to average")
    series = []
    for pol in pols:
        arr = getattr(source, pol, None)
        if arr is None:
            continue
        arr = _values(arr)
        if np.iscomplexobj(arr):
            arr = np.abs(arr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            series.append(np.nanmean(arr.astype(float), axis=1))
    if not series:
        raise ValueError(f"None of {list(pols)} are available on the data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack(series), axis=0)


def restrict_matched(matched: HDF5Data, pols: Sequence[str]) -> HDF5Data:
    """
    Set correlations outside ``pols`` to ``None`` on the mean/std containers (in place).

    ``match_data_and_pointing`` fills unselected channels with NaN; removing them keeps the
    downstream "available channels" logic (offsets, HEALPix, beam fit) restricted to the user's
    choice.
    """
    keep = set(pols)
    for attr in ("calibrated_spec_mean", "calibrated_spec_std", "spec_mean", "spec_std"):
        src = getattr(matched, attr, None)
        if src is None:
            continue
        new = CalibratedSpec()
        for name in CAL_POL_NAMES:
            setattr(new, name, getattr(src, name, None) if name in keep else None)
        setattr(matched, attr, new)
    return matched


# --------------------------------------------------------------------------- matching / offsets
def match_window(obs: HDF5Data, pointing: PointingData, pols: Sequence[str]) -> HDF5Data:
    """Window-average spectra onto pointing samples (:func:`skymap.io.match_data_and_pointing`)."""
    return restrict_matched(io.match_data_and_pointing(obs, pointing), pols)


def pointing_offsets(matched: HDF5Data, source: str, n_scans: int, n_el_scans: int) -> dict:
    """Az/El and RA/Dec pointing offsets (:func:`skymap.io.get_pointing_offset`)."""
    return io.get_pointing_offset(matched, source, **scan_leg_args(n_scans, n_el_scans))


def timing_estimate(
    obs: HDF5Data,
    pointing: PointingData,
    source: str,
    pols: Sequence[str],
    n_scans: int,
    n_el_scans: int,
    max_offset_seconds: float | None = None,
) -> dict:
    """Spatial-then-timing offset estimate (:func:`io.estimate_timing_offset_after_spatial_correction`)."""
    kwargs = scan_leg_args(n_scans, n_el_scans)
    if max_offset_seconds is not None:
        kwargs["max_offset_seconds"] = float(max_offset_seconds)
    return io.estimate_timing_offset_after_spatial_correction(
        obs, pointing, source, attribute=timing_attribute(pols), **kwargs
    )


def timing_corrected(
    obs: HDF5Data,
    pointing: PointingData,
    pols: Sequence[str],
    *,
    timing_offset_seconds: float,
    max_offset_seconds: float | None = None,
) -> HDF5Data:
    """
    Spectrum-cadence Az/El/RA/Dec at ``data.time + timing_offset_seconds``
    (:func:`io.match_data_and_pointing_with_timing_offset` with a given offset).
    """
    kwargs = {}
    if max_offset_seconds is not None:
        kwargs["max_offset_seconds"] = float(max_offset_seconds)
    out = io.match_data_and_pointing_with_timing_offset(
        obs, pointing, timing_offset_seconds=float(timing_offset_seconds), **kwargs
    )
    return restrict_matched(out, pols)


# --------------------------------------------------------------------------- HEALPix
def finite_pointing_subset(matched: HDF5Data, pols: Sequence[str]) -> SimpleNamespace:
    """
    Matched data restricted to ``pols`` and to pointings with finite RA/Dec and values.

    ``HealPixMap.fill_from_pointing`` does not mask NaNs, so unmatched pointings (windows with no
    spectra) would otherwise poison every pixel they touch.
    """
    ra = np.asarray(matched.ra, dtype=float)
    dec = np.asarray(matched.dec, dtype=float)
    src = io.get_pol_source(matched, kind="mean")
    if src is None:
        raise ValueError("matched data has no mean spectra")
    arrays = {}
    mask = np.isfinite(ra) & np.isfinite(dec)
    for pol in pols:
        arr = getattr(src, pol, None)
        if arr is None:
            continue
        arr = _values(arr).astype(float)
        arrays[pol] = arr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mask &= np.isfinite(np.nanmean(arr, axis=1))
    if not arrays:
        raise ValueError(f"None of {list(pols)} are available on the matched data")
    sub = CalibratedSpec()
    for name in CAL_POL_NAMES:
        setattr(sub, name, arrays[name][mask] if name in arrays else None)
    return SimpleNamespace(ra=ra[mask], dec=dec[mask], calibrated_spec_mean=sub, n_used=int(mask.sum()))


def build_healpix(
    matched: HDF5Data,
    pols: Sequence[str],
    nside: int = 512,
    freq_index: int | None = None,
) -> HealPixMap:
    """Fill a :class:`HealPixMap` (one channel map per pol) from matched pointings."""
    sub = finite_pointing_subset(matched, pols)
    if sub.n_used == 0:
        raise ValueError("No pointings with finite RA/Dec and spectra in the beam window")
    hmap = HealPixMap(int(nside))
    hmap.fill_from_pointing_data(sub, attributes=[p for p in pols if getattr(sub.calibrated_spec_mean, p) is not None], freq_index=freq_index)
    return hmap


# --------------------------------------------------------------------------- beam fit
def radial_profile(
    matched: HDF5Data,
    pol: str,
    source: str,
    freq_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial offset from the source and brightness per pointing (finite samples only)."""
    off, vals = Beam.radial_offsets_from_source(
        matched, attribute=pol, source_name=source, freq_index=freq_index
    )
    ok = np.isfinite(off) & np.isfinite(vals)
    return off[ok], vals[ok]


def fit_beam(
    offset_deg: np.ndarray,
    values_k: np.ndarray,
    baseline_outer_deg: float = 1.0,
) -> tuple[dict, list[str]]:
    """
    :func:`skymap.Beam.fit_beam_gaussian` without creating a pyplot figure.

    Returns ``(params, warning_messages)``.
    """
    ax = Figure().add_subplot()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = Beam.fit_beam_gaussian(
            offset_deg,
            values_k,
            baseline_outer_deg=float(baseline_outer_deg),
            ax=ax,
            show=False,
        )
    return params, [str(w.message) for w in caught]


def fit_beam_all(
    matched: HDF5Data,
    pols: Sequence[str],
    source: str,
    baseline_outer_deg: float = 1.0,
    freq_index: int | None = None,
) -> dict[str, dict]:
    """Per-correlation fit results: ``{pol: {"params": ..., "n": ...}}`` or ``{"error": msg}``."""
    out: dict[str, dict] = {}
    for pol in pols:
        try:
            off, vals = radial_profile(matched, pol, source, freq_index)
            params, warns = fit_beam(off, vals, baseline_outer_deg)
            out[pol] = {"params": params, "n": int(off.size), "warnings": warns}
        except Exception as exc:  # noqa: BLE001 - reported to the user per correlation
            out[pol] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
