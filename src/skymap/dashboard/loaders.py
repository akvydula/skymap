"""
Light-weight, time-window aware loaders for the dashboard.

The raw spectrometer files are ~2 GB (``data/spec`` is ``(n_time, 10, n_freq)`` complex64), so the
dashboard never reads a whole file:

1. :func:`peek_obs_hdf5` reads only ``data/time`` (plus shapes) to learn the time span.
2. :func:`slice_from_wall_times` / :func:`slice_from_indices` turn a window into a ``slice``.
3. :func:`read_obs_window` reads just that slice, and only the requested correlations,
   into the existing :class:`skymap.io.HDF5Data` container.
4. :func:`calibrate_window` applies :func:`skymap.Calibrator.lab_cal` per selected correlation.

Time conversion follows :func:`skymap.io.read_obs_hdf5` (``time_unit`` attribute: ``"mjd"`` or
``"unix_ms"``; otherwise Unix milliseconds).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from astropy.time import Time

from skymap import Calibrator, io
from skymap.io import (
    CAL_POL_NAMES,
    CalData,
    CalibratedSpec,
    HDF5Data,
    PointingData,
    SpecData,
)

_UNIX_EPOCH_MJD = 40587.0
_MS_PER_DAY = 86_400_000.0


# --------------------------------------------------------------------------- time helpers
def _raw_time_to_mjd(raw: np.ndarray, time_unit: str | None) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    if time_unit == "mjd":
        return raw
    return raw / _MS_PER_DAY + _UNIX_EPOCH_MJD  # "unix_ms" or unlabelled (Unix ms)


def _raw_time_to_datetimes(raw: np.ndarray, time_unit: str | None) -> np.ndarray:
    """Vectorized conversion to an object array of naive-UTC ``datetime`` (as ``read_obs_hdf5``)."""
    raw = np.asarray(raw, dtype=float)
    if raw.size == 0:
        return np.array([], dtype=object)
    if time_unit == "mjd":
        return np.asarray(Time(raw, format="mjd", scale="utc").to_datetime(), dtype=object)
    ms = np.round(raw).astype("int64").astype("datetime64[ms]")
    return ms.astype("datetime64[us]").astype(object)


def mjd_to_naive_utc(mjd: float) -> datetime:
    """UTC MJD (days) -> naive UTC ``datetime``."""
    return Time(float(mjd), format="mjd", scale="utc").to_datetime()


def mjd_to_wall(mjd: float, tz: str) -> datetime:
    """UTC MJD (days) -> naive wall-clock ``datetime`` in zone ``tz`` (for date/time widgets)."""
    dt = Time(float(mjd), format="mjd", scale="utc").to_datetime(timezone=timezone.utc)
    return dt.astimezone(ZoneInfo(tz)).replace(tzinfo=None)


# --------------------------------------------------------------------------- peek + slicing
@dataclass(frozen=True)
class ObsPeek:
    """Cheap summary of an observation file (only the time axis was read)."""

    path: str
    n_time: int
    n_freq: int
    n_pol: int
    time_unit: str | None
    mjd: np.ndarray  # (n_time,) UTC MJD days
    has_spec: bool
    has_calibrated: bool

    @property
    def start_mjd(self) -> float:
        return float(self.mjd[0])

    @property
    def end_mjd(self) -> float:
        return float(self.mjd[-1])

    @property
    def start_utc(self) -> datetime:
        return mjd_to_naive_utc(self.start_mjd)

    @property
    def end_utc(self) -> datetime:
        return mjd_to_naive_utc(self.end_mjd)

    @property
    def cadence_s(self) -> float:
        if self.n_time < 2:
            return float("nan")
        return float(np.median(np.diff(self.mjd[: min(self.n_time, 10001)])) * 86400.0)


def peek_obs_hdf5(path: str | Path) -> ObsPeek:
    """Read only the time axis and dataset shapes of an observation HDF5 file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        grp = f["data"]
        time_ds = grp["time"]
        time_unit = time_ds.attrs.get("time_unit", None)
        if isinstance(time_unit, bytes):
            time_unit = time_unit.decode()
        mjd = _raw_time_to_mjd(time_ds[()], time_unit)
        has_spec = "spec" in grp
        has_cal = "calibrated_spec" in grp
        shape_ds = grp["spec"] if has_spec else grp["calibrated_spec"]
        n_time, n_pol, n_freq = shape_ds.shape
    return ObsPeek(
        path=str(path),
        n_time=int(n_time),
        n_freq=int(n_freq),
        n_pol=int(n_pol),
        time_unit=time_unit,
        mjd=mjd,
        has_spec=has_spec,
        has_calibrated=has_cal,
    )


def slice_from_indices(peek: ObsPeek, i0: int, i1: int) -> slice:
    """Clamp an index range to the file and return a ``slice`` (end-exclusive)."""
    i0 = int(np.clip(i0, 0, peek.n_time))
    i1 = int(np.clip(i1, 0, peek.n_time))
    if i1 < i0:
        raise ValueError(f"end index {i1} must be >= start index {i0}")
    return slice(i0, i1)


def slice_from_wall_times(
    peek: ObsPeek,
    start: datetime | np.datetime64 | str | float,
    end: datetime | np.datetime64 | str | float,
    tz: str | None,
) -> slice:
    """
    Index slice for the observation window ``[start, end)``.

    Same semantics as :func:`skymap.io._slice_from_observation_times` (naive datetimes are
    interpreted in ``tz``; floats are UTC MJD), but works on the peeked time axis so the file
    never has to be loaded.
    """
    m0 = io._observation_bound_to_mjd_utc(start, tz)
    m1 = io._observation_bound_to_mjd_utc(end, tz)
    if m1 < m0:
        raise ValueError("window end must be >= window start")
    i0 = int(np.searchsorted(peek.mjd, m0, side="left"))
    i1 = int(np.searchsorted(peek.mjd, m1, side="left"))
    return slice(i0, i1)


def window_utc_bounds(peek: ObsPeek, sl: slice) -> tuple[datetime, datetime]:
    """Naive-UTC first/last sample time of a slice (for pointing look-up)."""
    idx = range(*sl.indices(peek.n_time))
    if len(idx) == 0:
        raise ValueError("Selected time window contains no samples")
    return mjd_to_naive_utc(peek.mjd[idx[0]]), mjd_to_naive_utc(peek.mjd[idx[-1]])


def estimate_window_memory_mb(n_samples: int, n_pols: int, n_freq: int) -> float:
    """
    Rough peak RAM (MB) to read + calibrate ``n_samples`` for ``n_pols`` correlations.

    complex64 raw (8 B) + float64 calibrated (8 B) + calibration temporaries (~32 B) per value.
    """
    return n_samples * n_pols * n_freq * 48 / 1e6


# --------------------------------------------------------------------------- reading
class PartialSpecData(SpecData):
    """``SpecData`` where unselected correlations are ``None`` (safe ``repr``)."""

    def __repr__(self) -> str:
        parts = [
            f"{n}: {getattr(self, n).shape if getattr(self, n) is not None else None}"
            for n in CAL_POL_NAMES
        ]
        return f"PartialSpecData({', '.join(parts)})"


def _read_pol_block(ds: h5py.Dataset, sl: slice, pols: list[str]) -> dict[str, np.ndarray]:
    """Read ``ds[sl, pol, :]`` for the selected pols (one pass over all pols when ALL)."""
    idx = {name: i for i, name in enumerate(CAL_POL_NAMES)}
    out: dict[str, np.ndarray] = {}
    if len(pols) == len(CAL_POL_NAMES):
        stack = ds[sl, :, :]
        for name in pols:
            out[name] = np.ascontiguousarray(stack[:, idx[name], :])
    else:
        for name in pols:
            out[name] = np.ascontiguousarray(ds[sl, idx[name], :])
    return out


def read_obs_window(
    path: str | Path,
    time_slice: slice | None = None,
    pols: list[str] | None = None,
) -> HDF5Data:
    """
    Read ``time_slice`` of an observation file for the requested correlations only.

    Returns an :class:`skymap.io.HDF5Data` with ``spec`` (unselected correlations are ``None``)
    and, for derived files that already hold ``calibrated_spec``, a ``calibrated_spec`` for the
    selected correlations.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    pols = list(CAL_POL_NAMES) if pols is None else list(pols)
    bad = [p for p in pols if p not in CAL_POL_NAMES]
    if bad:
        raise ValueError(f"Unknown correlations {bad}; valid: {CAL_POL_NAMES}")
    sl = slice(None) if time_slice is None else time_slice
    if sl.step not in (None, 1):
        raise ValueError("time_slice must have step 1")

    with h5py.File(path, "r") as f:
        grp = f["data"]
        freq = grp["freq"][()] / 1e6
        time_ds = grp["time"]
        time_unit = time_ds.attrs.get("time_unit", None)
        if isinstance(time_unit, bytes):
            time_unit = time_unit.decode()
        time = _raw_time_to_datetimes(time_ds[sl], time_unit)

        spec = None
        if "spec" in grp:
            blocks = _read_pol_block(grp["spec"], sl, pols)
            spec = PartialSpecData(**{n: blocks.get(n) for n in CAL_POL_NAMES}, freq=freq, time=time)

        kwargs: dict = {}
        if "calibrated_spec" in grp:
            blocks = _read_pol_block(grp["calibrated_spec"], sl, pols)
            cal = CalibratedSpec()
            for name, arr in blocks.items():
                setattr(cal, name, arr if np.any(np.isfinite(arr)) else None)
            if any(getattr(cal, n) is not None for n in CAL_POL_NAMES):
                kwargs["calibrated_spec"] = cal

    return HDF5Data(freq=freq, time=time, spec=spec, **kwargs)


def load_cal(path: str | Path) -> CalData:
    """Read a calibration file (gain / Te per correlation)."""
    return io.read_cal_hdf5(path)


def calibrate_window(obs: HDF5Data, cal: CalData, pols: list[str]) -> HDF5Data:
    """
    Apply :func:`skymap.Calibrator.lab_cal` to each selected correlation (in place).

    Correlations that are already calibrated (derived files) or absent are skipped.
    """
    if len(cal.freq) != len(obs.freq):
        raise ValueError(
            f"Calibration file has {len(cal.freq)} channels but the data has {len(obs.freq)}"
        )
    if obs.spec is None:
        return obs
    for name in pols:
        existing = getattr(obs, "calibrated_spec", None)
        if existing is not None and getattr(existing, name, None) is not None:
            continue
        if getattr(obs.spec, name, None) is None:
            continue
        Calibrator.lab_cal(obs, cal, attribute=name)
    return obs


def load_pointing_window(
    pointing_dir: str | Path,
    start_utc: datetime,
    end_utc: datetime,
    *,
    add_pointing_offset: bool = True,
    pad_s: float = 0.0,
) -> PointingData:
    """
    Pointing samples for ``[start_utc, end_utc]`` via :func:`skymap.io.get_pointing_data`
    (which picks only the FITS files covering the window).
    """
    from datetime import timedelta

    pad = timedelta(seconds=float(pad_s))
    return io.get_pointing_data(
        pointing_dir,
        start_utc - pad,
        end_utc + pad,
        add_pointing_offset=add_pointing_offset,
    )
