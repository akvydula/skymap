"""
Utility functions for reading HDF5 files and plotting polarization data.

"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence, Union
from zoneinfo import ZoneInfo
from astropy.io import fits
import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time

from skymap.utils import _time_to_mjd


class SpecData:
    """Container for polarization data with attribute access."""
    
    def __init__(self, AA_: np.ndarray, BB_: np.ndarray, 
        CC_: np.ndarray, DD_: np.ndarray, AB_: np.ndarray, 
        BC_: np.ndarray, CD_: np.ndarray, AC_: np.ndarray, 
        BD_: np.ndarray, AD_: np.ndarray, 
        freq: np.ndarray = None, time: np.ndarray = None):
        """
        Initialize SpecData with polarization arrays.
        
        Parameters
        ----------
        AA*,BB*, CC*, DD*,AB*, BC*,CD*,AC*,BD*,AD*: These are the polarization data for the different polarizations.

        AA* (AA_): np.ndarray
            AA* polarization data (2D array: time x frequency)
        BB* (BB_): np.ndarray
            BB* polarization data (2D array: time x frequency)
        CC* (CC_): np.ndarray
            CC* polarization data (2D array: time x frequency)
        DD* (DD_): np.ndarray
            DD* polarization data (2D array: time x frequency)
        AB* (AB_): np.ndarray
            AB* polarization data (2D array: time x frequency)
        BC* (BC_): np.ndarray
            BC* polarization data (2D array: time x frequency)
        CD* (CD_): np.ndarray
            CD* polarization data (2D array: time x frequency)
        AC* (AC_): np.ndarray
            AC* polarization data (2D array: time x frequency)
        BD* (BD_): np.ndarray
            BD* polarization data (2D array: time x frequency)
        AD* (AD_): np.ndarray
            AD* polarization data (2D array: time x frequency)
        """
        self.AA_ = AA_
        self.BB_ = BB_
        self.CC_ = CC_
        self.DD_ = DD_
        self.AB_ = AB_
        self.BC_ = BC_
        self.CD_ = CD_
        self.AC_ = AC_
        self.BD_ = BD_
        self.AD_ = AD_
        self.freq = freq
        self.time = time
    
    def __repr__(self) -> str:
        return (
            f"SpecData(AA_: {self.AA_.shape}, BB_: {self.BB_.shape}, "
            f"CC_: {self.CC_.shape}, DD_: {self.DD_.shape}, "
            f"AB_: {self.AB_.shape}, BC_: {self.BC_.shape}, "
            f"CD_: {self.CD_.shape}, AC_: {self.AC_.shape}, "
            f"BD_: {self.BD_.shape}, AD_: {self.AD_.shape}, "
            f"freq: {self.freq.shape}, time: {self.time.shape})"
        
        )


class CalibratedSpec:
    """
    Container for calibrated polarization data with the same attribute names as SpecData.
    Access as data.calibrated_spec.AA_, data.calibrated_spec.AB_, etc.
    Uncalibrated channels are None until set by lab_cal.
    """
    def __init__(self):
        for name in CAL_POL_NAMES:
            setattr(self, name, None)

    def __repr__(self) -> str:
        parts = [f"{n}: {getattr(self, n).shape if getattr(self, n) is not None else None}" for n in CAL_POL_NAMES]
        return f"CalibratedSpec({', '.join(parts[:3])}...)"


class HDF5Data:
    """Container for HDF5 data with attribute-style access."""
    
    def __init__(
        self,
        freq: np.ndarray,
        time: np.ndarray | list[datetime],
        spec: SpecData | None = None,
        **kwargs
    ):
        """
        Initialize HDF5Data with frequency, time, and spec data.
        
        Parameters
        ----------
        freq : np.ndarray
            Frequency array (in MHz)
        time : np.ndarray or list[datetime]
            Time array (can be datetime objects if converted from MJD)
        spec : SpecData, optional
            Correlation data container
        **kwargs
            Additional data attributes
        """
        self.freq = freq
        self.time = np.asarray(time) if not isinstance(time, np.ndarray) else time
        self.spec = spec
        
        # Add any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self) -> str:
        time_info = f"time: {self.time.shape}"
        if len(self.time) > 0 and isinstance(self.time[0], datetime):
            time_info += f" (datetime: {self.time[0]} to {self.time[-1]})"
        attrs = [f"freq: {self.freq.shape}", time_info]
        if self.spec is not None:
            attrs.append(f"spec: {self.spec}")
        for key in dir(self):
            if not key.startswith('_') and key not in ['freq', 'time', 'spec']:
                value = getattr(self, key)
                if isinstance(value, np.ndarray):
                    attrs.append(f"{key}: {value.shape}")
        return f"HDF5Data({', '.join(attrs)})"


def read_obs_hdf5(
    file_path: str | Path,
    skip_errors: bool = True,
    time_slice: slice | None = None,
    time_mjd: bool = False
) -> HDF5Data:
    """
    Read an HDF5 file and return the data as an HDF5Data object with attribute access.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.
    skip_errors : bool, default=True
        If True, skip datasets that cannot be read and continue with others.
        If False, raise an error when a dataset cannot be read.

    time_slice : slice, optional
        Slice to apply to time dimension (e.g., slice(0, 38000)). 
        Default is None. Set to slice(0, 38000) to read first 38000 data. Set to None to read all data.
    time_mjd : bool, optional
        Used only when the file has no "time_unit" attribute. If True, treat time as MJD (days);
        if False, treat time as Unix timestamp in milliseconds. Files written by write_obs_hdf5
        store time in Unix ms and set time_unit="unix_ms", so the correct conversion is used automatically.
    Returns
    -------
    HDF5Data
        HDF5Data object with attribute-style access:
        - data.freq : frequency array (in MHz)
        - data.time : time array
        - data.spec.AA* : AA* Correlation data
        - data.spec.BB* : BB* Correlation data
        - data.spec.CC* : CC* Correlation data
        - data.spec.DD* : DD* Correlation data
        - data.spec.AB* : AB* Correlation data
        - data.spec.BC* : BC* Correlation data
        - data.spec.CD* : CD* Correlation data
        - data.spec.AC* : AC* Correlation data
        - data.spec.BD* : BD* Correlation data
        - data.spec.AD* : AD* Correlation data

    """
    file_path = Path(file_path)
    
    # check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    f = h5py.File(file_path, 'r')
    
    try:
        grp = f['data']
        # Read frequency data (convert to MHz)
        freq = grp['freq'][()] / 1e6

        if time_slice is not None:
            time_data = grp['time'][time_slice]
        else:
            time_data = grp['time'][()]

        # Convert time_data to datetime (naive UTC). Use stored unit so write/read round-trip matches.
        time_unit = grp["time"].attrs.get("time_unit", None)
        if time_unit == "mjd":
            # MJD (days) UTC -> naive UTC datetime
            time_data = np.array([Time(ts, format="mjd", scale="utc").datetime for ts in time_data])
        elif time_unit == "unix_ms":
            # Unix ms -> naive UTC datetime (avoid local-time fromtimestamp)
            time_data = np.array([
                datetime.fromtimestamp(ts / 1000, tz=timezone.utc).replace(tzinfo=None) for ts in time_data
            ])
        elif time_mjd:
            time_data = np.array([Time(ts, format="mjd", scale="utc").datetime for ts in time_data])
        else:
            time_data = np.array([
                datetime.fromtimestamp(ts / 1000, tz=timezone.utc).replace(tzinfo=None) for ts in time_data
            ])

        spec = None
        if 'spec' in grp:
            if time_slice is not None:
                spec_data = grp['spec'][time_slice, :, :]
            else:
                spec_data = grp['spec'][()]
            spec = SpecData(**{name: spec_data[:, i, :] for i, name in enumerate(CAL_POL_NAMES)})

        kwargs = {}
        if 'calibrated_spec' in grp:
            if time_slice is not None:
                cal_stack = grp['calibrated_spec'][time_slice, :, :]
            else:
                cal_stack = grp['calibrated_spec'][()]
            cal = CalibratedSpec()
            for i, name in enumerate(CAL_POL_NAMES):
                ch = cal_stack[:, i, :]
                setattr(cal, name, ch if np.any(np.isfinite(ch)) else None)
            kwargs['calibrated_spec'] = cal

        for key in ('ra', 'dec', 'el', 'az'):
            if key in grp:
                kwargs[key] = grp[key][time_slice] if time_slice is not None else grp[key][()]

        return HDF5Data(freq=freq, time=time_data, spec=spec, **kwargs)

    finally:
        if f is not None:
            f.close()




def write_obs_hdf5(data: HDF5Data, file_path: str | Path) -> None:
    """
    Write an HDF5 observation file with whatever information is present in the object.

    Writes at any processing stage: only freq, time, and spec are required;
    calibrated_spec, ra, dec, el, az are written if present (otherwise omitted or None).
    Layout matches read_obs_hdf5: group "data" with datasets freq (Hz), time (MJD in days, with
    time_unit="mjd"), spec (n_time, 10, n_freq), and optionally calibrated_spec, ra, dec, el, az.
    Times are stored in MJD (UTC) so write/read round-trip preserves the same time axis.

    Parameters
    ----------
    data : HDF5Data
        Observation data (must have freq and time). spec, calibrated_spec, ra, dec, el, az
        are written only when present.
    file_path : str or Path
        Output HDF5 path.
    """
    file_path = Path(file_path)
    time_arr = np.asarray(data.time)
    if len(time_arr) == 0:
        raise ValueError("data.time must not be empty to write obs HDF5")
    n_time = len(time_arr)
    n_freq = len(data.freq)

    with h5py.File(file_path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("freq", data=np.asarray(data.freq, dtype=float) * 1e6)
        time_mjd = _time_to_mjd(time_arr)
        dset_time = grp.create_dataset("time", data=time_mjd.astype(float))
        dset_time.attrs["time_unit"] = "mjd"  # MJD (days) UTC; read_obs_hdf5 converts to datetime

        if data.spec is not None:
            spec_stack = np.stack(
                [getattr(data.spec, name) for name in CAL_POL_NAMES],
                axis=1,
            )
            grp.create_dataset("spec", data=spec_stack)

        cal = getattr(data, "calibrated_spec", None)
        if cal is not None:
            ref_shape = None
            for name in CAL_POL_NAMES:
                v = getattr(cal, name, None)
                if v is not None:
                    ref_shape = (n_time, n_freq)
                    break
            if ref_shape is not None:
                cal_stack = np.full((n_time, len(CAL_POL_NAMES), n_freq), np.nan, dtype=float)
                for i, name in enumerate(CAL_POL_NAMES):
                    v = getattr(cal, name, None)
                    if v is not None:
                        cal_stack[:, i, :] = getattr(v, "value", v) if hasattr(v, "unit") else v
                grp.create_dataset("calibrated_spec", data=cal_stack)

        for key in ("ra", "dec", "el", "az"):
            val = getattr(data, key, None)
            if val is not None and isinstance(val, np.ndarray):
                grp.create_dataset(key, data=np.asarray(val, dtype=float))


# Same polarization order as SpecData (used for cal gain/te per channel)
CAL_POL_NAMES = ['AA_', 'BB_', 'CC_', 'DD_', 'AB_', 'BC_', 'CD_', 'AC_', 'BD_', 'AD_']


def get_pol_source(data: object, *, kind: Literal["mean", "std"] = "mean") -> object | None:
    """
    Return the polarization summary container on a matched pointing dataset.

    Expects output from `match_data_and_pointing`, which attaches:
    - calibrated_spec_mean / calibrated_spec_std (preferred), OR
    - spec_mean / spec_std (fallback)
    """
    if kind == "mean":
        return getattr(data, "calibrated_spec_mean", None) or getattr(data, "spec_mean", None)
    return getattr(data, "calibrated_spec_std", None) or getattr(data, "spec_std", None)


def get_available_pol_names(data: object, *, kind: Literal["mean", "std"] = "mean") -> list[str]:
    """Return pol channel names that exist and have data on the given object."""
    source = get_pol_source(data, kind=kind)
    if source is None:
        return []
    return [n for n in CAL_POL_NAMES if getattr(source, n, None) is not None]


class PolChannels:
    """Container for per-polarization arrays (e.g. gain or Te), with attribute access like SpecData (AA_, BB_, ...)."""
    def __init__(self, **kwargs: np.ndarray):
        for name in CAL_POL_NAMES:
            setattr(self, name, kwargs[name])

    def __repr__(self) -> str:
        return f"PolChannels(AA_: {self.AA_.shape}, BB_: {self.BB_.shape}, ...)"


class CalData:
    """Container for calibration data with attribute access (cal_data.gain.AA_, cal_data.te.BB_, etc.)."""
    def __init__(self, freq: np.ndarray, te: PolChannels, gain: PolChannels):
        """
        Initialize CalData with Gain (G) and Effective noise temperature (T_e) per polarization.
        Parameters
        ----------
        freq : np.ndarray
            Frequency channels (in MHz) - typically 256 channels
        te : PolChannels
            Effective noise temperature (Te) in Kelvin; access as cal_data.te.AA_, cal_data.te.BB_, etc.
        gain : PolChannels
            Gain (G) in linear scale; access as cal_data.gain.AA_, cal_data.gain.BB_, etc.
        """
        self.freq = freq
        self.te = te
        self.gain = gain

    def __repr__(self) -> str:
        return f"CalData(freq: {self.freq.shape}, te: PolChannels(...), gain: PolChannels(...))"



def read_cal_hdf5(file_path: str | Path) -> CalData:   
    """
    Read a calibration HDF5 file and return the data as a CalData object with attribute access.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.
    """
    file_path = Path(file_path)
    freq = np.linspace(310e6 - 20.48e6 / 2, 310e6 + 20.48e6 / 2, 256)  # 310 MHz, 20.48 MHz BW, 256 channels
    with h5py.File(file_path, 'r') as f:
        te_2d = f['te'][()]
        gain_2d = f['gain'][()]
    # te/gain shape: (10, n_freq) -> PolChannels with 1D array per polarization
    te_ch = PolChannels(**{name: te_2d[i] for i, name in enumerate(CAL_POL_NAMES)})
    gain_ch = PolChannels(**{name: gain_2d[i] for i, name in enumerate(CAL_POL_NAMES)})
    return CalData(freq=freq, te=te_ch, gain=gain_ch)


def _observation_bound_to_mjd_utc(
    value: datetime | np.datetime64 | str | float,
    tz: str | timezone | ZoneInfo | None,
) -> float:
    """Convert a wall-clock bound or MJD to UTC MJD (days) for time-axis matching."""
    if isinstance(value, (float, np.floating, int, np.integer)):
        v = float(value)
        if 4e4 <= v < 7e5:
            return v
        raise ValueError(
            f"Numeric observation bound must be MJD days in [40000, 700000), got {v!r}"
        )

    if isinstance(value, np.datetime64):
        return float(Time(value).mjd)

    if isinstance(value, str):
        s = value.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    elif isinstance(value, datetime):
        dt = value
    else:
        return float(Time(value).utc.mjd)

    if dt.tzinfo is None:
        if tz is None:
            raise ValueError(
                "tz is required when start/end are naive datetimes or timezone-naive ISO strings"
            )
        zi = ZoneInfo(tz) if isinstance(tz, str) else tz
        dt = dt.replace(tzinfo=zi)
    return float(Time(dt).utc.mjd)


def _slice_from_observation_times(
    data: HDF5Data,
    start: datetime | np.datetime64 | str | float,
    end: datetime | np.datetime64 | str | float,
    tz: str | timezone | ZoneInfo | None,
) -> slice:
    """Map observation [start, end) in the given timezone (for naive times) to a time index slice.

    Notes
    -----
    This uses **end-exclusive** semantics to match normal Python slicing:
    ``data.time[slice(i0, i1)]`` includes indices ``i0..i1-1``.
    """
    if len(data.time) == 0:
        return slice(0, 0)
    
    times64 = np.asarray(data.time, dtype="datetime64[ns]")
    start_dt = _mjd_utc_to_dt64ns(_observation_bound_to_mjd_utc(start, tz))
    end_dt = _mjd_utc_to_dt64ns(_observation_bound_to_mjd_utc(end, tz))
    if start_dt > end_dt:
        raise ValueError(f"start {start_dt} must be <= end {end_dt} (UTC)")
    i0 = int(np.searchsorted(times64, start_dt, side="left"))
    i1 = int(np.searchsorted(times64, end_dt, side="left"))
    return slice(i0, i1)


def _mjd_utc_to_dt64ns(mjd: float) -> np.datetime64:
    """Convert UTC MJD (days) to numpy datetime64[ns] (UTC)."""
    dt = Time(mjd, format="mjd", scale="utc").to_datetime(timezone=timezone.utc)
    # Store as UTC-naive datetime64 (we consistently treat stored datetimes as UTC).
    return np.datetime64(dt.replace(tzinfo=None), "ns")


def get_slice_from_time(
    data: HDF5Data,
    time_slice: slice | None = None,
    *,
    start: datetime | np.datetime64 | str | float | None = None,
    end: datetime | np.datetime64 | str | float | None = None,
    tz: str | timezone | ZoneInfo | None = None,
) -> HDF5Data:
    """
    Get a slice of the data from the time slice. Slices time and all
    time-varying attributes (spec, calibrated_spec if present).

    Parameters
    ----------
    data : HDF5Data
        Input data.
    time_slice : slice, optional
        Slice to apply along the time dimension (e.g. slice(0, 1000)).
        Use this **or** ``start``/``end``, not both.
    start, end : datetime, numpy.datetime64, str, or float, optional
        Observation window in wall time when ``time_slice`` is omitted.
        ``float`` values are interpreted as MJD (days, UTC). Strings use
        :func:`datetime.fromisoformat` (append ``Z`` for UTC). For naive
        datetimes or naive ISO strings, ``tz`` is required.
    tz : str, datetime.timezone, or zoneinfo.ZoneInfo, optional
        IANA zone name (e.g. ``\"America/New_York\"``) or timezone object.
        Used only when ``start``/``end`` are naive wall times.

    Returns
    -------
    HDF5Data
        New HDF5Data with time, spec, and calibrated_spec (if present) sliced.
    """
    if time_slice is not None and (start is not None or end is not None):
        raise ValueError("Pass either time_slice or start/end, not both.")
    if time_slice is None:
        if start is None or end is None:
            raise ValueError("Provide time_slice or both start and end.")
        time_slice = _slice_from_observation_times(data, start, end, tz)

    # Slice time
    time_sliced = data.time[time_slice]

    # Slice spec: each polarization is (n_time, n_freq)
    spec_sliced = None
    if data.spec is not None:
        spec_sliced = SpecData(**{
            name: getattr(data.spec, name)[time_slice, :] for name in CAL_POL_NAMES
        })

    kwargs = {}
    # Slice calibrated_spec if present (each channel when not None)
    if getattr(data, "calibrated_spec", None) is not None:
        cal = data.calibrated_spec
        cal_sliced = CalibratedSpec()
        for name in CAL_POL_NAMES:
            val = getattr(cal, name, None)
            if val is not None:
                setattr(cal_sliced, name, val[time_slice, :])
        kwargs["calibrated_spec"] = cal_sliced

    return HDF5Data(freq=data.freq, time=time_sliced, spec=spec_sliced, **kwargs)


class PointingData:
    """Container for pointing data with attribute access."""
    def __init__(self, dmjd: np.ndarray, az: np.ndarray, el: np.ndarray, ra: np.ndarray, dec: np.ndarray, **kwargs):
        self.dmjd = dmjd
        self.az = az
        self.el = el
        self.ra = ra
        self.dec = dec
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"PointingData(dmjd: {self.dmjd.shape}, az: {self.az.shape}, el: {self.el.shape}, ra: {self.ra.shape}, dec: {self.dec.shape}, **kwargs)"


def read_pointing_fits(file_path: str | Path) -> PointingData:
    """
    Read a pointing data FITS file and return the data as a PointingData object.
    Parameters
    ----------
    file_path : str or Path
        Path to the FITS file.
    Returns
    -------
    PointingData
        PointingData object with attribute access.
    """
    with fits.open(file_path) as hdul:
        dmjd = hdul['ANTPOSGR'].data['DMJD']
        az = hdul['ANTPOSGR'].data['MNT_AZ']
        el = hdul['ANTPOSGR'].data['MNT_EL']
        ra = hdul['ANTPOSGR'].data['RAJ2000']
        dec = hdul['ANTPOSGR'].data['DECJ2000']
        
        return PointingData(dmjd=dmjd, az=az, el=el, ra=ra, dec=dec)

# Filename pattern: YYYY_MM_DD_HH:MM:SS.fits (UTC time in filename; colons in time part)
_POINTING_FILENAME_PATTERN = re.compile(
    r"^(\d{4})_(\d{2})_(\d{2})_(\d{2}):(\d{2}):(\d{2})\.fits$"
)


def _parse_pointing_filename_utc(path: Path) -> datetime | None:
    """
    Parse UTC time from pointing filename YYYY_MM_DD_HH:MM:SS.fits.
    E.g. 2026_01_16_18:38:48.fits -> datetime(2026, 1, 16, 18, 38, 48).
    Returns None if filename does not match.
    """
    m = _POINTING_FILENAME_PATTERN.match(path.name)
    if m is None:
        return None
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s)


def _list_pointing_files_with_times(datadir: str | Path) -> list[tuple[datetime, Path]]:
    """Return all pointing FITS files in datadir with parsed filename UTC time, sorted."""
    datadir = Path(datadir)
    items: list[tuple[datetime, Path]] = []
    for path in datadir.glob("*.fits"):
        t = _parse_pointing_filename_utc(path)
        if t is None:
            continue
        items.append((t, path))
    items.sort(key=lambda x: x[0])
    return items


def _to_utc_naive(dt: datetime) -> datetime:
    """Normalize datetime to naive UTC (assume naive inputs are already UTC)."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _dt_utc_to_mjd(dt: datetime) -> float:
    """Convert a (possibly tz-aware) datetime to UTC MJD days. Naive datetimes are treated as UTC."""
    dt_utc = _to_utc_naive(dt).replace(tzinfo=timezone.utc)
    return float(Time(dt_utc).mjd)


_MAX_POINTING_SPILL_FILES = 12
_POINTING_EL_OFFSET_DEG = 0.06945
_POINTING_AZ_OFFSET_DEG = 0.01011


def _utc_naive_floor_minute(dt: datetime) -> datetime:
    """UTC-naive datetime truncated to the minute (for coarse spill heuristics)."""
    d = _to_utc_naive(dt)
    return d.replace(second=0, microsecond=0)


def _antposgr_nrows(path: str | Path) -> int:
    """Number of rows in ANTPOSGR without loading the table (fast empty check)."""
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        return int(hdul["ANTPOSGR"].header.get("NAXIS2", 0) or 0)


def _select_pointing_paths_covering_window(
    all_with_times: list[tuple[datetime, Path]],
    start_utc: datetime,
    end_utc: datetime,
    *,
    start_mjd: float,
    end_mjd: float,
    max_spill: int = _MAX_POINTING_SPILL_FILES,
) -> tuple[list[str], dict[str, Any]]:
    """
    Choose FITS paths whose tabulated DMJD can cover [start_mjd, end_mjd].

    Filename timestamps are chunk *starts*; samples can extend before the first in-window
    filename (need older chunks, skipping empty tables) and can use the first chunk
    *after* end_utc when its DMJD still falls inside the requested wall-time window.
    """
    meta: dict[str, Any] = {
        "i_first": None,
        "i_last": None,
        "core": [],
        "spill_start": [],
        "spill_end": [],
        "skipped_empty": [],
    }
    if not all_with_times:
        return [], meta

    file_dt64 = np.array(
        [np.datetime64(_to_utc_naive(t), "s") for t, _ in all_with_times],
        dtype="datetime64[s]",
    )
    start64 = np.datetime64(start_utc, "s")
    end64 = np.datetime64(end_utc, "s")
    n = len(all_with_times)
    i_first = int(np.searchsorted(file_dt64, start64, side="left"))
    i_last = int(np.searchsorted(file_dt64, end64, side="right")) - 1
    meta["i_first"] = i_first
    meta["i_last"] = i_last

    lo = int(max(0, i_first))
    hi = int(min(n - 1, i_last))
    core: list[str] = []
    if lo <= hi:
        core = [str(all_with_times[i][1].resolve()) for i in range(lo, hi + 1)]
    meta["core"] = [Path(p).name for p in core]

    spill_start: list[str] = []
    # If the requested start is already at or after the first core chunk's filename
    # (same minute or later), older chunks cannot contribute without reopening an
    # earlier scan — skip backward spill so we do not open empty predecessors
    # (e.g. 17:49) when the user sets start to 18:04 and the first file is 18:04:07.
    need_spill_before = True
    if lo <= hi and lo < n:
        first_core_name_t = _to_utc_naive(all_with_times[lo][0])
        if _utc_naive_floor_minute(start_utc) >= _utc_naive_floor_minute(first_core_name_t):
            need_spill_before = False
            meta["spill_start_skipped"] = (
                "start_utc is on or after the first core file's name (minute floor); "
                "backward spill not needed"
            )
    if need_spill_before:
        j = i_first - 1
        hops = 0
        while j >= 0 and hops < max_spill:
            hops += 1
            path = all_with_times[j][1]
            if _antposgr_nrows(path) == 0:
                meta["skipped_empty"].append(path.name)
                j -= 1
                continue
            pd = read_pointing_fits(path)
            d = np.asarray(pd.dmjd, dtype=float)
            dmax, dmin = float(np.max(d)), float(np.min(d))
            if dmax < start_mjd:
                j -= 1
                continue
            spill_start.insert(0, str(path.resolve()))
            if dmin <= start_mjd:
                break
            j -= 1

    spill_end: list[str] = []
    j = i_last + 1
    hops = 0
    while j < n and hops < max_spill:
        hops += 1
        path = all_with_times[j][1]
        if _antposgr_nrows(path) == 0:
            meta["skipped_empty"].append(path.name)
            j += 1
            continue
        pd = read_pointing_fits(path)
        d = np.asarray(pd.dmjd, dtype=float)
        dmin, dmax = float(np.min(d)), float(np.max(d))
        if dmin > end_mjd:
            break
        spill_end.append(str(path.resolve()))
        if dmax >= end_mjd:
            break
        j += 1

    meta["spill_start"] = [Path(p).name for p in spill_start]
    meta["spill_end"] = [Path(p).name for p in spill_end]

    combined = spill_start + core + spill_end
    ordered = list(dict.fromkeys(combined))
    return ordered, meta


def read_pointing_files(pointing_files: list[str]) -> PointingData:
    """
    Read the pointing files and combine them into a single PointingData object.
    Parameters
    ----------
    pointing_files : list[str]
        List of pointing file paths.
    Returns
    -------
    PointingData
        Single PointingData object with all rows from all files concatenated
        (dmjd, az, el, ra, dec in time order).
    """
    if not pointing_files:
        raise ValueError("pointing_files must not be empty")
    parts = [read_pointing_fits(f) for f in pointing_files]
    parts = [p for p in parts if len(np.asarray(p.dmjd)) > 0]
    if not parts:
        raise ValueError(
            "All pointing FITS had no ANTPOSGR rows (empty tables); nothing to concatenate"
        )
    return PointingData(
        dmjd=np.concatenate([p.dmjd for p in parts]),
        az=np.concatenate([p.az for p in parts]),
        el=np.concatenate([p.el for p in parts]),
        ra=np.concatenate([p.ra for p in parts]),
        dec=np.concatenate([p.dec for p in parts]),
    )

def get_pointing_data(
    datadir: str | Path,
    start_utc: datetime,
    end_utc: datetime,
    *,
    add_pointing_offset: bool = True,
) -> PointingData:
    """
    Get pointing data for files within a UTC time range.

    Parameters
    ----------
    datadir : str or Path
        Data directory with pointing FITS files (names like YYYY_MM_DD_HH_MM_SS.fits).
    start_utc : datetime
        Start of time range (UTC), inclusive.
    end_utc : datetime
        End of time range (UTC), inclusive.
    add_pointing_offset : bool, optional
        If True, apply fixed telescope pointing offsets to returned samples:
        +0.06945 deg in elevation and +0.01011 deg in azimuth.

    Returns
    -------
    PointingData
        PointingData object combining all files in the range.
    """
    start_utc = _to_utc_naive(start_utc)
    end_utc = _to_utc_naive(end_utc)
    if start_utc > end_utc:
        raise ValueError(f"start_utc must be <= end_utc, got {start_utc} and {end_utc}")

    start_mjd = _dt_utc_to_mjd(start_utc)
    end_mjd = _dt_utc_to_mjd(end_utc)
    datadir = Path(datadir)
    if not datadir.exists():
        raise FileNotFoundError(f"Data directory not found: {datadir}")

    all_with_times = _list_pointing_files_with_times(datadir)
    pointing_files, sel_meta = _select_pointing_paths_covering_window(
        all_with_times,
        start_utc,
        end_utc,
        start_mjd=start_mjd,
        end_mjd=end_mjd,
    )

    if len(pointing_files) == 0:
        raise FileNotFoundError(f"No pointing files found in {datadir} between {start_utc} and {end_utc}")

    n_core = len(sel_meta["core"])
    n_sb = len(sel_meta["spill_start"])
    n_se = len(sel_meta["spill_end"])
    print(
        f"Pointing: {len(pointing_files)} file(s) "
        f"(core by filename={n_core}, spill_before={n_sb}, spill_after={n_se}) "
        f"for {start_utc} .. {end_utc} UTC"
    )
    if sel_meta["skipped_empty"]:
        uq = sorted(set(sel_meta["skipped_empty"]))
        print(f"  Skipped {len(uq)} empty ANTPOSGR table(s): {', '.join(uq[:5])}" + (" ..." if len(uq) > 5 else ""))
    if sel_meta["spill_start"]:
        print(f"  spill_before: {sel_meta['spill_start']}")
    if sel_meta["spill_end"]:
        print(f"  spill_after: {sel_meta['spill_end']}")
    if sel_meta.get("spill_start_skipped"):
        print(f"  {sel_meta['spill_start_skipped']}")
    pd = read_pointing_files(pointing_files)

    # Trim to the requested time window using the actual sample timestamps.
    dmjd = np.asarray(pd.dmjd, dtype=float)
    mask = (dmjd >= start_mjd) & (dmjd <= end_mjd)
    n_raw = int(len(dmjd))
    n_keep = int(np.count_nonzero(mask))
    print(f"  ANTPOSGR rows: {n_raw} combined -> {n_keep} with {start_mjd:.6f} <= DMJD <= {end_mjd:.6f}")
    if not np.any(mask):
        raise FileNotFoundError(
            f"Pointing files were found, but no pointing samples fall within {start_utc}..{end_utc}."
        )
    az = np.asarray(pd.az[mask], dtype=float)
    el = np.asarray(pd.el[mask], dtype=float)
    if add_pointing_offset:
        az = az + _POINTING_AZ_OFFSET_DEG
        el = el + _POINTING_EL_OFFSET_DEG

    return PointingData(
        dmjd=pd.dmjd[mask],
        az=az,
        el=el,
        ra=pd.ra[mask],
        dec=pd.dec[mask],
    )




def match_data_and_pointing(
    data: HDF5Data,
    pointing_data: Union[PointingData, list[str]],
) -> HDF5Data:
    """
    Match spec (or calibrated_spec) to pointing by time windows: for each pointing sample,
    use all spec data within 50% of the interval to adjacent pointing samples; compute
    mean and std over those spec samples for all 10 polarizations.

    Spec is at higher time resolution than pointing. For each pointing time, the window
    is [t_i - 0.5*dt_left, t_i + 0.5*dt_right] where dt_left = t_i - t_{i-1} and
    dt_right = t_{i+1} - t_i (at edges, use the single adjacent interval).

    Parameters
    ----------
    data : HDF5Data
        Observation data with .time and .spec and/or .calibrated_spec.
    pointing_data : PointingData or list[str]
        Pointing data, or list of pointing file paths (read with read_pointing_files).
    Returns
    -------
    HDF5Data
        One row per pointing: time (pointing times), ra, dec, el, az, and
        calibrated_spec_mean / calibrated_spec_std (or spec_mean / spec_std if no
        calibrated_spec), each with 10 pols of shape (n_pointing, n_freq).
    """
    if isinstance(pointing_data, list):
        pointing_data = read_pointing_files(pointing_data)
    # Ensure both times are in the same units (MJD in days) for window matching
    data_mjd = _time_to_mjd(np.asarray(data.time))
    pointing_mjd = _time_to_mjd(np.asarray(pointing_data.dmjd, dtype=float))
    n_pointing = len(pointing_mjd)
    n_freq = len(data.freq)

    # Time window per pointing: 50% offset on each side of the interval
    dt_left = np.empty(n_pointing)
    dt_right = np.empty(n_pointing)
    if n_pointing == 1:
        span = np.max(data_mjd) - np.min(data_mjd) if len(data_mjd) > 1 else 1.0 / 86400  # 1 sec in days
        dt_left[0] = dt_right[0] = span
    else:
        dt_left[0] = pointing_mjd[1] - pointing_mjd[0]
        dt_right[-1] = pointing_mjd[-1] - pointing_mjd[-2]
        for i in range(1, n_pointing):
            dt_left[i] = pointing_mjd[i] - pointing_mjd[i - 1]
        for i in range(n_pointing - 1):
            dt_right[i] = pointing_mjd[i + 1] - pointing_mjd[i]
    left = pointing_mjd - 0.5 * dt_left
    right = pointing_mjd + 0.5 * dt_right

    # Sanity check: raise only if no window has any spec data (some windows may have none)
    at_least_one_window_overlaps = any(
        np.any((data_mjd >= left[i]) & (data_mjd <= right[i])) for i in range(n_pointing)
    )
    if not at_least_one_window_overlaps:
        raise ValueError(
            "No pointing window contains any spec times. "
            "Check that data.time and pointing_data.dmjd use the same time convention "
            "(both are converted to MJD days internally). "
            f"data_mjd range: [{data_mjd.min():.4f}, {data_mjd.max():.4f}]; "
            f"pointing window range: [{left.min():.4f}, {right.max():.4f}]"
        )

    # Prefer calibrated_spec; fall back to spec (use .value if Quantity)
    if getattr(data, "calibrated_spec", None) is not None:
        spec_source = data.calibrated_spec
        mean_suffix, std_suffix = "calibrated_spec_mean", "calibrated_spec_std"
    else:
        if data.spec is None:
            raise ValueError("data must have spec or calibrated_spec for match_data_and_pointing")
        spec_source = data.spec
        mean_suffix, std_suffix = "spec_mean", "spec_std"

    def _to_array(v):
        return getattr(v, "value", v) if getattr(v, "unit", None) is not None else np.asarray(v)

    mean_arrays = {}
    std_arrays = {}
    for name in CAL_POL_NAMES:
        arr = getattr(spec_source, name, None)
        if arr is None:
            mean_arrays[name] = np.full((n_pointing, n_freq), np.nan, dtype=float)
            std_arrays[name] = np.full((n_pointing, n_freq), np.nan, dtype=float)
            continue
        arr = _to_array(arr)
        if arr.ndim != 2 or arr.shape[1] != n_freq:
            raise ValueError(f"spec {name} shape {arr.shape} inconsistent with n_freq={n_freq}")
        means = np.full((n_pointing, n_freq), np.nan, dtype=float)
        stds = np.full((n_pointing, n_freq), np.nan, dtype=float)
        for i in range(n_pointing):
            mask = (data_mjd >= left[i]) & (data_mjd <= right[i])
            if np.any(mask):
                means[i, :] = np.nanmean(arr[mask, :], axis=0)
                stds[i, :] = np.nanstd(arr[mask, :], axis=0)
        mean_arrays[name] = means
        std_arrays[name] = stds

    mean_spec = CalibratedSpec()
    std_spec = CalibratedSpec()
    for name in CAL_POL_NAMES:
        setattr(mean_spec, name, mean_arrays[name])
        setattr(std_spec, name, std_arrays[name])

    # Pointing time: use datetime if data.time was datetime, else keep MJD
    if len(data.time) > 0 and isinstance(np.asarray(data.time).flat[0], (datetime, np.datetime64)):
        pointing_time = np.array([Time(t, format="mjd").datetime for t in pointing_mjd])
    else:
        pointing_time = pointing_mjd

    # Build return explicitly at pointing resolution (do not copy ra/dec/etc from data)
    return HDF5Data(
        freq=data.freq,
        time=pointing_time,
        spec=None,
        ra=np.asarray(pointing_data.ra),
        dec=np.asarray(pointing_data.dec),
        el=np.asarray(pointing_data.el),
        az=np.asarray(pointing_data.az),
        **{mean_suffix: mean_spec, std_suffix: std_spec},
    )


_MAX_SPECTRA_TIMING_OFFSET_S = 0.5  # hard bound on mean el/az/combined clock lag


def _spectrum_brightness_1d(data: HDF5Data, attribute: str | None = None) -> np.ndarray:
    """Frequency-averaged brightness time series from calibrated_spec or spec."""
    if getattr(data, "calibrated_spec", None) is not None:
        source = data.calibrated_spec
    elif getattr(data, "spec", None) is not None:
        source = data.spec
    else:
        raise ValueError("data must have calibrated_spec or spec to find spectrum peaks")

    def _to_array(v):
        return getattr(v, "value", v) if getattr(v, "unit", None) is not None else np.asarray(v)

    if attribute is not None:
        arr = getattr(source, attribute, None)
        if arr is None:
            raise ValueError(f"attribute {attribute!r} not available on spectrum data")
        arr = np.asarray(_to_array(arr), dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"spectrum {attribute} must be 2D (n_time, n_freq), got {arr.shape}")
        return np.nanmean(arr, axis=1)

    per_pol = []
    for name in CAL_POL_NAMES:
        arr = getattr(source, name, None)
        if arr is None:
            continue
        arr = np.asarray(_to_array(arr), dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"spectrum {name} must be 2D (n_time, n_freq), got {arr.shape}")
        per_pol.append(np.nanmean(arr, axis=1))
    if not per_pol:
        raise ValueError("No polarization channels available to form a brightness time series")
    return np.nanmean(np.vstack(per_pol), axis=0)


def _unwrap_azimuth_deg(az: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    """Unwrap azimuth (deg) in time order; NaNs preserved."""
    unwrapped = np.full_like(az, np.nan, dtype=float)
    valid = np.isfinite(az) & np.isfinite(seconds)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return unwrapped
    ordered = valid_indices[np.argsort(seconds[valid])]
    unwrapped[ordered] = np.rad2deg(np.unwrap(np.deg2rad(az[ordered])))
    return unwrapped


def estimate_timing_offset_from_source_peaks(
    data: HDF5Data,
    pointing_data: Union[PointingData, list[str]],
    source_name: str,
    *,
    n_scans: int = 4,
    el_scans: Sequence[int] | None = None,
    az_scans: Sequence[int] | None = None,
    attribute: str | None = None,
    max_offset_seconds: float = _MAX_SPECTRA_TIMING_OFFSET_S,
    az_offset_deg: float = 0.0,
    el_offset_deg: float = 0.0,
) -> dict[str, float | list[float] | list[int]]:
    """
    Estimate spectra clock lag from source transit times on az and el scans.

    Uses the same cross-scan leg splitting as :func:`get_pointing_offset`
    (default: el legs 0–1, az legs 2–3). Legs prefer pointing recording gaps
    via :func:`resolve_scan_legs` (falls back to equal splits when the number
    of detected blocks is not ``n_scans``):

    - On ``pointing_data``, find the time each scan leg is *closest* to the
      catalog source plus optional spatial pointing offsets
      (``az_offset_deg`` / ``el_offset_deg``, same sign as
      :func:`get_pointing_offset`: measured minus expected).
    - On ``data``, find the time of the brightness *peak* in the same absolute
      time window as that pointing leg.
    - Per-leg timing offset is ``t_pointing_closest - t_spectrum_peak`` (seconds).
      A positive value means the spectra clock is behind the pointing clock.

    Returns the mean over el legs, mean over az legs, and overall mean, plus
    diagnostic indices/times (including ``scan_split`` and gap diagnostics).
    """
    if isinstance(pointing_data, list):
        pointing_data = read_pointing_files(pointing_data)

    from skymap.Beam import get_source_radec

    src_ra, src_dec = get_source_radec(source_name)

    pointing_mjd = _time_to_mjd(np.asarray(pointing_data.dmjd, dtype=float))
    pointing_az = np.asarray(pointing_data.az, dtype=float)
    pointing_el = np.asarray(pointing_data.el, dtype=float)
    if pointing_mjd.size < n_scans:
        raise ValueError(f"Need at least {n_scans} pointing samples, got {pointing_mjd.size}")
    if pointing_az.shape != pointing_mjd.shape or pointing_el.shape != pointing_mjd.shape:
        raise ValueError("pointing_data.dmjd, az, and el must have the same length")

    order = np.argsort(pointing_mjd)
    pointing_mjd = pointing_mjd[order]
    pointing_az = pointing_az[order]
    pointing_el = pointing_el[order]

    data_mjd = _time_to_mjd(np.asarray(data.time))
    brightness = _spectrum_brightness_1d(data, attribute=attribute)
    if brightness.shape[0] != data_mjd.shape[0]:
        raise ValueError("brightness time series length must match data.time")

    src_az, src_el = expected_source_altaz_deg(src_ra, src_dec, pointing_mjd)
    # Shift catalog AltAz by spatial pointing offset so "closest" tracks the
    # true source direction (brightness peak), not the catalog alone.
    src_az = np.asarray(src_az, dtype=float) + float(az_offset_deg)
    src_el = np.asarray(src_el, dtype=float) + float(el_offset_deg)

    if el_scans is None:
        el_scans = tuple(range(min(2, n_scans)))
    if az_scans is None:
        az_scans = tuple(range(2, n_scans))
    el_leg_indices = _validate_scan_leg_indices(el_scans, n_scans=n_scans, name="el_scans")
    az_leg_indices = _validate_scan_leg_indices(az_scans, n_scans=n_scans, name="az_scans")

    pointing_legs, scan_split, gap_diag = resolve_scan_legs(pointing_mjd, n_scans=n_scans)

    def _leg_time_window(leg_idx: np.ndarray) -> tuple[float, float]:
        return float(pointing_mjd[leg_idx[0]]), float(pointing_mjd[leg_idx[-1]])

    def _spectrum_peak_in_window(t0: float, t1: float) -> tuple[int, float]:
        mask = (data_mjd >= t0) & (data_mjd <= t1) & np.isfinite(brightness)
        if not np.any(mask):
            raise ValueError(f"No spectrum samples in pointing window [{t0}, {t1}]")
        idx_local = int(np.nanargmax(brightness[mask]))
        i_peak = int(np.flatnonzero(mask)[idx_local])
        return i_peak, float(data_mjd[i_peak])

    def _offsets_for_legs(
        leg_indices: Sequence[int],
        *,
        axis: str,
    ) -> tuple[list[float], list[int], list[int], list[float], list[float]]:
        offsets: list[float] = []
        pointing_peak_indices: list[int] = []
        spectrum_peak_indices: list[int] = []
        pointing_times: list[float] = []
        spectrum_times: list[float] = []
        for leg_i in leg_indices:
            leg = pointing_legs[leg_i]
            if axis == "el":
                sep = np.abs(pointing_el[leg] - src_el[leg])
            else:
                sep = np.array(
                    [_azimuth_offset_deg(pointing_az[j], src_az[j]) for j in leg],
                    dtype=float,
                )
                sep = np.abs(sep)
            if not np.any(np.isfinite(sep)):
                raise ValueError(f"No finite pointing samples on {axis} scan leg {leg_i}")
            i_closest = int(leg[int(np.nanargmin(sep))])
            t_point = float(pointing_mjd[i_closest])
            t0, t1 = _leg_time_window(leg)
            i_spec, t_spec = _spectrum_peak_in_window(t0, t1)
            # Positive => spectra clock behind pointing clock.
            offsets.append((t_point - t_spec) * 86400.0)
            pointing_peak_indices.append(i_closest)
            spectrum_peak_indices.append(i_spec)
            pointing_times.append(t_point)
            spectrum_times.append(t_spec)
        return offsets, pointing_peak_indices, spectrum_peak_indices, pointing_times, spectrum_times

    el_offsets, el_point_idx, el_spec_idx, el_point_t, el_spec_t = _offsets_for_legs(
        el_leg_indices, axis="el"
    )
    az_offsets, az_point_idx, az_spec_idx, az_point_t, az_spec_t = _offsets_for_legs(
        az_leg_indices, axis="az"
    )

    el_timing_offset = float(np.mean(el_offsets)) if el_offsets else float("nan")
    az_timing_offset = float(np.mean(az_offsets)) if az_offsets else float("nan")
    all_means = [v for v in (el_timing_offset, az_timing_offset) if np.isfinite(v)]
    timing_offset = float(np.mean(all_means)) if all_means else float("nan")

    # Per-leg values can be large (spatial pointing offsets reverse with scan
    # direction); the clock lag is the mean over opposite legs. Enforce the
    # max_offset bound on those means only.
    max_offset = float(max_offset_seconds)
    for label, value in (
        ("el_timing_offset_seconds", el_timing_offset),
        ("az_timing_offset_seconds", az_timing_offset),
        ("timing_offset_seconds", timing_offset),
    ):
        if np.isfinite(value) and abs(value) > max_offset + 1e-12:
            raise ValueError(
                f"Estimated {label}={value} exceeds max_offset_seconds={max_offset} "
                f"(el_offsets={el_offsets}, az_offsets={az_offsets})"
            )

    return {
        "src_ra": float(src_ra),
        "src_dec": float(src_dec),
        "timing_offset_seconds": timing_offset,
        "el_timing_offset_seconds": el_timing_offset,
        "az_timing_offset_seconds": az_timing_offset,
        "el_timing_offsets": el_offsets,
        "az_timing_offsets": az_offsets,
        "el_pointing_closest_indices": el_point_idx,
        "az_pointing_closest_indices": az_point_idx,
        "el_spectrum_peak_indices": el_spec_idx,
        "az_spectrum_peak_indices": az_spec_idx,
        "el_pointing_closest_mjd": el_point_t,
        "az_pointing_closest_mjd": az_point_t,
        "el_spectrum_peak_mjd": el_spec_t,
        "az_spectrum_peak_mjd": az_spec_t,
        "el_scans": el_leg_indices,
        "az_scans": az_leg_indices,
        "n_scans": int(n_scans),
        "scan_split": scan_split,
        "segment_start_mjds": gap_diag["segment_start_mjds"],
        "segment_end_mjds": gap_diag["segment_end_mjds"],
        "gap_seconds": gap_diag["gap_seconds"],
        "az_offset_deg": float(az_offset_deg),
        "el_offset_deg": float(el_offset_deg),
    }


def estimate_timing_offset_after_spatial_correction(
    data: HDF5Data,
    pointing_data: Union[PointingData, list[str]],
    source_name: str,
    *,
    n_scans: int = 4,
    el_scans: Sequence[int] | None = None,
    az_scans: Sequence[int] | None = None,
    attribute: str | None = None,
    max_offset_seconds: float = _MAX_SPECTRA_TIMING_OFFSET_S,
) -> dict[str, float | list[float] | list[int] | dict]:
    """
    Estimate clock lag after removing spatial pointing offsets.

    Pipeline:
    1. Window-match spectra to pointing (:func:`match_data_and_pointing`).
    2. Measure signed az/el pointing offsets (:func:`get_pointing_offset`).
    3. Re-estimate timing with expected AltAz shifted by the signed means of
       those per-leg offsets (:func:`estimate_timing_offset_from_source_peaks`).

    Returns the timing-info dict plus ``az_offset_deg``, ``el_offset_deg``, and
    ``spatial_offset_info`` from step 2.
    """
    if isinstance(pointing_data, list):
        pointing_data = read_pointing_files(pointing_data)

    if el_scans is None:
        el_scans = tuple(range(min(2, n_scans)))
    if az_scans is None:
        az_scans = tuple(range(2, n_scans))

    matched = match_data_and_pointing(data, pointing_data)
    spatial = get_pointing_offset(
        matched,
        source_name,
        n_scans=n_scans,
        el_scans=el_scans,
        az_scans=az_scans,
    )
    az_offset_deg = float(np.mean(np.asarray(spatial["az_offsets"], dtype=float)))
    el_offset_deg = float(np.mean(np.asarray(spatial["el_offsets"], dtype=float)))

    timing_info = estimate_timing_offset_from_source_peaks(
        data,
        pointing_data,
        source_name,
        n_scans=n_scans,
        el_scans=el_scans,
        az_scans=az_scans,
        attribute=attribute,
        max_offset_seconds=max_offset_seconds,
        az_offset_deg=az_offset_deg,
        el_offset_deg=el_offset_deg,
    )
    timing_info["spatial_offset_info"] = spatial
    return timing_info


def match_data_and_pointing_with_timing_offset(
    data: HDF5Data,
    pointing_data: Union[PointingData, list[str]],
    source_name: str | None = None,
    *,
    timing_offset_seconds: float | None = None,
    max_offset_seconds: float = _MAX_SPECTRA_TIMING_OFFSET_S,
    n_scans: int = 4,
    el_scans: Sequence[int] | None = None,
    az_scans: Sequence[int] | None = None,
    attribute: str | None = None,
    apply_spatial_offset_correction: bool = True,
) -> HDF5Data:
    """
    Synchronize spectra times to pointing, then evaluate true pointing per spectrum.

    When ``timing_offset_seconds`` is not given (default), the clock lag is
    estimated as:

    1. Window-match spectra to pointing and measure spatial az/el offsets
       (:func:`estimate_timing_offset_after_spatial_correction`), unless
       ``apply_spatial_offset_correction=False``.
    2. Find per-leg pointing closest-approach vs spectrum brightness peaks
       (with expected AltAz shifted by those spatial offsets when step 1 ran).
    3. ``timing_offset = mean(t_pointing_closest - t_spectrum_peak)`` over el and
       az legs. Must satisfy ``|offset| <= max_offset_seconds`` (default 0.5 s).

    True Az/El/RA/Dec at each spectrum are then taken from piecewise-linear
    interpolation of ``pointing_data`` at ``data.time + timing_offset``.

    A positive offset means the spectra clock is behind the pointing clock.
    """
    if isinstance(pointing_data, list):
        pointing_data = read_pointing_files(pointing_data)

    data_time = np.asarray(data.time)
    data_mjd = _time_to_mjd(data_time)
    pointing_mjd = _time_to_mjd(np.asarray(pointing_data.dmjd, dtype=float))
    if data_mjd.size == 0:
        raise ValueError("data.time must not be empty")
    if pointing_mjd.size < 2:
        raise ValueError("At least two pointing samples are required")

    max_offset = float(max_offset_seconds)
    if max_offset <= 0.0:
        raise ValueError("max_offset_seconds must be positive")

    pointing_az = np.asarray(pointing_data.az, dtype=float)
    pointing_el = np.asarray(pointing_data.el, dtype=float)
    pointing_ra = np.asarray(pointing_data.ra, dtype=float)
    pointing_dec = np.asarray(pointing_data.dec, dtype=float)
    if (
        pointing_mjd.ndim != 1
        or pointing_az.shape != pointing_mjd.shape
        or pointing_el.shape != pointing_mjd.shape
    ):
        raise ValueError(
            "pointing_data.dmjd, pointing_data.az, and pointing_data.el must have "
            "the same one-dimensional shape"
        )

    timing_info: dict[str, float | list[float] | list[int] | dict] | None = None
    if timing_offset_seconds is not None:
        timing_offset = float(timing_offset_seconds)
        if abs(timing_offset) > max_offset + 1e-12:
            raise ValueError(
                f"timing_offset_seconds={timing_offset} exceeds max_offset_seconds={max_offset}"
            )
    else:
        if source_name is None:
            raise ValueError(
                "source_name is required to estimate the timing offset from source "
                "peaks, or pass timing_offset_seconds explicitly"
            )
        if apply_spatial_offset_correction:
            timing_info = estimate_timing_offset_after_spatial_correction(
                data,
                pointing_data,
                source_name,
                n_scans=n_scans,
                el_scans=el_scans,
                az_scans=az_scans,
                attribute=attribute,
                max_offset_seconds=max_offset,
            )
        else:
            timing_info = estimate_timing_offset_from_source_peaks(
                data,
                pointing_data,
                source_name,
                n_scans=n_scans,
                el_scans=el_scans,
                az_scans=az_scans,
                attribute=attribute,
                max_offset_seconds=max_offset,
            )
        timing_offset = float(timing_info["timing_offset_seconds"])

    reference_mjd = float(np.nanmedian(pointing_mjd))
    pointing_seconds = (pointing_mjd - reference_mjd) * 86400.0
    data_seconds = (data_mjd - reference_mjd) * 86400.0

    pointing_az_u = _unwrap_azimuth_deg(pointing_az, pointing_seconds)
    valid_pointing = (
        np.isfinite(pointing_seconds)
        & np.isfinite(pointing_az_u)
        & np.isfinite(pointing_el)
    )
    if np.count_nonzero(valid_pointing) < 2:
        raise ValueError("At least two finite pointing samples are required")

    order = np.argsort(pointing_seconds[valid_pointing])
    p_t = pointing_seconds[valid_pointing][order]
    p_az = pointing_az_u[valid_pointing][order]
    p_el = pointing_el[valid_pointing][order]
    p_ra = pointing_ra[valid_pointing][order] if pointing_ra.shape == pointing_mjd.shape else None
    p_dec = pointing_dec[valid_pointing][order] if pointing_dec.shape == pointing_mjd.shape else None

    corrected_mjd = data_mjd + timing_offset / 86400.0
    corrected_seconds = data_seconds + timing_offset
    true_az = np.mod(np.interp(corrected_seconds, p_t, p_az, left=np.nan, right=np.nan), 360.0)
    true_el = np.interp(corrected_seconds, p_t, p_el, left=np.nan, right=np.nan)

    # Do not invent mount positions across pointing recording gaps.
    pointing_mjd_sorted = pointing_mjd[valid_pointing][order]
    gap_legs = split_pointing_scans_by_gaps(pointing_mjd_sorted, gap_threshold_seconds=1.0)
    in_recording = np.zeros(corrected_mjd.shape, dtype=bool)
    for leg in gap_legs:
        t_lo = float(pointing_mjd_sorted[leg[0]])
        t_hi = float(pointing_mjd_sorted[leg[-1]])
        in_recording |= (corrected_mjd >= t_lo) & (corrected_mjd <= t_hi)
    true_az = np.where(in_recording, true_az, np.nan)
    true_el = np.where(in_recording, true_el, np.nan)

    if isinstance(data_time.flat[0], (datetime, np.datetime64)):
        corrected_time = np.array(
            [Time(t, format="mjd", scale="utc").datetime for t in corrected_mjd]
        )
    else:
        corrected_time = corrected_mjd

    if (
        p_ra is not None
        and p_dec is not None
        and np.any(np.isfinite(p_ra))
        and np.any(np.isfinite(p_dec))
    ):
        true_ra = np.interp(corrected_seconds, p_t, p_ra, left=np.nan, right=np.nan)
        true_dec = np.interp(corrected_seconds, p_t, p_dec, left=np.nan, right=np.nan)
        true_ra = np.where(in_recording, true_ra, np.nan)
        true_dec = np.where(in_recording, true_dec, np.nan)
    else:
        true_ra, true_dec = az_el_to_radec_deg(true_az, true_el, corrected_time)

    kwargs: dict[str, Any] = {
        "ra": true_ra,
        "dec": true_dec,
        "az": true_az,
        "el": true_el,
        "original_time": data_time.copy(),
        "timing_offset_seconds": float(timing_offset),
    }
    if timing_info is not None:
        kwargs["el_timing_offset_seconds"] = float(timing_info["el_timing_offset_seconds"])
        kwargs["az_timing_offset_seconds"] = float(timing_info["az_timing_offset_seconds"])
        kwargs["az_offset_deg"] = float(timing_info.get("az_offset_deg", 0.0))
        kwargs["el_offset_deg"] = float(timing_info.get("el_offset_deg", 0.0))
        kwargs["timing_offset_info"] = timing_info
    calibrated_spec = getattr(data, "calibrated_spec", None)
    if calibrated_spec is not None:
        kwargs["calibrated_spec"] = calibrated_spec
        # Same arrays at spectrum cadence so get_pointing_offset / plots work.
        kwargs["calibrated_spec_mean"] = calibrated_spec

    return HDF5Data(
        freq=data.freq,
        time=corrected_time,
        spec=data.spec,
        **kwargs,
    )


def _azimuth_offset_deg(measured_az_deg: float, reference_az_deg: float) -> float:
    """Shortest signed difference in degrees (−180, 180]."""
    d = float(measured_az_deg) - float(reference_az_deg)
    return (d + 180.0) % 360.0 - 180.0


def _ra_offset_deg(measured_ra_deg: float, reference_ra_deg: float) -> float:
    """Shortest signed RA difference in degrees (−180, 180]."""
    d = float(measured_ra_deg) - float(reference_ra_deg)
    return (d + 180.0) % 360.0 - 180.0


# GBT site coordinates (NAD83 lat/lon; track elevation NAVD88).
# https://greenbankobservatory.org/portal/gbt/instruments/
_GBT_LAT = Angle("38d25m59.236s")
_GBT_LON = Angle("-79d50m23.406s")
_GBT_HEIGHT_M = 807.43

GBT_OBSERVER_LOCATION = EarthLocation.from_geodetic(
    lon=_GBT_LON, lat=_GBT_LAT, height=_GBT_HEIGHT_M * u.m
)


def gbt_observer_location() -> EarthLocation:
    """EarthLocation for the Green Bank Telescope."""
    return GBT_OBSERVER_LOCATION


def _obstime_from_times(time_arr: np.ndarray) -> Time:
    mjd = _time_to_mjd(np.asarray(time_arr))
    return Time(mjd, format="mjd", scale="utc")


def local_sidereal_time_deg(time_arr: np.ndarray) -> np.ndarray:
    """
    Local mean sidereal time (deg) at each sample time at the GBT.

    Site coordinates from the
    `GBT instruments page <https://greenbankobservatory.org/portal/gbt/instruments/>`_.
    """
    obstime = _obstime_from_times(time_arr)
    lst = obstime.sidereal_time("mean", GBT_OBSERVER_LOCATION)
    return np.asarray(lst.to_value(u.deg), dtype=float)


def az_el_to_radec_deg(
    az_deg: np.ndarray | float,
    el_deg: np.ndarray | float,
    time_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform topocentric Az/El (deg) to ICRS RA/Dec (deg) at each sample time.

    ``az_deg``, ``el_deg``, and ``time_arr`` must broadcast to the same shape.
    Uses the GBT horizon frame; site coordinates from the
    `GBT instruments page <https://greenbankobservatory.org/portal/gbt/instruments/>`_.
    """
    az = np.atleast_1d(np.asarray(az_deg, dtype=float))
    el = np.atleast_1d(np.asarray(el_deg, dtype=float))
    time_arr = np.asarray(time_arr)
    if az.size != el.size or az.size != time_arr.size:
        raise ValueError(
            f"az_deg, el_deg, and time_arr must have the same length, "
            f"got {az.size}, {el.size}, and {time_arr.size}"
        )
    frame = AltAz(obstime=_obstime_from_times(time_arr), location=GBT_OBSERVER_LOCATION)
    sc = SkyCoord(az=az * u.deg, alt=el * u.deg, frame=frame)
    icrs = sc.transform_to("icrs")
    return (
        np.asarray(icrs.ra.to_value(u.deg), dtype=float),
        np.asarray(icrs.dec.to_value(u.deg), dtype=float),
    )


def radec_to_az_el_deg(
    ra_deg: np.ndarray | float,
    dec_deg: np.ndarray | float,
    time_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform ICRS RA/Dec (deg) to topocentric Az/El (deg) at each sample time.

    Scalar RA/Dec broadcast over ``time_arr``. Uses the GBT horizon frame; site
    coordinates from the
    `GBT instruments page <https://greenbankobservatory.org/portal/gbt/instruments/>`_.
    """
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    time_arr = np.asarray(time_arr)
    if ra.size != dec.size and not (ra.size == 1 or dec.size == 1):
        raise ValueError(
            f"ra_deg and dec_deg must match in length or be scalar, got {ra.size} and {dec.size}"
        )
    if ra.size not in (1, time_arr.size) or dec.size not in (1, time_arr.size):
        if ra.size != time_arr.size or dec.size != time_arr.size:
            raise ValueError(
                f"ra_deg/dec_deg must be scalar or length {time_arr.size}, "
                f"got {ra.size} and {dec.size}"
            )
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    altaz = sc.transform_to(AltAz(obstime=_obstime_from_times(time_arr), location=GBT_OBSERVER_LOCATION))
    return (
        np.asarray(altaz.az.to_value(u.deg), dtype=float),
        np.asarray(altaz.alt.to_value(u.deg), dtype=float),
    )


def split_time_ordered_scans(n_samples: int, n_scans: int = 4) -> list[np.ndarray]:
    """Index arrays for ``n_scans`` equal, time-ordered scan legs."""
    if n_scans < 1:
        raise ValueError("n_scans must be >= 1")
    if n_samples < n_scans:
        raise ValueError(f"Need at least {n_scans} samples for {n_scans} scans, got {n_samples}")
    return list(np.array_split(np.arange(n_samples), n_scans))


def split_pointing_scans_by_gaps(
    pointing_mjd: np.ndarray,
    *,
    gap_threshold_seconds: float = 1.0,
) -> list[np.ndarray]:
    """
    Split a time-sorted pointing timeline into contiguous recording blocks.

    Breaks wherever consecutive MJD samples are separated by more than
    ``gap_threshold_seconds`` (pointing pauses between scan legs). Returns a
    list of index arrays into ``pointing_mjd``.
    """
    mjd = np.asarray(pointing_mjd, dtype=float)
    if mjd.ndim != 1:
        raise ValueError(f"pointing_mjd must be 1-D, got shape {mjd.shape}")
    if mjd.size == 0:
        return []
    if mjd.size == 1:
        return [np.array([0], dtype=int)]

    dt_s = np.diff(mjd) * 86400.0
    # Gaps after sample i (break between i and i+1).
    break_after = np.flatnonzero(dt_s > float(gap_threshold_seconds))
    starts = np.concatenate(([0], break_after + 1))
    ends = np.concatenate((break_after + 1, [mjd.size]))
    return [np.arange(int(s), int(e), dtype=int) for s, e in zip(starts, ends)]


def split_scans_by_finite_mask(valid: np.ndarray) -> list[np.ndarray]:
    """Index arrays for contiguous runs where ``valid`` is True."""
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"valid must be 1-D, got shape {mask.shape}")
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [np.arange(int(s), int(e), dtype=int) for s, e in zip(starts, ends)]


def _scan_leg_gap_diagnostics(
    pointing_mjd: np.ndarray,
    legs: list[np.ndarray],
) -> dict[str, list[float]]:
    """Segment start/end MJDs and inter-segment gap durations (seconds)."""
    mjd = np.asarray(pointing_mjd, dtype=float)
    starts = [float(mjd[leg[0]]) for leg in legs if leg.size]
    ends = [float(mjd[leg[-1]]) for leg in legs if leg.size]
    gaps: list[float] = []
    for i in range(len(starts) - 1):
        gaps.append((starts[i + 1] - ends[i]) * 86400.0)
    return {
        "segment_start_mjds": starts,
        "segment_end_mjds": ends,
        "gap_seconds": gaps,
    }


def resolve_scan_legs(
    pointing_mjd: np.ndarray,
    *,
    n_scans: int = 4,
    gap_threshold_seconds: float = 1.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], str, dict[str, list[float]]]:
    """
    Prefer gap-based scan legs; fall back to equal splits if needed.

    Returns ``(legs, scan_split, diagnostics)`` where ``scan_split`` is
    ``\"gaps\"`` when the number of detected recording blocks equals
    ``n_scans``, ``\"valid\"`` when contiguous finite-coordinate runs match
    ``n_scans`` (e.g. timing-corrected spectra with NaNs in pointing gaps),
    otherwise ``\"equal\"``.
    """
    mjd = np.asarray(pointing_mjd, dtype=float)
    gap_legs = split_pointing_scans_by_gaps(
        mjd, gap_threshold_seconds=gap_threshold_seconds
    )
    diagnostics = _scan_leg_gap_diagnostics(mjd, gap_legs)
    if len(gap_legs) == int(n_scans):
        return gap_legs, "gaps", diagnostics
    if valid_mask is not None:
        valid_legs = split_scans_by_finite_mask(valid_mask)
        if len(valid_legs) == int(n_scans):
            return valid_legs, "valid", _scan_leg_gap_diagnostics(mjd, valid_legs)
    equal_legs = split_time_ordered_scans(mjd.size, n_scans=n_scans)
    return equal_legs, "equal", diagnostics


def radec_corrected_for_pointing_offset(
    az_deg: np.ndarray | float,
    el_deg: np.ndarray | float,
    time_arr: np.ndarray,
    *,
    az_offset_deg: float,
    el_offset_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ICRS RA/Dec (deg) after removing constant az/el pointing offsets.

    Offsets use the same sign as :func:`get_pointing_offset` (measured minus
    expected). Corrected mount coordinates are ``az - az_offset`` and
    ``el - el_offset``, then transformed to the sky at each ``time_arr`` sample
    at the GBT.
    """
    az_corr = np.asarray(az_deg, dtype=float) - float(az_offset_deg)
    el_corr = np.asarray(el_deg, dtype=float) - float(el_offset_deg)
    return az_el_to_radec_deg(az_corr, el_corr, time_arr)


def expected_source_altaz_deg(
    src_ra: float,
    src_dec: float,
    time_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform catalog RA/Dec (deg) to topocentric Az/El (deg) at each sample time.

    Uses the same GBT horizon frame as :func:`get_pointing_offset`.
    """
    return radec_to_az_el_deg(src_ra, src_dec, time_arr)


def _validate_scan_leg_indices(leg_indices: Sequence[int], *, n_scans: int, name: str) -> tuple[int, ...]:
    if not leg_indices:
        raise ValueError(f"{name} must not be empty")
    out = tuple(int(i) for i in leg_indices)
    for i in out:
        if i < 0 or i >= n_scans:
            raise ValueError(f"{name} leg index {i} out of range for n_scans={n_scans}")
    return out


def _offsets_from_scan_legs(
    val: np.ndarray,
    measured: np.ndarray,
    expected: np.ndarray,
    scan_legs: list[np.ndarray],
    leg_indices: Sequence[int],
    *,
    wrap_azimuth: bool = False,
) -> tuple[list[int], list[float]]:
    """Peak per scan leg; offset = measured - expected at each peak index."""
    peak_indices: list[int] = []
    offsets: list[float] = []
    for leg_i in leg_indices:
        idx = scan_legs[leg_i]
        if idx.size == 0:
            raise ValueError(f"scan leg {leg_i} is empty")
        i_peak = int(idx[np.nanargmax(val[idx])])
        peak_indices.append(i_peak)
        if wrap_azimuth:
            offsets.append(_azimuth_offset_deg(measured[i_peak], expected[i_peak]))
        else:
            offsets.append(float(measured[i_peak] - expected[i_peak]))
    return peak_indices, offsets


def get_pointing_offset(
    data_matched: HDF5Data,
    source_name: str,
    *,
    n_scans: int = 4,
    el_scans: Sequence[int] | None = None,
    az_scans: Sequence[int] | None = None,
) -> dict[str, float | list[float] | list[int]]:
    """
    Compute pointing offsets in **azimuth and elevation** (deg) for a cross (X) pattern.

    Expects the *matched* output from :func:`match_data_and_pointing`, i.e. an object
    with:
    - ``ra``, ``dec``, ``az``, and ``el`` arrays (deg), one per pointing sample
    - ``time`` aligned with those samples
    - ``calibrated_spec_mean`` as a CalibratedSpec-like container where each
      polarization is shape (n_pointing, n_freq)

    The catalog position (RA/Dec) is transformed to Alt/Az at each sample time using
    :func:`radec_to_az_el_deg` at the GBT. Samples are split into ``n_scans`` legs
    via :func:`resolve_scan_legs` (prefer pointing recording gaps; fall back to
    equal time-ordered splits). For each leg listed in
    ``el_scans`` / ``az_scans``, the maximum-response sample **within that leg** is
    found and compared to the expected source Az/El at that sample time (signed;
    azimuth wrapped to (−180°, 180°]). Per-leg signed values are in
    ``el_offsets`` / ``az_offsets``; returned ``el_offset`` / ``az_offset`` are the
    mean of the absolute per-leg values. Sky-frame offsets ``dec_offsets`` /
    ``ra_offsets`` (per el/az leg peak vs catalog) are also returned, with
    ``dec_offset`` / ``ra_offset`` as the mean of their absolute values.

    Defaults: ``el_scans=(0, 1)`` (scans 1–2), ``az_scans=(2, 3)`` (scans 3–4).

    For fewer than ``n_scans`` samples, both offsets use the single global maximum
    sample.
    """
    if getattr(data_matched, "ra", None) is None or getattr(data_matched, "dec", None) is None:
        raise ValueError("data_matched must have ra and dec (output from match_data_and_pointing)")
    if getattr(data_matched, "az", None) is None or getattr(data_matched, "el", None) is None:
        raise ValueError(
            "data_matched must have az and el to compute az/el offsets (output from match_data_and_pointing)"
        )
    mean_spec = getattr(data_matched, "calibrated_spec_mean", None)
    if mean_spec is None:
        raise ValueError(
            "data_matched must have calibrated_spec_mean to compute pointing offset (run calibration + match_data_and_pointing)"
        )

    # Import locally to avoid pulling plotting/healpy deps on module import.
    from skymap.Beam import get_source_radec

    src_ra, src_dec = get_source_radec(source_name)

    per_pol = []
    for name in CAL_POL_NAMES:
        arr = getattr(mean_spec, name, None)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"calibrated_spec_mean.{name} must be 2D (n_pointing, n_freq), got {arr.shape}")
        per_pol.append(np.nanmean(arr, axis=1))

    if not per_pol:
        raise ValueError("calibrated_spec_mean has no available polarization channels to average")

    val = np.nanmean(np.vstack(per_pol), axis=0)  # (n_pointing,)
    if val.size == 0:
        raise ValueError("No pointing samples in data_matched")
    if not np.any(np.isfinite(val)):
        raise ValueError("Frequency-averaged calibrated_spec_mean is all-NaN; cannot determine peak pointing")

    ra = np.asarray(data_matched.ra, dtype=float)
    dec = np.asarray(data_matched.dec, dtype=float)
    az = np.asarray(data_matched.az, dtype=float)
    el = np.asarray(data_matched.el, dtype=float)
    n = val.size

    time_arr = np.asarray(data_matched.time)
    if time_arr.shape[0] != n:
        raise ValueError(
            f"data_matched.time length ({time_arr.shape[0]}) must match number of pointing samples ({n})"
        )
    src_az, src_el = expected_source_altaz_deg(src_ra, src_dec, time_arr)

    if el_scans is None:
        el_scans = tuple(range(min(2, n_scans)))
    if az_scans is None:
        az_scans = tuple(range(2, n_scans))
    el_leg_indices = _validate_scan_leg_indices(el_scans, n_scans=n_scans, name="el_scans")
    az_leg_indices = _validate_scan_leg_indices(az_scans, n_scans=n_scans, name="az_scans")

    if n < n_scans:
        i_peak = int(np.nanargmax(val))
        peak_ra = float(ra[i_peak])
        peak_dec = float(dec[i_peak])
        el_off_signed = float(el[i_peak] - src_el[i_peak])
        az_off_signed = _azimuth_offset_deg(az[i_peak], src_az[i_peak])
        dec_off_signed = float(dec[i_peak] - src_dec)
        ra_off_signed = _ra_offset_deg(ra[i_peak], src_ra)
        return {
            "src_ra": float(src_ra),
            "src_dec": float(src_dec),
            "peak_ra": peak_ra,
            "peak_dec": peak_dec,
            "peak_az": float(az[i_peak]),
            "peak_el": float(el[i_peak]),
            "src_az": float(src_az[i_peak]),
            "src_el": float(src_el[i_peak]),
            "az_offset": float(abs(az_off_signed)),
            "el_offset": float(abs(el_off_signed)),
            "az_offsets": [az_off_signed],
            "el_offsets": [el_off_signed],
            "ra_offset": float(abs(ra_off_signed)),
            "dec_offset": float(abs(dec_off_signed)),
            "ra_offsets": [ra_off_signed],
            "dec_offsets": [dec_off_signed],
            "az_peak_indices": [i_peak],
            "el_peak_indices": [i_peak],
            "el_scans": el_leg_indices,
            "az_scans": az_leg_indices,
        }

    scan_legs, scan_split, gap_diag = resolve_scan_legs(
        _time_to_mjd(time_arr),
        n_scans=n_scans,
        valid_mask=np.isfinite(az) & np.isfinite(el) & np.isfinite(val),
    )
    el_peak_indices, el_offsets = _offsets_from_scan_legs(
        val, el, src_el, scan_legs, el_leg_indices, wrap_azimuth=False
    )
    az_peak_indices, az_offsets = _offsets_from_scan_legs(
        val, az, src_az, scan_legs, az_leg_indices, wrap_azimuth=True
    )
    el_offset = float(np.mean(np.abs(el_offsets)))
    az_offset = float(np.mean(np.abs(az_offsets)))
    dec_offsets = [float(dec[i] - src_dec) for i in el_peak_indices]
    ra_offsets = [_ra_offset_deg(ra[i], src_ra) for i in az_peak_indices]
    dec_offset = float(np.mean(np.abs(dec_offsets)))
    ra_offset = float(np.mean(np.abs(ra_offsets)))
    i_el = el_peak_indices[0]
    i_az = az_peak_indices[0]

    return {
        "src_ra": float(src_ra),
        "src_dec": float(src_dec),
        "peak_ra": float(ra[i_az]),
        "peak_dec": float(dec[i_el]),
        "peak_ra_leg_az": float(ra[i_az]),
        "peak_dec_leg_el": float(dec[i_el]),
        "peak_az": float(az[i_az]),
        "peak_el": float(el[i_el]),
        "src_az": float(src_az[i_az]),
        "src_el": float(src_el[i_el]),
        "az_offset": az_offset,
        "el_offset": el_offset,
        "az_offsets": az_offsets,
        "el_offsets": el_offsets,
        "ra_offset": ra_offset,
        "dec_offset": dec_offset,
        "ra_offsets": ra_offsets,
        "dec_offsets": dec_offsets,
        "az_peak_indices": az_peak_indices,
        "el_peak_indices": el_peak_indices,
        "el_scans": el_leg_indices,
        "az_scans": az_leg_indices,
        "scan_split": scan_split,
        "segment_start_mjds": gap_diag["segment_start_mjds"],
        "segment_end_mjds": gap_diag["segment_end_mjds"],
        "gap_seconds": gap_diag["gap_seconds"],
    }


    '''
use nside of 512

to get the number of pixels in the map, use the following code:

4*pi*360 deg 

Each pixel is 3.8 arcmin in radius 
(77 sq. arcmin)
beam is 1 sq. degree in size, so the radius is 0.56 deg

every 0.56 deg of the beam gets 77 pixels that get assigned the same value. 
Take mean and std of the pixels to get the mean and std of all the values that get assigned to the pixel. 

we also need to take a convolution of the beam and the pixels. 

    '''
