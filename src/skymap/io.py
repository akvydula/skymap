"""
Utility functions for reading HDF5 files and plotting polarization data.

"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Union
from zoneinfo import ZoneInfo
from astropy.io import fits
import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
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


def find_pointing_files(
    datadir: str | Path,
    start_utc: datetime,
    end_utc: datetime,
) -> list[str]:
    """
    Find pointing FITS files whose filename UTC time falls within [start_utc, end_utc].

    Expects filenames like 2026_01_16_18:38:48.fits (YYYY_MM_DD_HH:MM:SS.fits).
    Only files matching this pattern and with time in the given range are returned.

    Parameters
    ----------
    datadir : str or Path
        Directory containing pointing .fits files.
    start_utc : datetime
        Start of time range (UTC). Inclusive.
    end_utc : datetime
        End of time range (UTC). Inclusive.

    Returns
    -------
    list[str]
        Sorted list of full paths to matching .fits files.
    """
    datadir = Path(datadir)
    if not datadir.exists():
        raise FileNotFoundError(f"Data directory not found: {datadir}")

    # Normalize to naive UTC for comparison (file times are parsed as naive UTC)
    start_utc = _to_utc_naive(start_utc)
    end_utc = _to_utc_naive(end_utc)
    if start_utc > end_utc:
        raise ValueError(f"start_utc must be <= end_utc, got {start_utc} and {end_utc}")

    pointing_files = []
    for path in datadir.glob("*.fits"):
        file_utc = _parse_pointing_filename_utc(path)
        if file_utc is None:
            continue
        if start_utc <= file_utc <= end_utc:
            pointing_files.append(str(path.resolve()))

    return sorted(pointing_files)  

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


def _azimuth_offset_deg(measured_az_deg: float, reference_az_deg: float) -> float:
    """Shortest signed difference in degrees (−180, 180]."""
    d = float(measured_az_deg) - float(reference_az_deg)
    return (d + 180.0) % 360.0 - 180.0


def get_pointing_offset(
    data_matched: HDF5Data,
    source_name: str,
    *,
    observer_location: EarthLocation | None = None,
) -> dict[str, float]:
    """
    Compute pointing offsets in **azimuth and elevation** (deg) for a cross (X) pattern.

    Expects the *matched* output from :func:`match_data_and_pointing`, i.e. an object
    with:
    - ``ra``, ``dec``, ``az``, and ``el`` arrays (deg), one per pointing sample
    - ``time`` aligned with those samples
    - ``calibrated_spec_mean`` as a CalibratedSpec-like container where each
      polarization is shape (n_pointing, n_freq)

    The catalog position (RA/Dec) is transformed to Alt/Az at each sample time using
    ``observer_location`` (default: Green Bank Telescope). Then:

    - **az_offset**: at the maximum-response sample in the **first** half (same split as
      :func:`skymap.plots.plot_freq_avg_vs_pointing`), measured az minus expected source az.
    - **el_offset**: at the maximum-response sample in the **second** half, measured el minus
      expected source elevation.

    Azimuth difference is wrapped to (−180°, 180°]. For fewer than two samples, both offsets
    use the single global maximum sample.
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
    n_mid = n // 2

    time_arr = np.asarray(data_matched.time)
    if time_arr.shape[0] != n:
        raise ValueError(
            f"data_matched.time length ({time_arr.shape[0]}) must match number of pointing samples ({n})"
        )
    mjd = _time_to_mjd(time_arr)
    obstime = Time(mjd, format="mjd", scale="utc")
    location = observer_location if observer_location is not None else EarthLocation.of_site("Green Bank Telescope")
    sc_src = SkyCoord(ra=float(src_ra) * u.deg, dec=float(src_dec) * u.deg, frame="icrs")
    src_altaz = sc_src.transform_to(AltAz(obstime=obstime, location=location))
    src_az = np.asarray(src_altaz.az.to_value(u.deg), dtype=float)
    src_el = np.asarray(src_altaz.alt.to_value(u.deg), dtype=float)

    if n < 2 or n_mid == 0:
        i_peak = int(np.nanargmax(val))
        peak_ra = float(ra[i_peak])
        peak_dec = float(dec[i_peak])
        return {
            "src_ra": float(src_ra),
            "src_dec": float(src_dec),
            "peak_ra": peak_ra,
            "peak_dec": peak_dec,
            "peak_az": float(az[i_peak]),
            "peak_el": float(el[i_peak]),
            "src_az": float(src_az[i_peak]),
            "src_el": float(src_el[i_peak]),
            "az_offset": _azimuth_offset_deg(az[i_peak], src_az[i_peak]),
            "el_offset": float(el[i_peak] - src_el[i_peak]),
        }

    i1 = int(np.nanargmax(val[:n_mid]))
    i2 = n_mid + int(np.nanargmax(val[n_mid:]))
    peak_ra_leg1 = float(ra[i1])
    peak_dec_leg1 = float(dec[i1])
    peak_ra_leg2 = float(ra[i2])
    peak_dec_leg2 = float(dec[i2])

    return {
        "src_ra": float(src_ra),
        "src_dec": float(src_dec),
        "peak_ra": peak_ra_leg1,
        "peak_dec": peak_dec_leg2,
        "peak_ra_leg2": peak_ra_leg2,
        "peak_dec_leg1": peak_dec_leg1,
        "peak_az": float(az[i1]),
        "peak_el": float(el[i2]),
        "src_az": float(src_az[i1]),
        "src_el": float(src_el[i2]),
        "az_offset": _azimuth_offset_deg(az[i1], src_az[i1]),
        "el_offset": float(el[i2] - src_el[i2]),
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