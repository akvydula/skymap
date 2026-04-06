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
from astropy.io import fits
import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
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


def get_slice_from_time(data: HDF5Data, time_slice: slice) -> HDF5Data:
    """
    Get a slice of the data from the time slice. Slices time and all
    time-varying attributes (spec, calibrated_spec if present).
    Parameters
    ----------
    data : HDF5Data
        Input data.
    time_slice : slice
        Slice to apply along the time dimension (e.g. slice(0, 1000)).
    Returns
    -------
    HDF5Data
        New HDF5Data with time, spec, and calibrated_spec (if present) sliced.
    """
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
    def _to_utc_naive(dt: datetime) -> datetime:
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

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
    return PointingData(
        dmjd=np.concatenate([p.dmjd for p in parts]),
        az=np.concatenate([p.az for p in parts]),
        el=np.concatenate([p.el for p in parts]),
        ra=np.concatenate([p.ra for p in parts]),
        dec=np.concatenate([p.dec for p in parts]),
    )

def get_pointing_data(datadir: str | Path, start_utc: datetime, end_utc: datetime) -> PointingData:
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

    Returns
    -------
    PointingData
        PointingData object combining all files in the range.
    """
    pointing_files = find_pointing_files(datadir, start_utc, end_utc)
    if len(pointing_files) == 0:
        raise FileNotFoundError(f"No pointing files found in {datadir} between {start_utc} and {end_utc}")
    print(f"Found {len(pointing_files)} pointing files between {start_utc} and {end_utc}")
    return read_pointing_files(pointing_files)




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