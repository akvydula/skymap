"""
Utility functions for reading HDF5 files and plotting polarization data.

"""

from __future__ import annotations

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time



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
        If True, convert time values from MJD to datetime.
        If False, convert time values from Unix timestamp to datetime. Default is False
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
        # Read frequency data (convert to MHz)
        freq = f['data']['freq'][()] / 1e6
        
        # Read spec data (complex array with shape: n_time, n-spectra, n_freq)
        if time_slice is not None:
            spec_data = f['data']['spec'][time_slice, :, :]
            time_data = f['data']['time'][time_slice]
        else:
            spec_data = f['data']['spec']
            time_data = f['data']['time']

        # Convert time_data to datetime 
        if time_mjd:
            time_data = np.array([Time(ts, format='mjd').datetime for ts in time_data])
        else:
        # time values likely in Unix timestamp format in ms
            time_data = np.array([datetime.fromtimestamp(ts/1000) for ts in time_data])

        # spec_data shape: (n_time, n_spectra, n_freq) -> 2D (n_time, n_freq) per polarization
        spec = SpecData(**{name: spec_data[:, i, :] for i, name in enumerate(CAL_POL_NAMES)})
        
        # Create HDF5Data object with datetime
        data = HDF5Data(freq=freq, time=time_data, spec=spec)
        
        return data
    
    finally:
        if f is not None:
            f.close()

# Same polarization order as SpecData (used for cal gain/te per channel)
CAL_POL_NAMES = ['AA_', 'BB_', 'CC_', 'DD_', 'AB_', 'BC_', 'CD_', 'AC_', 'BD_', 'AD_']


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