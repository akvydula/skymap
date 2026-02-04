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

        # Extract polarization data (indices 0, 1, 2, 3 correspond to XX, YY, XY, YX)
        # spec_data shape: (n_time, n_spectra, n_freq)
        # We want 2D arrays: (n_time, n_freq) for each polarization
        AA_ = spec_data[:, 0, :]
        BB_ = spec_data[:, 1, :]
        CC_ = spec_data[:, 2, :]
        DD_ = spec_data[:, 3, :]
        AB_ = spec_data[:, 4, :]
        BC_ = spec_data[:, 5, :]
        CD_ = spec_data[:, 6, :]
        AC_ = spec_data[:, 7, :]
        BD_ = spec_data[:, 8, :]
        AD_ = spec_data[:, 9, :]
        
        # Create SpecData object
        spec = SpecData(AA_=AA_, BB_=BB_, CC_=CC_, DD_=DD_, AB_=AB_, BC_=BC_, CD_=CD_, AC_=AC_, BD_=BD_, AD_=AD_)
        
        # Create HDF5Data object with datetime
        data = HDF5Data(freq=freq, time=time_data, spec=spec)
        
        return data
    
    finally:
        if f is not None:
            f.close()


def plot_waterwall(
    data: HDF5Data,
    figsize: tuple[int, int] = (15, 5),
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    clim_per_plot: dict[str, tuple[float, float]] | None = None,
    data_type: str = "mag",
    attribute: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Create a waterwall plot of spectrograph data: either a 2x5 grid of all ten
    channel pairs or a single plot for one attribute.

    Usage:
    # 2x5 grid of magnitude for all 10 correlations
    plot_waterwall(data, data_type="mag", attribute=None)

    # Single plot of AA_ phase
    plot_waterwall(data, data_type="phase", attribute="AA_")

    # 2x5 grid of real part
    plot_waterwall(data, data_type="real")

    Parameters
    ----------
    data : HDF5Data
        HDF5Data object containing frequency, time, and spec data.
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
    """
    if data.spec is None:
        raise ValueError("HDF5Data object must have spec data to plot")

    plot_order = ['AA_', 'BB_', 'CC_', 'DD_', 'AB_', 'BC_', 'CD_', 'AC_', 'BD_', 'AD_']
    correlations = {
        'AA_': data.spec.AA_,
        'BB_': data.spec.BB_,
        'CC_': data.spec.CC_,
        'DD_': data.spec.DD_,
        'AB_': data.spec.AB_,
        'BC_': data.spec.BC_,
        'CD_': data.spec.CD_,
        'AC_': data.spec.AC_,
        'BD_': data.spec.BD_,
        'AD_': data.spec.AD_,
    }

    if attribute is not None and attribute not in plot_order:
        raise ValueError(
            f"attribute must be None or one of {plot_order}, got {attribute!r}"
        )

    freq = data.freq
    time = data.time

    # Process complex data based on data_type (real, imag, mag, phase)
    data_type_lower = data_type.lower()
    if data_type_lower not in ("real", "imag", "mag", "magnitude", "phase"):
        raise ValueError(
            f"data_type must be one of 'real', 'imag', 'mag', 'phase', got {data_type!r}"
        )

    plot_data = {}
    for name, corr_data in correlations.items():
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

    # Decide which attributes to plot
    if attribute is not None:
        to_plot = [attribute]
        fig, axes_flat = plt.subplots(1, 1, figsize=(5, 5))
        axes_flat = np.atleast_1d(axes_flat)
    else:
        to_plot = plot_order
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
            ax.set_ylabel("Time", fontsize=10)
            ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
            ax.yaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            ax.set_ylabel("Time (s)", fontsize=10)
        ax.set_title(f"Correlation: {name}", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Raw measurements").ax.tick_params(labelsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


class CalData:
    """Container for calibration data with attribute access."""
    def __init__(self, freq: np.ndarray, te: np.ndarray, gain: np.ndarray):
        """
        Initialize CalData with Gain (G) and Effective noise temperature (T_e)
        Parameters
        ----------
        freq : np.ndarray
            Frequency channels (in MHz) - typically 256 channels (shape: 256, n_freq)
        Te : np.ndarray or list[datetime]
            effective noise temperature (Te) is in Kelvin (shape: n_channels, n_freq)
        Gain : np.ndarray
            The Gain (G) is in a linear scale(shape: n_channels, n_freq)
        """
        self.freq = freq
        self.te = te
        self.gain = gain

    def __repr__(self) -> str:
        return (
            f"CalData(freq: {self.freq.shape}, te: {self.te.shape}, gain: {self.gain.shape})"
        )



def read_cal_hdf5(file_path: str | Path) -> CalData:   
    """
    Read a calibration HDF5 file and return the data as a CalData object with attribute access.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.
    """
    file_path = Path(file_path)
    f = h5py.File(file_path, 'r')
    data = {}
    for key in f.keys():
        data[key] = f[key][()]

    freq = np.linspace(310e6 - 20.48e6/2, 310e6 + 20.48e6/2, 256)
    '''
    310 MHz center w/ 20.48MHz BW at 256 FFT channels -> RBW: 80 kHz)
    '''
    te = data['te']
    gain = data['gain']
    return CalData(freq=freq, te=te, gain=gain)

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
    

    label = ['AA*', 'BB*', 'CC*', 'DD*', 'AB*', 'BC*', 'CD*', 'AC*', 'BD*', 'AD*']
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))    
    axes = axes.flatten()
    for i in range(2):
        for j in range(5):
            ax = axes[i*5 + j]
            ax.plot(cal_data.freq/1e6, getattr(cal_data, attribute)[i*5 + j, :], label=label[i*5 + j])
            ax.set_xlabel('Frequency (MHz)', fontsize=10)
            ax.legend(fontsize=8)
        
    fig.suptitle(f'{attribute.title()}', fontsize=10)
        
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()