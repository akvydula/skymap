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
    
    def __init__(self, XX: np.ndarray, YY: np.ndarray, XY: np.ndarray, YX: np.ndarray):
        """
        Initialize SpecData with polarization arrays.
        
        Parameters
        ----------
        XX : np.ndarray
            XX polarization data (2D array: time x frequency)
        YY : np.ndarray
            YY polarization data (2D array: time x frequency)
        XY : np.ndarray
            XY polarization data (2D array: time x frequency)
        YX : np.ndarray
            YX polarization data (2D array: time x frequency)
        """
        self.XX = XX
        self.YY = YY
        self.XY = XY
        self.YX = YX
    
    def __repr__(self) -> str:
        return (
            f"SpecData(XX: {self.XX.shape}, YY: {self.YY.shape}, "
            f"XY: {self.XY.shape}, YX: {self.YX.shape})"
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
        Initialize HDF5Data with frequency, time, and optional spec data.
        
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


def read_hdf5_data(
    file_path: str | Path,
    skip_errors: bool = True,
    time_slice: slice | None = slice(0, 38000),
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
        Default is slice(0, 38000). Set to None to read all data.
    time_mjd : bool, optional
        If True, convert time values from MJD to datetime.
        If False, convert time values from Unix timestamp to datetime. Default is False
    Returns
    -------
    HDF5Data
        HDF5Data object with attribute-style access:
        - data.freq : frequency array (in MHz)
        - data.time : time array
        - data.spec.XX : XX Correlation data
        - data.spec.YY : YY Correlation data
        - data.spec.XY : XY Correlation data
        - data.spec.YX : YX Correlation data

    """
    file_path = Path(file_path)
    
    # check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    f = h5py.File(file_path, 'r')
    
    try:
        # Read frequency data (convert to MHz)
        freq = f['data']['freq'][()] / 1e6
        
        time_data = f['data']['time'][time_slice]
        
        # Convert time_data to datetime 
        if time_mjd:
            time_datetime = np.array([Time(ts, format='mjd').datetime for ts in time_data])
        else:
        # time values likely in Unix timestamp format in ms
            time_datetime = np.array([datetime.fromtimestamp(ts/1000) for ts in time_data])

        
        # Read spec data (complex array with shape: n_time, n_pol, n_freq)
        spec_data = f['data']['spec'][time_slice, 0:4, :]
        

        # Extract polarization data (indices 0, 1, 2, 3 correspond to XX, YY, XY, YX)
        # spec_data shape: (n_time, n_pol, n_freq)
        # We want 2D arrays: (n_time, n_freq) for each polarization
        XX = spec_data[:, 0, :]
        YY = spec_data[:, 1, :]
        XY = spec_data[:, 2, :]
        YX = spec_data[:, 3, :]
        
        # Create SpecData object
        spec = SpecData(XX=XX, YY=YY, XY=XY, YX=YX)
        
        # Create HDF5Data object with datetime
        data = HDF5Data(freq=freq, time=time_datetime, spec=spec)
        
        return data
    
    finally:
        if f is not None:
            f.close()


def plot_waterwall(
    data: HDF5Data,
    figsize: tuple[int, int] = (15, 10),
    cmap: str = "viridis",
    clim: tuple[float, float] | None = None,
    clim_per_plot: dict[str, tuple[float, float]] | None = None,
    data_type: str = "magnitude",
    save_path: str | Path | None = None,
) -> None:
    """
    Create a 2x2 waterwall plot grid showing all four correlations.
    
    Parameters
    ----------
    data : HDF5Data
        HDF5Data object containing frequency, time, and spec data.
    figsize : tuple[int, int], default=(15, 10)
        Figure size (width, height) in inches.
    cmap : str, default="viridis"
        Colormap to use for the plots.
    clim : tuple[float, float], optional
        Global color limits (vmin, vmax) applied to all plots. 
        If None, computed from all data. Overridden by clim_per_plot if specified.
    clim_per_plot : dict[str, tuple[float, float]], optional
        Per-plot color limits. Keys are correlation names ('XX', 'YY', 'XY', 'YX'),
        values are (vmin, vmax) tuples. Overrides clim for specified plots.
    data_type : str, default="magnitude"
        Type of data to plot for complex arrays. Options:
        - "magnitude": Plot |data| (default)
        - "real": Plot Re(data)
        - "imag": Plot Im(data)
        - "phase": Plot phase in radians
    save_path : str or Path, optional
        If provided, save the figure to this path.
    

    """
    if data.spec is None:
        raise ValueError("HDF5Data object must have spec data to plot")
    
    # Extract data
    freq = data.freq  
    time = data.time
    
    # Get correlation data
    correlations = {
        'XX': data.spec.XX,
        'YY': data.spec.YY,
        'XY': data.spec.XY,
        'YX': data.spec.YX,
    }
    
    # Process complex data based on data_type
    plot_data = {}
    for name, corr_data in correlations.items():
        if np.iscomplexobj(corr_data):
            if data_type == "magnitude":
                plot_data[name] = np.abs(corr_data)
            elif data_type == "real":
                plot_data[name] = np.real(corr_data)
            elif data_type == "imag":
                plot_data[name] = np.imag(corr_data)
            elif data_type == "phase":
                plot_data[name] = np.angle(corr_data)
            else:
                raise ValueError(
                    f"data_type must be one of 'magnitude', 'real', 'imag', 'phase', "
                    f"got '{data_type}'"
                )
        else:
            plot_data[name] = corr_data
    
    # Compute global color limits if not provided
    if clim is None and clim_per_plot is None:
        all_values = np.concatenate([plot_data[name].flatten() for name in plot_data.keys()])
        clim = (np.nanmin(all_values), np.nanmax(all_values))
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot order: XX, YY, XY, YX
    plot_order = ['XX', 'YY', 'XY', 'YX']
    
    is_datetime = (
        isinstance(time, np.ndarray) and 
        len(time) > 0 and 
        (isinstance(time[0], datetime) or isinstance(time[0], np.datetime64))
    )
    
    if is_datetime:
        if isinstance(time[0], datetime):
            time_plot = mdates.date2num(time)
        else:
            time_dt = [datetime.fromtimestamp(ts.astype('datetime64[s]').astype(int)) for ts in time]
            time_plot = mdates.date2num(time_dt)
        F, T = np.meshgrid(freq, time_plot, indexing='xy')
    else:
        F, T = np.meshgrid(freq, time, indexing='xy')
    
    for idx, corr_name in enumerate(plot_order):
        ax = axes[idx]
        corr_data = plot_data[corr_name]
        
        # Validate data shape
        if corr_data.shape != (len(time), len(freq)):
            raise ValueError(
                f"Correlation '{corr_name}' shape {corr_data.shape} does not match "
                f"expected shape ({len(time)}, {len(freq)})"
            )
        
        # Determine color limits for this plot
        if clim_per_plot is not None and corr_name in clim_per_plot:
            vmin, vmax = clim_per_plot[corr_name]
        elif clim is not None:
            vmin, vmax = clim
        else:
            # Fallback: use data range for this plot
            vmin, vmax = np.nanmin(corr_data), np.nanmax(corr_data)
        
        im = ax.pcolormesh(F, T, corr_data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        
        ax.set_xlabel("Frequency (MHz)", fontsize=10)
        
        if is_datetime:
            ax.set_ylabel("Time", fontsize=10)
            ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
            ax.yaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.yaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.set_ylabel("Time (s)", fontsize=10)
        
        ax.set_title(f"Correlation: {corr_name}", fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax, label="Raw measurements")
        cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

