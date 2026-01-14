"""
Utility functions for reading HDF5 files and plotting polarization data.

"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_hdf5_data(file_path: str | Path) -> tuple[dict[str, Any], list[str]]:
    """
    Read an HDF5 file and return the data and keys.
    
    Parameters
    ----------
    file_path : str or Path
    
    Returns
    -------
    data : dict
        Dictionary containing all datasets from the HDF5 file.
        Keys are the dataset names, values are numpy arrays.
    keys : list[str]
        List of all top-level keys in the HDF5 file.
    
    """
    file_path = Path(file_path)
    
    # check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")
    
    data = {}
    keys = []
    
    with h5py.File(file_path, 'r') as f:
        # Get all top-level keys
        keys = list(f.keys())
        
        def extract_data(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            """Recursively extract datasets from HDF5 file."""
            if isinstance(obj, h5py.Dataset):
                data[name] = np.array(obj)
            elif isinstance(obj, h5py.Group):
                # Recursively process groups
                for key in obj.keys():
                    extract_data(f"{name}/{key}" if name else key, obj[key])
        
        # Extract all datasets
        for key in keys:
            extract_data(key, f[key])
    
    return data, keys


def plot_polarizations(
    time: np.ndarray,
    frequency: np.ndarray,
    polarizations: dict[str, np.ndarray] | list[np.ndarray],
    polarization_names: list[str] | None = None,
    figsize: tuple[int, int] = (15, 10),
    cmap: str = "viridis",
    save_path: str | Path | None = None,
) -> None:
    """
    Plot 2D time-frequency plots for multiple polarizations.
    
    Parameters
    ----------
    time : np.ndarray
        Time values (1D array). Shape should be (n_time,).
    frequency : np.ndarray
        Frequency values (1D array). Shape should be (n_freq,).
    polarizations : dict[str, np.ndarray] or list[np.ndarray]
        Polarization data. Can be:
        - dict: keys are polarization names, values are 2D arrays of shape (n_time, n_freq)
        - list: list of 2D arrays, each of shape (n_time, n_freq)
    polarization_names : list[str], optional
        Names for polarizations. Required if polarizations is a list.
        If polarizations is a dict, this parameter is ignored (uses dict keys).
    figsize : tuple[int, int], default=(15, 10)
        Figure size (width, height) in inches.
    cmap : str, default="viridis"
        Colormap to use for the plots.
    save_path : str or Path, optional
        If provided, save the figure to this path.
    
    Examples
    --------
    >>> time = np.linspace(0, 3600, 100)  # 1 hour of data
    >>> freq = np.linspace(1e9, 2e9, 50)  # 1-2 GHz
    >>> pol_data = {
    ...     'XX': np.random.randn(100, 50),
    ...     'YY': np.random.randn(100, 50),
    ...     'XY': np.random.randn(100, 50),
    ...     'YX': np.random.randn(100, 50)
    ... }
    >>> plot_polarizations(time, freq, pol_data)
    """
    # Handle different input formats
    if isinstance(polarizations, dict):
        pol_dict = polarizations
        if polarization_names is None:
            polarization_names = list(pol_dict.keys())
        else:
            # Use provided names but ensure they match dict keys
            if set(polarization_names) != set(pol_dict.keys()):
                raise ValueError(
                    "polarization_names must match dict keys when polarizations is a dict"
                )
    elif isinstance(polarizations, list):
        if polarization_names is None:
            polarization_names = [f"Pol_{i+1}" for i in range(len(polarizations))]
        elif len(polarization_names) != len(polarizations):
            raise ValueError(
                "polarization_names length must match polarizations length"
            )
        pol_dict = {name: pol for name, pol in zip(polarization_names, polarizations)}
    else:
        raise TypeError(
            "polarizations must be either a dict or list of numpy arrays"
        )
    
    n_pols = len(pol_dict)
    
    # Determine subplot layout
    if n_pols == 1:
        nrows, ncols = 1, 1
    elif n_pols == 2:
        nrows, ncols = 1, 2
    elif n_pols <= 4:
        nrows, ncols = 2, 2
    else:
        ncols = 2
        nrows = (n_pols + 1) // 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    # Create meshgrid for plotting
    T, F = np.meshgrid(time, frequency, indexing='ij')
    
    for idx, (pol_name, pol_data) in enumerate(pol_dict.items()):
        ax = axes[idx]
        
        # Validate data shape
        if pol_data.shape != (len(time), len(frequency)):
            raise ValueError(
                f"Polarization '{pol_name}' shape {pol_data.shape} does not match "
                f"expected shape ({len(time)}, {len(frequency)})"
            )
        
        # Create 2D plot
        im = ax.pcolormesh(T, F, pol_data, cmap=cmap, shading='auto')
        
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", fontsize=10)
        ax.set_title(f"Polarization: {pol_name}", fontsize=12, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Intensity")
        
        # Format frequency axis if needed
        if frequency.max() > 1e6:
            # Convert to MHz if frequencies are large
            ax.set_ylabel("Frequency (MHz)", fontsize=10)
            # Update y-axis ticks
            yticks = ax.get_yticks()
            ax.set_yticklabels([f"{t/1e6:.1f}" for t in yticks])
    
    # Hide unused subplots
    for idx in range(n_pols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def read_all_hdf5_files(data_dir: str | Path = "data_310") -> dict[str, tuple[dict, list]]:
    """
    Read all HDF5 files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path, default="data_310"
        Directory containing HDF5 files.
    
    Returns
    -------
    results : dict
        Dictionary mapping filename to (data, keys) tuple.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent / data_dir
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    hdf5_files = glob.glob(str(data_dir / "*.hdf5"))
    
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")
    
    results = {}
    for file_path in hdf5_files:
        filename = Path(file_path).name
        data, keys = read_hdf5_data(file_path)
        results[filename] = (data, keys)
    
    return results
