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


def read_hdf5_data(
    file_path: str | Path,
    skip_errors: bool = True,
    libver: str | None = None
) -> tuple[dict[str, Any], list[str]]:
    """
    Read an HDF5 file and return the data and keys.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.
    skip_errors : bool, default=True
        If True, skip datasets that cannot be read and continue with others.
        If False, raise an error when a dataset cannot be read.
    libver : str, optional
        HDF5 library version compatibility. Options: 'earliest', 'latest', 'v108', 'v110', 'v112'.
        If None, uses default. Try 'latest' if encountering read errors.
    
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
    failed_datasets = []
    
    # Try different file opening strategies
    file_kwargs = {}
    if libver is not None:
        file_kwargs['libver'] = libver
    
    # Try opening with different strategies
    f = None
    strategies = [
        {'mode': 'r', **file_kwargs},
        {'mode': 'r', 'libver': 'latest', **{k: v for k, v in file_kwargs.items() if k != 'libver'}},
        {'mode': 'r', 'libver': 'earliest', **{k: v for k, v in file_kwargs.items() if k != 'libver'}},
        {'mode': 'r', 'swmr': True, **file_kwargs},  # Single Writer Multiple Reader mode
    ]
    
    for strategy in strategies:
        try:
            f = h5py.File(file_path, **strategy)
            # Test if we can actually read from the file
            try:
                _ = list(f.keys())
                break
            except Exception:
                f.close()
                f = None
                continue
        except Exception:
            continue
    
    if f is None:
        # Last resort: try default opening
        f = h5py.File(file_path, 'r', **file_kwargs)
    
    try:
        # Get all top-level keys
        keys = list(f.keys())
        
        def extract_data(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            """Recursively extract datasets from HDF5 file."""
            if isinstance(obj, h5py.Dataset):
                try:
                    # Check if this is a compound type (structured array)
                    dtype = obj.dtype
                    is_compound = dtype.names is not None
                    
                    # Check if it's a complex number stored as compound type (r, i fields)
                    is_complex_compound = (
                        is_compound and 
                        len(dtype.names) == 2 and 
                        'r' in dtype.names and 
                        'i' in dtype.names
                    )
                    
                    if obj.size == 0:
                        # Empty dataset
                        if is_complex_compound:
                            data[name] = np.array([], dtype=np.complex64)
                        else:
                            data[name] = np.array([], dtype=obj.dtype)
                    elif obj.ndim == 0:
                        # Scalar dataset
                        if is_complex_compound:
                            val = obj[()]
                            data[name] = np.complex64(val['r'] + 1j * val['i'])
                        else:
                            data[name] = np.array(obj[()])
                    else:
                        # Multi-dimensional dataset
                        # Try reading with low-level API first if regular read fails
                        try:
                            if is_complex_compound:
                                # Read as structured array first, then convert to complex
                                # Read in chunks to handle large datasets
                                if obj.chunks is not None and obj.ndim == 3:
                                    # Read plane by plane for 3D compound arrays
                                    chunk_0 = obj.chunks[0] if obj.chunks else 64
                                    arr_real = np.empty(obj.shape, dtype=np.float32)
                                    arr_imag = np.empty(obj.shape, dtype=np.float32)
                                    
                                    for i in range(0, obj.shape[0], chunk_0):
                                        end_i = min(i + chunk_0, obj.shape[0])
                                        chunk_data = obj[i:end_i, :, :]
                                        arr_real[i:end_i, :, :] = chunk_data['r']
                                        arr_imag[i:end_i, :, :] = chunk_data['i']
                                    
                                    data[name] = arr_real + 1j * arr_imag
                                else:
                                    # Read entire array
                                    arr_structured = obj[:]
                                    data[name] = arr_structured['r'] + 1j * arr_structured['i']
                            else:
                                # Regular array - use full slice
                                data[name] = obj[:]
                        except (OSError, RuntimeError) as read_error:
                            # If regular read fails, try using low-level API with custom properties
                            try:
                                import h5py._hl.dataset as ds
                                # Create a new dataset access property list
                                dapl = obj.id.get_access_plist()
                                
                                # Try reading with the dataset's own transfer properties
                                if is_complex_compound:
                                    arr_structured = np.empty(obj.shape, dtype=obj.dtype)
                                    # Use read_direct with the dataset's own properties
                                    obj.read_direct(arr_structured)
                                    data[name] = arr_structured['r'] + 1j * arr_structured['i']
                                else:
                                    arr = np.empty(obj.shape, dtype=obj.dtype)
                                    obj.read_direct(arr)
                                    data[name] = arr
                            except Exception:
                                # Re-raise the original error to trigger fallback methods
                                raise read_error
                except (OSError, RuntimeError, ValueError) as e:
                    # If slicing fails, try reading with read_direct
                    try:
                        dtype = obj.dtype
                        is_compound = dtype.names is not None
                        is_complex_compound = (
                            is_compound and 
                            len(dtype.names) == 2 and 
                            'r' in dtype.names and 
                            'i' in dtype.names
                        )
                        
                        if is_complex_compound:
                            # For compound types, read as structured array
                            arr_structured = np.empty(obj.shape, dtype=obj.dtype)
                            obj.read_direct(arr_structured, source_sel=None, dest_sel=None)
                            data[name] = arr_structured['r'] + 1j * arr_structured['i']
                        else:
                            arr = np.empty(obj.shape, dtype=obj.dtype)
                            obj.read_direct(arr, source_sel=None, dest_sel=None)
                            data[name] = arr
                    except Exception as e2:
                        # Last resort: try reading in chunks aligned with dataset chunking
                        try:
                            dtype = obj.dtype
                            is_compound = dtype.names is not None
                            is_complex_compound = (
                                is_compound and 
                                len(dtype.names) == 2 and 
                                'r' in dtype.names and 
                                'i' in dtype.names
                            )
                            
                            # Use dataset's chunk size if available, otherwise use reasonable defaults
                            if obj.chunks is not None:
                                chunk_sizes = obj.chunks
                            else:
                                # Default chunk sizes if not chunked
                                if obj.ndim == 1:
                                    chunk_sizes = (min(10000, obj.shape[0]),)
                                elif obj.ndim == 2:
                                    chunk_sizes = (min(1000, obj.shape[0]), obj.shape[1])
                                else:
                                    chunk_sizes = tuple(min(100, s) for s in obj.shape)
                            
                            if is_complex_compound:
                                # For complex compound types, read as structured array in chunks
                                arr_real = np.empty(obj.shape, dtype=np.float32)
                                arr_imag = np.empty(obj.shape, dtype=np.float32)
                                
                                if obj.ndim == 3:
                                    chunk_0 = chunk_sizes[0] if len(chunk_sizes) > 0 else 64
                                    for i in range(0, obj.shape[0], chunk_0):
                                        end_i = min(i + chunk_0, obj.shape[0])
                                        chunk_data = obj[i:end_i, :, :]
                                        arr_real[i:end_i, :, :] = chunk_data['r']
                                        arr_imag[i:end_i, :, :] = chunk_data['i']
                                elif obj.ndim == 2:
                                    chunk_0 = chunk_sizes[0] if len(chunk_sizes) > 0 else 1000
                                    for i in range(0, obj.shape[0], chunk_0):
                                        end_i = min(i + chunk_0, obj.shape[0])
                                        chunk_data = obj[i:end_i, :]
                                        arr_real[i:end_i, :] = chunk_data['r']
                                        arr_imag[i:end_i, :] = chunk_data['i']
                                else:
                                    chunk_data = obj[:]
                                    arr_real = chunk_data['r']
                                    arr_imag = chunk_data['i']
                                
                                data[name] = arr_real + 1j * arr_imag
                            else:
                                # Regular array reading
                                arr = np.empty(obj.shape, dtype=obj.dtype)
                                
                                if obj.ndim == 1:
                                    # 1D: read in chunks
                                    chunk_size = chunk_sizes[0]
                                    for i in range(0, obj.shape[0], chunk_size):
                                        end = min(i + chunk_size, obj.shape[0])
                                        arr[i:end] = obj[i:end]
                                elif obj.ndim == 2:
                                    # 2D: read in blocks aligned with chunks
                                    chunk_0, chunk_1 = chunk_sizes[0], chunk_sizes[1]
                                    for i in range(0, obj.shape[0], chunk_0):
                                        end_i = min(i + chunk_0, obj.shape[0])
                                        for j in range(0, obj.shape[1], chunk_1):
                                            end_j = min(j + chunk_1, obj.shape[1])
                                            arr[i:end_i, j:end_j] = obj[i:end_i, j:end_j]
                                elif obj.ndim == 3:
                                    # 3D: read in blocks aligned with chunks
                                    chunk_0, chunk_1, chunk_2 = chunk_sizes[0], chunk_sizes[1], chunk_sizes[2]
                                    for i in range(0, obj.shape[0], chunk_0):
                                        end_i = min(i + chunk_0, obj.shape[0])
                                        for j in range(0, obj.shape[1], chunk_1):
                                            end_j = min(j + chunk_1, obj.shape[1])
                                            for k in range(0, obj.shape[2], chunk_2):
                                                end_k = min(k + chunk_2, obj.shape[2])
                                                arr[i:end_i, j:end_j, k:end_k] = obj[i:end_i, j:end_j, k:end_k]
                                else:
                                    # Higher dimensions: read in smaller blocks
                                    chunk_size = chunk_sizes[0] if len(chunk_sizes) > 0 else 100
                                    for i in range(0, obj.shape[0], chunk_size):
                                        end_i = min(i + chunk_size, obj.shape[0])
                                        slices = (slice(i, end_i),) + (slice(None),) * (obj.ndim - 1)
                                        arr[slices] = obj[slices]
                                
                                data[name] = arr
                        except Exception as e3:
                            # If all methods fail, try reading element-by-element or plane-by-plane
                            # This is a last resort for corrupted or problematic files
                            try:
                                dtype = obj.dtype
                                is_compound = dtype.names is not None
                                is_complex_compound = (
                                    is_compound and 
                                    len(dtype.names) == 2 and 
                                    'r' in dtype.names and 
                                    'i' in dtype.names
                                )
                                
                                if is_complex_compound:
                                    # For complex compound types, read plane by plane
                                    arr_real = np.empty(obj.shape, dtype=np.float32)
                                    arr_imag = np.empty(obj.shape, dtype=np.float32)
                                    
                                    if obj.ndim == 3 and obj.shape[0] < 100000:
                                        for i in range(obj.shape[0]):
                                            try:
                                                plane_data = obj[i, :, :]
                                                arr_real[i, :, :] = plane_data['r']
                                                arr_imag[i, :, :] = plane_data['i']
                                            except Exception:
                                                raise e3
                                    elif obj.ndim == 2 and obj.shape[0] < 100000:
                                        for i in range(obj.shape[0]):
                                            try:
                                                row_data = obj[i, :]
                                                arr_real[i, :] = row_data['r']
                                                arr_imag[i, :] = row_data['i']
                                            except Exception:
                                                raise e3
                                    else:
                                        raise e3
                                    
                                    data[name] = arr_real + 1j * arr_imag
                                else:
                                    # Regular array
                                    arr = np.empty(obj.shape, dtype=obj.dtype)
                                    if obj.ndim == 3 and obj.shape[0] < 100000:
                                        for i in range(obj.shape[0]):
                                            try:
                                                arr[i, :, :] = obj[i, :, :]
                                            except Exception:
                                                raise e3
                                    elif obj.ndim == 2 and obj.shape[0] < 100000:
                                        for i in range(obj.shape[0]):
                                            try:
                                                arr[i, :] = obj[i, :]
                                            except Exception:
                                                raise e3
                                    else:
                                        raise e3
                                    data[name] = arr
                            except Exception as e4:
                                # If all methods fail, raise an error
                                # This will be caught by the outer handler if skip_errors=True
                                raise OSError(
                                    f"Failed to read dataset '{name}' (shape={obj.shape}, "
                                    f"dtype={obj.dtype}, chunks={obj.chunks}). "
                                    f"Original error: {e}. "
                                    f"Alternative methods failed: {e2}, {e3}, {e4}"
                                ) from e4
            elif isinstance(obj, h5py.Group):
                # Recursively process groups
                for key in obj.keys():
                    extract_data(f"{name}/{key}" if name else key, obj[key])
        
        # Extract all datasets
        for key in keys:
            try:
                extract_data(key, f[key])
            except Exception as e:
                if skip_errors:
                    failed_datasets.append((key, str(e)))
                    print(f"Warning: Skipping dataset '{key}': {e}")
                else:
                    raise
        
        if failed_datasets and skip_errors:
            print(f"\nNote: {len(failed_datasets)} dataset(s) could not be read and were skipped.")
            print("Failed datasets:", [name for name, _ in failed_datasets])
            print("\nIf you're getting 'wrong B-tree signature' errors, the file may be corrupted.")
            print("Try:")
            print("  1. Use utils.open_hdf5_file() to access datasets directly")
            print("  2. Repair the file using: h5repack input.hdf5 output.hdf5")
            print("  3. Check if the file was properly closed when written")
    
    finally:
        if f is not None:
            f.close()
    
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


def open_hdf5_file(
    file_path: str | Path,
    mode: str = 'r',
    libver: str | None = None
) -> h5py.File:
    """
    Open an HDF5 file with various fallback strategies.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the HDF5 file.
    mode : str, default='r'
        File access mode ('r' for read, 'r+' for read-write, etc.)
    libver : str, optional
        HDF5 library version compatibility.
    
    Returns
    -------
    h5py.File
        Open HDF5 file object.
    
    Notes
    -----
    If you encounter "wrong B-tree signature" errors, the file may be corrupted
    or written with an incompatible HDF5 version. Try:
    1. Using h5repack to repair the file: `h5repack input.hdf5 output.hdf5`
    2. Opening with different libver options
    3. Checking if the file was properly closed when written
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")
    
    file_kwargs = {}
    if libver is not None:
        file_kwargs['libver'] = libver
    
    strategies = [
        {'mode': mode, **file_kwargs},
        {'mode': mode, 'libver': 'latest', **{k: v for k, v in file_kwargs.items() if k != 'libver'}},
        {'mode': mode, 'libver': 'earliest', **{k: v for k, v in file_kwargs.items() if k != 'libver'}},
    ]
    
    last_error = None
    for strategy in strategies:
        try:
            f = h5py.File(file_path, **strategy)
            # Test if we can actually read from the file
            try:
                _ = list(f.keys())
                return f
            except Exception as e:
                f.close()
                last_error = e
                continue
        except Exception as e:
            last_error = e
            continue
    
    # Last resort: try default opening
    try:
        return h5py.File(file_path, mode, **file_kwargs)
    except Exception as e:
        raise OSError(
            f"Failed to open HDF5 file '{file_path}'. "
            f"Last error: {last_error or e}. "
            f"The file may be corrupted or require repair. "
            f"Try using 'h5repack' to repair the file."
        ) from (last_error or e)


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
