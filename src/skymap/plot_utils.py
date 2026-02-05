'''
ploting utils functions
'''

from skymap.io import CalData, HDF5Data, CAL_POL_NAMES
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
import astropy.units as u

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
    

    channels = getattr(cal_data, attribute)  # PolChannels (gain or te)
    labels = ['AA*', 'BB*', 'CC*', 'DD*', 'AB*', 'BC*', 'CD*', 'AC*', 'BD*', 'AD*']
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    axes = axes.flatten()
    for idx, name in enumerate(CAL_POL_NAMES):
        ax = axes[idx]
        ax.plot(cal_data.freq / 1e6, getattr(channels, name), label=labels[idx])
        ax.set_xlabel('Frequency (MHz)', fontsize=10)
        ax.legend(fontsize=8)
        
    fig.suptitle(f'{attribute.title()}', fontsize=10)
        
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_waterwall(
    data: HDF5Data,
    calibrated: bool = False,
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
    plot_waterwall(data, data_type="mag", attribute=None, calibrated=False)

    # Single plot of AA_ phase
    plot_waterwall(data, data_type="phase", attribute="AA_")

    # 2x5 grid of real part
    plot_waterwall(data, data_type="real")

    Parameters
    ----------
    data : HDF5Data
        HDF5Data object containing frequency, time, and spec data.
    calibrated: bool, default=False
        If True, plot the calibrated data.
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
    if calibrated and getattr(data, "calibrated_spec", None) is None:
        raise ValueError("HDF5Data must have calibrated_spec to plot with calibrated=True")

    spec_source = data.calibrated_spec if calibrated else data.spec
    correlations = {name: getattr(spec_source, name) for name in CAL_POL_NAMES}
    plot_order = CAL_POL_NAMES

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
        if corr_data is None:
            continue
        # Convert to K for calibrated (astropy Quantity); plain array otherwise
        if getattr(corr_data, "unit", None) is not None:
            corr_data = corr_data.to(u.K).value
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

    # Decide which attributes to plot (only those with data, in case some calibrated channels are None)
    if attribute is not None:
        if attribute not in plot_data:
            raise ValueError(f"No data for attribute {attribute!r} (may be uncalibrated)")
        to_plot = [attribute]
        fig, axes_flat = plt.subplots(1, 1, figsize=(5, 5))
        axes_flat = np.atleast_1d(axes_flat)
    else:
        to_plot = [n for n in plot_order if n in plot_data]
        if not to_plot:
            raise ValueError("No correlation data to plot (calibrated_spec may have no channels set)")
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
        cbar_label = "Temperature (K)" if calibrated else "Raw measurements"
        plt.colorbar(im, ax=ax, label=cbar_label).ax.tick_params(labelsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()