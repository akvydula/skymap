'''
utilities for skymap package
'''

import numpy as np
from datetime import datetime
from astropy.time import Time

def _time_to_mjd(time_array: np.ndarray) -> np.ndarray:
    """Convert time array to MJD (days, float) for matching. Handles datetime and numeric (MJD or Unix)."""
    if len(time_array) == 0:
        return np.array([], dtype=float)
    sample = time_array.flat[0]
    if isinstance(sample, (datetime, np.datetime64)):
        return Time(time_array).mjd
    time_array = np.asarray(time_array, dtype=float)
    # Numeric: assume MJD if in calendar range (~5e4–6e5), else Unix (sec or ms)
    if np.all(time_array >= 4e4) and np.all(time_array < 7e5):
        return time_array
    if np.any(time_array > 1e12):
        time_array = time_array / 1000.0  # Unix ms -> sec
    return Time(time_array, format="unix").mjd