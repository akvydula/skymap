'''
utilities for skymap package
'''

import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
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



def _time_to_unix_ms(time_array: np.ndarray) -> np.ndarray:
    """Convert time array to Unix timestamp in milliseconds for HDF5 write."""
    if len(time_array) == 0:
        return np.array([], dtype=float)
    sample = np.asarray(time_array).flat[0]
    if isinstance(sample, (datetime, np.datetime64)):
        return (Time(time_array).unix * 1000).astype(float)
    time_array = np.asarray(time_array, dtype=float)
    if np.all(time_array >= 4e4) and np.all(time_array < 7e5):
        return (Time(time_array, format="mjd").unix * 1000).astype(float)
    if np.any(time_array > 1e12):
        return time_array.astype(float)
    return (time_array * 1000).astype(float)


def mjd_utc_to_et(mjd: np.ndarray, et_tz: str = "America/New_York") -> np.ndarray:
    """
    Convert MJD (UTC) to Eastern Time (ET) datetimes for plotting.

    Parameters
    ----------
    mjd : np.ndarray
        Modified Julian Date in days (assumed UTC).
    et_tz : str, default="America/New_York"
        Timezone name for ET (e.g. America/New_York for US Eastern).

    Returns
    -------
    np.ndarray
        Array of timezone-aware datetimes in ET (object dtype).
    """
    mjd = np.atleast_1d(np.asarray(mjd))
    t = Time(mjd, format="mjd", scale="utc")
    dt_utc = t.to_datetime()
    utc_zone = timezone.utc
    et_zone = ZoneInfo(et_tz)
    if dt_utc.ndim == 0:
        dt_utc = np.array([dt_utc.item()])
    out = np.array(
        [d.replace(tzinfo=utc_zone).astimezone(et_zone) for d in dt_utc.flat],
        dtype=object,
    ).reshape(dt_utc.shape)
    return out