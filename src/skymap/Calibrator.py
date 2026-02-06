import numpy as np
from skymap.io import HDF5Data, CalData, CAL_POL_NAMES, CalibratedSpec
from astropy.constants import k_B
import astropy.units as u
'''
Functions to calibrate the raw data
'''

const_bw= 20.48*10**6*u.Hz #Hz (bandwidth of the receiver)
const_rbw = const_bw/256 #Hz (resolution bandwidth)
const_R = 50*u.ohm #Ohm

def get_Tsys_lab(corr_spec: np.ndarray, gain: np.ndarray):
    """
    Get the system temperature from the data.
    Parameters
    ----------
    data: HDF5Data
        The uncalibrated data.


    Formula:
    P = k_B * Tsys * rbw
    power -> voltage conversion: P = V_input^2 / R
    Measured voltage: V_measured **2  = G * V_input **2
    V_measured**2 / (R * G) = k_B * Tsys * rbw

    Tsys = V_measured**2 / (k_B * rbw * R * G)

    for two sided spectrum normalization (+1/-1 scaling) in the spectrometer, we divide by 2. So,

    Tsys = V_measured**2 / (2 *k_B * rbw * R * G)
    where:
    V_measured^2 is unitless auto/cross correlation spectra on a +/- 1 scale
    coss_spec = V_measured **2
    k_B is the Boltzmann constant in 1.38x10^-23 m^2 kg s^-2 K^-1
    rbw is the resolution bandwidth in Hz
    G is the gain of the receiver
    R us 50 ohm load resistor
    Tsys is the system temperature in K

    """
    Tsys = corr_spec/ (2 * k_B * const_rbw * const_R * gain)
    return Tsys.to(u.K)

def get_Tsky_lab(corr_spec: np.ndarray, gain: np.ndarray, Te: np.ndarray):
    """
    Get the sky temperature from the data.
    Parameters
    ----------
    corr_spec: np.ndarray
        The uncalibrated data.
    gain: np.ndarray
        The gain of the receiver.
    Te: np.ndarray
        The effective noise temperature of the receiver.
    Returns
    -------
    The sky temperature in K.
    """
    Tsys = get_Tsys_lab(corr_spec, gain)

    #check here if the cross-correlations need to handled differently since they are complex numbers. 
    # Autos are real numbers, but need to do np.abs() to remove the zero imaginary part.
    Tsky = np.abs(Tsys) - Te
    return Tsky

def lab_cal(data: HDF5Data, cal_data: CalData, attribute: str | None = None) -> HDF5Data:

    """
    Apply the lab calibration to the data.
    Parameters
    ----------
    data: HDF5Data
        The uncalibrated data. 
        This data is auto/cross correlations from SDR in V^2 on a unitless +/- 1 scale.
        The data is in the shape of (n_time, n_spectra, n_freq).
    cal_data: CalData
        The calibration data.
        This data is the gain and te from the lab calibration.

    attribute: str | None = None
        The attribute of the data to calibrate.
        If None, all attributes will be calibrated.
    
    Returns
    -------
    The calibrated data HDF5Data object.

    """
    # set attributes to calibrate
    if attribute is None:
        attributes = CAL_POL_NAMES
    else:
        attributes = [attribute]

    # initialize calibrated_spec on data (same shape as data.spec: data.calibrated_spec.AA_, etc.)
    if not hasattr(data, 'calibrated_spec'):
        data.calibrated_spec = CalibratedSpec()

    # check if the requested attribute(s) are already calibrated
    for attr in attributes:
        if getattr(data.calibrated_spec, attr, None) is not None:
            raise ValueError(f"Data for {attr} is already calibrated")
            return data

    # calibrate the data (cal_data.gain.AA_, cal_data.te.AA_, etc.)
    for attr in attributes:
        corr_spec = getattr(data.spec, attr) * u.V**2
        g = getattr(cal_data.gain, attr)
        te = getattr(cal_data.te, attr) * u.K
        calibrated_spec = get_Tsky_lab(corr_spec, g, te)
        setattr(data.calibrated_spec, attr, calibrated_spec)
    return data