skymap

This package is a collection of tools for the analysis and visualization of radio sky map at 310 MHz.

## Notebooks (`notebooks/`)

| File | Description |
|------|-------------|
| `beam_jan_2026.ipynb` | Beam mapping: select spectrometer and pointing data and plot beam measurements (Jan 2026). |
| `commission_jan_2026.ipynb` | Commissioning: load and explore GBT beam-scan HDF5 data. |
| `pointing_data_tests.ipynb` | Pointing data I/O, FITS, match with spec, and healmap RA/Dec plots. |

## Source (`src/skymap/`)

| File | Description |
|------|-------------|
| `healmap.py` | HealPix maps from pointing data (1 sq.deg beam); fill from RA/Dec, plot mollview or RA/Dec axes. |
| `io.py` | HDF5 and FITS I/O, PointingData, SpecData, match_data_and_pointing, calibration read/write. |
| `plots.py` | Plotting utilities for calibration (gain/Te) and polarization/spec with optional pointing. |
| `utils.py` | Time conversions (MJD, Unix) and helpers for the skymap package. |
| `Calibrator.py` | Calibrate raw data: Tsys, gain, and temperature from lab/observation. |
| `__init__.py` | Package init and version. |


