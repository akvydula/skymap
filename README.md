skymap

This package is a collection of tools for the analysis and visualization of radio sky maps at 310 MHz.

## Notebooks (`notebooks/`)

| File | Description |
|------|-------------|
| `beam_jan_2026.ipynb` | Beam mapping: select spectrometer and pointing data and plot beam measurements (Jan 2026). |
| `commission_jan_2026.ipynb` | Commissioning: load and explore GBT beam-scan HDF5 data. |
| `pointing_data_tests.ipynb` | Pointing data I/O, FITS, match with spec, HealPix maps, beam radial offsets / fit, and RA/Dec plots. |

## Dashboard (`src/skymap/dashboard/`)

A Streamlit quick-look dashboard for beam-scan observations. It reuses the `skymap` package for all analysis and
only reads the time windows you select, so multi-GB spectrometer files stay usable.

```bash
conda activate skymap
pip install -e ".[dashboard]"        # once (adds streamlit and the `skymap-dashboard` command)
skymap-dashboard                     # run from the folder containing data_310/
```

Options: `--data-root PATH` (or env `SKYMAP_DATA_ROOT`), `--port N`, `--no-browser`; anything else is passed to
`streamlit run`. `python -m skymap.dashboard` is equivalent.

**Data layout.** Sessions are folders under `data_310/` named like `TGBT25B_614_01` (any `<PROJECT>_<NNN>_<NN>`, e.g.
`..._614_02`), containing `310/` (`cal_file_*.hdf5`, `GBTOBS_*.hdf5`) and `Antenna/` (pointing FITS,
`YYYY_MM_DD_HH:MM:SS.fits`). The data root can be changed in the sidebar.

**Trying it on "new" data.** `python -m skymap.dashboard.make_test_data` builds a synthetic session
`data_310/TGBT25B_614_99` from `TGBT25B_614_01` with `spec -> 1.2 * spec + c` (options `--scale`, `--offset-frac`,
`--offset`, `--files`, `--session`, `--force`); timestamps, calibration and pointing are copied unchanged.

**Sidebar.** Dropdowns for session, calibration file, pointing directory, receiver (spectrometer) HDF5 file and
cross-correlation (`AA*` ... `AD*` or `ALL`); source; display time zone. Two time windows are opened from the chosen
file, either by a time slider, typed start/end times, or sample indices: the *peak-scan window* (source cut used for
single/multi-scan and offset views) and the *beam-map window* (HEALPix and beam fit). Window size and an estimated
memory use are shown, with a guard for very large windows.

**Views.** Calibration (gain/Te, raw vs calibrated waterfall), Single scan (waterfall, amplitude vs time, pointing,
RA/Dec track), Peak scans (2-4 az/el legs with Dec/RA Gaussian fits), Pointing & timing offsets, HEALPix (RA/Dec
patch, full sky, gnomview), Beam fit (Gaussian beam fit, save/load JSON in `outputs/`). Defaults reproduce the Jan-16
test observation (`TGBT25B_614_01`, 3C353, AB*).

| File | Description |
|------|-------------|
| `src/skymap/dashboard/app.py` | Streamlit UI (sidebar + views). |
| `src/skymap/dashboard/cli.py` | `skymap-dashboard` launcher. |
| `src/skymap/dashboard/make_test_data.py` | Build a synthetic scaled session for testing. |
| `src/skymap/dashboard/discovery.py` | Find sessions, calibration/spectrometer files and pointing directories. |
| `src/skymap/dashboard/loaders.py` | Time-axis peek, wall-time to slice, correlation-selective windowed HDF5 read, calibration, pointing window. |
| `src/skymap/dashboard/analysis.py` | Wrappers around `skymap.io` / `Beam` / `HealPixMap` (matching, offsets, timing, HEALPix, beam fit). |
| `src/skymap/dashboard/figures.py` | Matplotlib capture of `skymap` plots and the dashboard-specific figures. |

## Source (`src/skymap/`)

| File | Description |
|------|-------------|
| `healmap.py` | `HealPixMap`: fill from pointing (1 sq.deg beam), per-pol channels, RA/Dec scatter plots, export for beam analysis. |
| `Beam.py` | Calibrator positions, radial offsets from source, Gaussian beam fit, HealPix PSF deconvolution (**radio-beam** + healpy). |
| `io.py` | HDF5 and FITS I/O, PointingData, SpecData, match_data_and_pointing, calibration read/write. |
| `plots.py` | Plotting utilities for calibration (gain/Te) and polarization/spec with optional pointing. |
| `utils.py` | Time conversions (MJD, Unix) and helpers for the skymap package. |
| `Calibrator.py` | Calibrate raw data: Tsys, gain, and temperature from lab/observation. |
| `__init__.py` | Package init and version. |

### HealPix maps (`healmap.py`)

- **`HealPixMap(nside)`** — Pixel geometry from full-sky solid angle (`FULL_SKY_STERADIANS`, `FULL_SKY_SQ_DEG`). Map-making uses a nominal **1 square degree** circular beam (`BEAM_AREA_SQ_DEG`, `BEAM_RADIUS_DEG`).
- **`fill_from_pointing(ra, dec, values=None)`** — For each pointing, all pixels whose centers fall inside the beam disk get a contribution; stored map is the **mean** per pixel, with **`_hit_count`** and per-pixel **`_map_std`** (standard deviation of contributing samples).
- **`fill_from_pointing_data(pointing_data, attributes=None, freq_index=None)`** — Fills from a pointing object (e.g. `PointingData` or matched HDF5 data). If calibrated/spec means with pol channels (`AA_`, `BB_`, …) are present, builds **separate channel maps**; otherwise a single hit-count style map. `freq_index` selects one frequency row when values are 2D; default averages over frequency.
- **`get_map` / `get_hit_count` / `get_std`** — Access the main map, hits, or std; for channels pass `attribute='AA_'` (etc.).
- **`get_map_radec_values(attribute=None)`** — Returns `(ra_deg, dec_deg, values)` for pixels with hits and finite values (e.g. custom analysis or exporting to other tools).
- **`plot(...)`** — Full-sky **`mollview`** by default, or **`ra_dec_map=True`** with **`region=(center_ra, center_dec, area_sqdeg)`** for a local **RA vs Dec** scatter of HealPix pixels in a disc. Optional **`beam_radius_deg`** scales the small beam circle drawn on RA/Dec plots (default: nominal 1 sq.deg beam). Plot **`stat='mean'`** or **`'std'`**; multiple channels render as a subplot grid when **`attribute`** is omitted.

### Beam analysis (`Beam.py`)

Workflow: match spec to pointing (**`match_data_and_pointing`**) → per-pointing **radial offset** and brightness (**`radial_offsets_from_source`**) → **Gaussian fit** → optional **`convolve_beam_with_fit`** (HealPix map + approximate PSF deconvolution using **radio-beam** + healpy).

- **`load_calibrators` / `get_source_radec`** — Read `calibrators.dat` (next to the module) and resolve a source name to `(ra_deg, dec_deg)`.
- **`radial_offsets_from_source`** — For each pointing in matched data (**`beam_obs_pointing`**, i.e. **`HDF5Data`** from **`match_data_and_pointing`**), radial offset (deg) from the source and one pol channel’s brightness (**`attribute`** such as **`AB_`**; optional **`freq_index`** vs mean over frequency, same as **`fill_from_pointing_data`**). Same **flat-sky** convention as **`plot_offset_map`**.
- **`fit_beam_gaussian`** — Fits **`baseline_k + A exp(-r²/(2σ²))`**; **`baseline_k`** is the mean *y* where **r > baseline_outer_deg** (default **1°**), not a free parameter. Returns **`baseline_k`**, **`A`**, **`sigma_deg`**, **`FWHM_deg`**, etc.
- **`beam_approximation(r, beam_params)`** — Evaluates **`baseline_k +`** Gaussian bump (convolution kernels use the bump only so weights still go to 0 at large radius when **`normalize=True`**).
- **`plot_beam_approximation`** — Radial profile plot; with **`normalize=True`** (default), the curve is scaled to **amplitude 1 at 0°** and **0 at 5°** (affine map of the fit, clipped to **[0, 1]**), consistent with the default convolution kernel.
- **`convolve_beam_with_fit`** — Builds a **`HealPixMap`** at **`nside`** from raw pointing values, then approximately **removes** the fitted circular Gaussian PSF in harmonic space (**`1 / B_l`** with **`bl_floor`** regularization; **`hp.gauss_beam`** matches **`hp.smoothing`**). The PSF width is defined with **`radio_beam.Beam`** from the fit’s FWHM (https://radio-beam.readthedocs.io/en/latest/). Returns **`ra`**, **`dec`**, **`deconvolved`** (per-channel samples interpolated from the sharpened map), **`psf_fwhm_deg`**, **`psf_radio_beam`**, and **`healpix_map`**. Use **`attribute`**, **`attributes`**, or all channels; **`freq_index`** matches **`fill_from_pointing_data`**.
- **`_beam_approximation_beam_bl`** — Lower-level **`B_ℓ`** from **`healpy.beam2bl`** (harmonic-space beam window); not used by **`convolve_beam_with_fit`**, which operates in the pointing domain.

Related helpers include **`plot_offset_map`** (offset from source on RA/Dec) for visualization alongside **`HealPixMap.plot(..., ra_dec_map=True)`**.


