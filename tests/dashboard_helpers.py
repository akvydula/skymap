"""Synthetic observation files shared by the dashboard tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from skymap.io import CAL_POL_NAMES, CalData, PolChannels

N_FREQ = 8
START_UNIX_MS = 1768580869726.0  # 2026-01-16 16:27:49.726 UTC
CADENCE_MS = 100.0


def make_raw_file(path: Path, n_time: int = 200, n_freq: int = N_FREQ, seed: int = 0) -> Path:
    """Raw-spectrometer-style file: unix-ms time (no ``time_unit`` attr), complex64 ``spec``."""
    rng = np.random.default_rng(seed)
    spec = (rng.normal(size=(n_time, 10, n_freq)) + 1j * rng.normal(size=(n_time, 10, n_freq))).astype(
        np.complex64
    )
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("freq", data=np.linspace(300e6, 320e6, n_freq)).attrs["units"] = "Hz"
        grp.create_dataset("time", data=START_UNIX_MS + CADENCE_MS * np.arange(n_time)).attrs[
            "units"
        ] = "UNIX TIME"
        ds = grp.create_dataset("spec", data=spec, chunks=(16, 2, n_freq))
        ds.attrs["labels"] = np.array([n.replace("_", "*") for n in CAL_POL_NAMES], dtype=object)
        diag = f.create_group("diag")
        diag.create_dataset("time", data=START_UNIX_MS + CADENCE_MS * np.arange(n_time))
    return path


def make_cal_file(path: Path, n_freq: int = N_FREQ) -> Path:
    with h5py.File(path, "w") as f:
        f.create_dataset("gain", data=np.full((10, n_freq), 1e9, dtype=np.float32))
        f.create_dataset("te", data=np.full((10, n_freq), 30.0, dtype=np.float32))
    return path


def make_cal_data(n_freq: int = N_FREQ) -> CalData:
    ones = np.ones(n_freq)
    gain = PolChannels(**{n: 1e9 * ones for n in CAL_POL_NAMES})
    te = PolChannels(**{n: 30.0 * ones for n in CAL_POL_NAMES})
    return CalData(freq=np.linspace(300e6, 320e6, n_freq), te=te, gain=gain)


def make_session(root: Path, name: str = "TGBT25B_614_02", antenna_dir: str = "ANTENNA") -> Path:
    """Session tree with cal, raw, derived HDF5 files and empty pointing FITS placeholders."""
    session = root / name
    (session / "310").mkdir(parents=True)
    (session / antenna_dir).mkdir()
    make_cal_file(session / "310" / "cal_file_2026_01_01.hdf5")
    make_raw_file(session / "310" / "GBTOBS_2026-01-16T112750.hdf5")
    make_raw_file(session / "310" / "beam_obs_2026_01_16.hdf5")  # derived-style name
    for stamp in ("2026_01_16_18:04:07", "2026_01_16_18:05:14"):
        (session / antenna_dir / f"{stamp}.fits").touch()
    (session / "310" / ".hidden.hdf5").touch()
    return session
