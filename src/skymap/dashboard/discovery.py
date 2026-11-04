"""
File discovery for the dashboard.

Expected layout (one folder per observing session under the data root)::

    data_310/
        TGBT25B_614_01/
            310/          cal_file_*.hdf5, GBTOBS_*.hdf5 (+ derived cuts)
            Antenna/      YYYY_MM_DD_HH:MM:SS.fits  (pointing)
            ...

Folder names only need to look like ``<PROJECT>_<NNN>_<NN>`` (e.g. ``TGBT25B_614_02``).
Sub-folder names are matched case-insensitively (``Antenna`` / ``ANTENNA``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py

from skymap.io import CAL_POL_NAMES, _list_pointing_files_with_times



def _project_root() -> Path:
    """Folder holding ``data_310/`` (and ``outputs/``).

    The current directory if it has a ``data_310`` folder, otherwise the source checkout
    (``<repo>/src/skymap/dashboard/discovery.py`` -> ``<repo>``), otherwise the current directory.
    """
    cwd = Path.cwd()
    if (cwd / "data_310").is_dir():
        return cwd
    repo = Path(__file__).resolve().parents[3]
    return repo if (repo / "data_310").is_dir() else cwd


DATA_ROOT_ENV = "SKYMAP_DATA_ROOT"
REPO_ROOT = _project_root()
DEFAULT_DATA_ROOT = Path(os.environ[DATA_ROOT_ENV]).expanduser() if os.environ.get(DATA_ROOT_ENV) else REPO_ROOT / "data_310"
DEFAULT_SESSION = "TGBT25B_614_01"

SESSION_RE = re.compile(r"^[A-Za-z0-9]+_\d+_\d+$")

ALL_LABEL = "ALL"
# Display label ("AA*") -> attribute name used throughout skymap ("AA_")
POL_LABELS: dict[str, str] = {
    name.replace("_", "*"): name for name in CAL_POL_NAMES
}
CORRELATION_CHOICES: list[str] = list(POL_LABELS) + [ALL_LABEL]


def pol_attributes(choice: str) -> list[str]:
    """Map a correlation dropdown label to the list of ``CAL_POL_NAMES`` it selects."""
    if choice == ALL_LABEL:
        return list(CAL_POL_NAMES)
    if choice in POL_LABELS:
        return [POL_LABELS[choice]]
    if choice in CAL_POL_NAMES:
        return [choice]
    raise ValueError(f"Unknown correlation {choice!r}; choose from {CORRELATION_CHOICES}")


def pol_label(attr: str) -> str:
    """``'AB_'`` -> ``'AB*'``."""
    return attr.replace("_", "*")


def _child_ci(parent: Path, name: str) -> Path | None:
    """Return the child of ``parent`` whose name equals ``name`` case-insensitively."""
    if not parent.is_dir():
        return None
    for child in sorted(parent.iterdir()):
        if child.name.lower() == name.lower():
            return child
    return None


def find_sessions(root: str | Path) -> list[Path]:
    """Session folders under ``root`` whose names match ``SESSION_RE``, sorted by name."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and SESSION_RE.match(p.name))


def _hdf5_in_session(session: Path) -> list[Path]:
    """All ``*.hdf5``/``*.h5`` files in the session folder and its direct sub-folders."""
    found: list[Path] = []
    dirs = [session] + sorted(p for p in session.iterdir() if p.is_dir())
    for d in dirs:
        for pattern in ("*.hdf5", "*.h5"):
            found.extend(sorted(d.glob(pattern)))
    return [p for p in found if not p.name.startswith(".")]


def classify_hdf5(path: str | Path) -> str:
    """
    Classify an HDF5 file as ``"cal"``, ``"raw"``, ``"derived"`` or ``"other"``.

    * ``raw``: spectrometer output (``GBTOBS_*``).
    * ``cal``: contains top-level ``gain`` and ``te`` datasets.
    * ``derived``: contains ``data/spec`` but is not a raw ``GBTOBS`` file
      (e.g. ``beam_obs_*`` and ``*_source_cut_*`` slices written by the notebooks).
    """
    path = Path(path)
    if path.name.upper().startswith("GBTOBS"):
        return "raw"
    try:
        with h5py.File(path, "r") as f:
            if "gain" in f and "te" in f:
                return "cal"
            if "data" in f and ("spec" in f["data"] or "calibrated_spec" in f["data"]):
                return "derived"
    except OSError:
        return "other"
    return "other"


def find_cal_files(session: str | Path) -> list[Path]:
    """Calibration HDF5 files in a session (top-level ``gain``/``te`` datasets)."""
    session = Path(session)
    files = [p for p in _hdf5_in_session(session) if classify_hdf5(p) == "cal"]
    return sorted(files, key=lambda p: p.name)


def find_data_files(session: str | Path, include_derived: bool = False) -> list[Path]:
    """Spectrometer HDF5 files (``GBTOBS_*``); optionally the derived cuts as well."""
    session = Path(session)
    kinds = {"raw", "derived"} if include_derived else {"raw"}
    files = [p for p in _hdf5_in_session(session) if classify_hdf5(p) in kinds]
    return sorted(files, key=lambda p: (classify_hdf5(p) != "raw", p.name))


def find_pointing_dirs(session: str | Path) -> list[Path]:
    """Pointing directories (``Antenna`` / ``ANTENNA``) that contain pointing FITS files."""
    session = Path(session)
    out: list[Path] = []
    antenna = _child_ci(session, "antenna")
    if antenna is not None:
        out.append(antenna)
    return out


@dataclass(frozen=True)
class PointingCoverage:
    n_files: int
    first: datetime | None
    last: datetime | None


def pointing_coverage(pointing_dir: str | Path) -> PointingCoverage:
    """Number of pointing files and the first/last filename start time (naive UTC)."""
    items = _list_pointing_files_with_times(pointing_dir)
    if not items:
        return PointingCoverage(0, None, None)
    return PointingCoverage(len(items), items[0][0], items[-1][0])
