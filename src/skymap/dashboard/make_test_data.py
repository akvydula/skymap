"""Create a synthetic "new" session for exercising the dashboard.

The new session is built from an existing one by transforming the spectra,

    y = scale * x + c          (default scale = 1.2)

where ``x`` is ``data/spec`` of every raw ``GBTOBS_*.hdf5`` file. Everything else
(timestamps, frequencies, diagnostics, header) is copied unchanged, the
calibration file and the pointing directory are copied alongside, so the new
session has exactly the layout the dashboard discovers (``<session>/310/`` and
``<session>/Antenna/``).

The raw spectra are of order 1e-8 and differ by orders of magnitude between
correlations, so a single absolute ``c`` would swamp some of them. By default
``c`` is therefore ``offset_frac * median|Re(x)|`` computed per correlation, and
added to the real part only. Use ``--offset`` for a fixed absolute ``c`` instead.

Usage::

    python -m skymap.dashboard.make_test_data                       # -> data_310/TGBT25B_614_99
    python -m skymap.dashboard.make_test_data --scale 1.5 --offset-frac 0.2
    python -m skymap.dashboard.make_test_data --files GBTOBS_2026-01-16T112750.hdf5

The output directory is git-ignored (``data_310/``). Re-running refuses to
overwrite an existing session unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

from skymap.dashboard.discovery import DEFAULT_DATA_ROOT, DEFAULT_SESSION, SESSION_RE, classify_hdf5

DEFAULT_OUT_SESSION = "TGBT25B_614_99"
CHUNK_ROWS = 2000


def per_pol_offsets(
    h5: h5py.File,
    offset_frac: float,
    offset: float | None,
    sample_rows: int = 40,
) -> np.ndarray:
    """Return the offset ``c`` for each correlation (length ``n_pol``)."""
    n_time, n_pol, _ = h5["data/spec"].shape
    if offset is not None:
        return np.full(n_pol, float(offset))
    idx = np.unique(np.linspace(0, n_time - 1, min(sample_rows, n_time)).astype(int))
    sample = h5["data/spec"][idx]  # (rows, n_pol, n_freq)
    ref = np.median(np.abs(sample.real), axis=(0, 2))
    return offset_frac * ref


def transform_spec_inplace(path: Path, scale: float, c: np.ndarray, chunk_rows: int = CHUNK_ROWS) -> None:
    """Rewrite ``data/spec`` as ``scale * x + c`` (c added to the real part)."""
    with h5py.File(path, "r+") as h5:
        spec = h5["data/spec"]
        n_time = spec.shape[0]
        c_c = c.astype(spec.dtype.type(0).real.dtype)[None, :, None]
        for i0 in range(0, n_time, chunk_rows):
            i1 = min(i0 + chunk_rows, n_time)
            block = spec[i0:i1]
            out = (scale * block + c_c).astype(spec.dtype)
            spec[i0:i1] = out
        spec.attrs["synthetic_scale"] = float(scale)
        spec.attrs["synthetic_offset"] = c.astype("float64")
        h5.attrs["synthetic"] = "y = scale * x + c on data/spec; made by skymap.dashboard.make_test_data"


def make_session(
    src_session: Path,
    dst_session: Path,
    *,
    scale: float = 1.2,
    offset_frac: float = 0.1,
    offset: float | None = None,
    files: list[str] | None = None,
    force: bool = False,
    log=print,
) -> list[Path]:
    """Build ``dst_session`` from ``src_session``; return the new raw files."""
    if dst_session.exists():
        if not force:
            raise FileExistsError(f"{dst_session} already exists (use --force to replace)")
        shutil.rmtree(dst_session)
    src_310 = src_session / "310"
    if not src_310.is_dir():
        raise FileNotFoundError(f"{src_310} not found")

    raw = sorted(p for p in src_310.glob("GBTOBS_*.hdf5"))
    if files:
        wanted = set(files)
        raw = [p for p in raw if p.name in wanted]
        missing = wanted - {p.name for p in raw}
        if missing:
            raise FileNotFoundError(f"not in {src_310}: {sorted(missing)}")
    if not raw:
        raise FileNotFoundError(f"no GBTOBS_*.hdf5 in {src_310}")
    cals = [p for p in sorted(src_310.glob("*.hdf5")) if classify_hdf5(p) == "cal"]
    pointing = next(
        (d for d in src_session.iterdir() if d.is_dir() and d.name.lower() == "antenna"), None
    )
    if pointing is None:
        raise FileNotFoundError(f"no Antenna directory in {src_session}")

    dst_310 = dst_session / "310"
    dst_310.mkdir(parents=True)

    log(f"copying pointing directory {pointing.name}/")
    shutil.copytree(pointing, dst_session / pointing.name)
    for cal in cals:
        log(f"copying calibration {cal.name}")
        shutil.copy2(cal, dst_310 / cal.name)

    out: list[Path] = []
    c: np.ndarray | None = None  # fixed from the first file so c is constant per session
    for p in raw:
        dst = dst_310 / p.name
        log(f"{p.name}: copying ({p.stat().st_size / 1e9:.2f} GB)")
        shutil.copyfile(p, dst)
        if c is None:
            with h5py.File(dst, "r") as h5:
                c = per_pol_offsets(h5, offset_frac, offset)
        log(f"{p.name}: y = {scale} * x + c, c per pol = {np.array2string(c, precision=3)}")
        transform_spec_inplace(dst, scale, c)
        out.append(dst)

    (dst_session / "SYNTHETIC.txt").write_text(
        f"Synthetic test session derived from {src_session.name}.\n"
        f"data/spec transformed as y = {scale} * x + c.\n"
        "c per correlation (AA*..AD*): " + ", ".join(f"{v:.4g}" for v in c) + "\n"
        "Delete this folder when real data arrives.\n"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--source", default=DEFAULT_SESSION, help="existing session to scale")
    ap.add_argument("--session", default=DEFAULT_OUT_SESSION, help="name of the new session")
    ap.add_argument("--scale", type=float, default=1.2)
    ap.add_argument("--offset-frac", type=float, default=0.1,
                    help="c = frac * median|Re x| per correlation (default 0.1)")
    ap.add_argument("--offset", type=float, default=None, help="absolute c for every correlation")
    ap.add_argument("--files", nargs="*", help="only these GBTOBS_*.hdf5 names (default: all)")
    ap.add_argument("--force", action="store_true", help="replace an existing session")
    args = ap.parse_args(argv)

    if not SESSION_RE.match(args.session):
        ap.error(f"session name must look like TGBT25B_614_02, got {args.session!r}")
    if args.session == args.source:
        ap.error("new session must differ from the source session")

    out = make_session(
        args.data_root / args.source,
        args.data_root / args.session,
        scale=args.scale,
        offset_frac=args.offset_frac,
        offset=args.offset,
        files=args.files,
        force=args.force,
    )
    print(f"\nCreated {args.data_root / args.session} with {len(out)} raw file(s).")
    print("Select it in the dashboard sidebar (Session dropdown).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
