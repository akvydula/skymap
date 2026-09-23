"""
CASA image-domain deconvolution for beam-gridded maps.

Uses ``casatasks.imaging.deconvolve`` on CASA images built from the output of
``Beam.grid_beam_obs_pointing`` (``.residual`` + ``.psf``, optional ``.mask``).

Requires modular CASA 6+ packages: ``casatasks``, ``casatools`` (auto-installed
with casatasks), and ``casadata``. Install with
``pip install --extra-index-url https://go.nrao.edu/pypi -e ".[casa]"``.
``casacore`` / ``python-casacore`` is not required separately.
"""

from __future__ import annotations

import glob
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skymap.Beam import (
    _apodize_gridded_map,
    _fwhm_deg_from_beam_params,
    _gaussian_psf_2d_on_grid,
    _interp_gridded_map_at_pointings,
    _sigma_deg_from_beam_params,
    grid_beam_obs_pointing,
    plot_gridded_map,
)

_DEFAULT_FREQ_HZ = 310.0e6


def _require_casa() -> tuple[Any, Any, Any]:
    try:
        from casatasks import deconvolve as casa_deconvolve
        from casatools import coordsys as casa_coordsys
        from casatools import image as casa_image
    except ImportError as exc:
        raise ImportError(
            "CASA deconvolution requires casatasks, casatools, and casadata. "
            "Install with: pip install --extra-index-url https://go.nrao.edu/pypi "
            "-e \".[casa]\""
        ) from exc
    return casa_deconvolve, casa_image, casa_coordsys


def _to_casa_pixels(map2d: np.ndarray) -> np.ndarray:
    """Skymap (n_dec, n_ra) -> CASA [ra, dec, pol, chan]."""
    arr = np.asarray(map2d, dtype=np.float64)
    return arr.T[:, :, np.newaxis, np.newaxis]


def _from_casa_pixels(pixels: np.ndarray) -> np.ndarray:
    """CASA [ra, dec, pol, chan] -> skymap (n_dec, n_ra)."""
    cube = np.asarray(pixels, dtype=np.float64)
    if cube.ndim == 2:
        return cube.T
    if cube.ndim >= 4:
        return cube[:, :, 0, 0].T
    raise ValueError(f"Expected CASA image with >=2 dimensions, got shape {cube.shape}")


def _build_casa_coordsys_record(
    coordsys_mod: Any,
    image_mod: Any,
    *,
    ra_centers: np.ndarray,
    dec_centers: np.ndarray,
    pixel_size_deg: float,
    freq_hz: float = _DEFAULT_FREQ_HZ,
) -> dict[str, Any]:
    """Direction + Stokes + spectral coords for a 4D CASA cube."""
    from casatools import quanta

    n_ra = len(ra_centers)
    n_dec = len(dec_centers)
    if n_ra < 1 or n_dec < 1:
        raise ValueError("Grid must have at least one RA and Dec pixel")

    pix = float(pixel_size_deg)
    ra_ref = float(ra_centers[n_ra // 2])
    dec_ref = float(dec_centers[n_dec // 2])
    freq_mhz = float(freq_hz) / 1.0e6
    qa = quanta()

    ia = image_mod()
    ia.fromshape(outfile="", shape=[n_ra, n_dec, 1, 1], type="f")
    cs = ia.coordsys()
    cs.setreferencepixel([n_ra / 2.0, n_dec / 2.0, 1.0, 1.0])
    cs.setreferencevalue(
        [qa.quantity(f"{ra_ref}deg"), qa.quantity(f"{dec_ref}deg")],
        type="direction",
    )
    cs.setreferencevalue(f"{freq_mhz}MHz", type="spectral")
    cs.setincrement(
        [
            qa.quantity(f"{pix}deg"),
            qa.quantity(f"{pix}deg"),
            1.0,
            qa.quantity("1Hz"),
        ]
    )
    record = cs.torecord()
    cs.done()
    ia.done()
    return record


def _write_casa_image(
    image_mod: Any,
    path: str | Path,
    map2d: np.ndarray,
    csys_record: dict[str, Any],
    *,
    overwrite: bool = True,
) -> None:
    path = Path(path)
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(path)
    pixels = _to_casa_pixels(map2d).astype(np.float32)
    ia = image_mod()
    ia.fromarray(
        outfile=str(path),
        pixels=pixels,
        csys=csys_record,
        overwrite=True,
        type="f",
    )
    ia.done()


def _read_casa_image(image_mod: Any, path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    ia = image_mod()
    ia.open(str(path))
    pixels = ia.getchunk()
    ia.close()
    ia.done()
    return _from_casa_pixels(pixels)


def _remove_casa_image_tree(path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _cleanup_imagename_prefix(prefix: str | Path) -> None:
    prefix = str(prefix)
    for match in glob.glob(f"{prefix}.*"):
        _remove_casa_image_tree(match)


def write_casa_deconvolve_images(
    grid_result: dict[str, Any],
    beam_params: dict[str, Any],
    imagename: str | Path,
    *,
    attribute: str = "AB_",
    freq_hz: float = _DEFAULT_FREQ_HZ,
    pad_value: float | str = "edge",
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = None,
    taper_fwhm_deg: float | None = None,
    taper_kind: str = "cosine",
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    Write ``.residual``, ``.psf``, and ``.mask`` CASA images from a gridded map.

    The residual is the apodized observed map (NaNs filled). The PSF is a
    Gaussian normalized to peak 1. The mask is 1 where kernel weight sum > 0.
    """
    _, image_mod, coordsys_mod = _require_casa()
    if attribute not in grid_result["maps"]:
        raise KeyError(
            f"Attribute {attribute!r} not in grid maps. "
            f"Available: {list(grid_result['maps'].keys())}"
        )

    imagename = Path(imagename)
    imagename.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(imagename)
    if overwrite:
        _cleanup_imagename_prefix(prefix)

    observed = grid_result["maps"][attribute]
    weight = grid_result["weight_sum"][attribute]
    pixel_size_deg = float(grid_result["pixel_size_deg"])
    sigma_deg = _sigma_deg_from_beam_params(beam_params)

    residual, _, _, fill_value = _apodize_gridded_map(
        observed,
        weight,
        pad_value,
        pixel_size_deg=pixel_size_deg,
        taper_width_deg=taper_width_deg,
        apodize_fwhm_deg=apodize_fwhm_deg,
        taper_fwhm_deg=taper_fwhm_deg,
        taper_kind=taper_kind,
    )
    residual = np.where(np.isfinite(residual), residual, 0.0)

    psf = _gaussian_psf_2d_on_grid(
        observed.shape,
        pixel_size_deg,
        grid_result["dec_centers"],
        sigma_deg,
    )
    psf_peak = float(np.max(psf))
    if not np.isfinite(psf_peak) or psf_peak <= 0:
        raise ValueError("PSF peak must be positive and finite")
    psf = psf / psf_peak

    mask = np.where(weight > 0, 1.0, 0.0)

    csys = _build_casa_coordsys_record(
        coordsys_mod,
        image_mod,
        ra_centers=grid_result["ra_centers"],
        dec_centers=grid_result["dec_centers"],
        pixel_size_deg=pixel_size_deg,
        freq_hz=freq_hz,
    )

    residual_path = f"{prefix}.residual"
    psf_path = f"{prefix}.psf"
    mask_path = f"{prefix}.mask"
    _write_casa_image(image_mod, residual_path, residual, csys, overwrite=overwrite)
    _write_casa_image(image_mod, psf_path, psf, csys, overwrite=overwrite)
    _write_casa_image(image_mod, mask_path, mask, csys, overwrite=overwrite)

    return {
        "imagename": prefix,
        "residual_path": residual_path,
        "psf_path": psf_path,
        "mask_path": mask_path,
        "observed_map": np.asarray(observed, dtype=float),
        "residual_map": residual,
        "psf": psf,
        "mask": mask,
        "fill_value": float(fill_value),
        "csys_record": csys,
    }


def run_casa_deconvolve(
    imagename: str | Path,
    *,
    deconvolver: str = "hogbom",
    niter: int = 100,
    gain: float = 0.1,
    threshold: float | str = 0.0,
    nsigma: float = 0.0,
    scales: list[int] | None = None,
    restoration: bool = True,
    restoringbeam: str = "",
    usemask: str = "user",
    mask: str | None = None,
    interactive: bool = False,
    fullsummary: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call ``casatasks.imaging.deconvolve`` on an existing image prefix."""
    casa_deconvolve, _, _ = _require_casa()
    prefix = str(imagename)
    default_mask = f"{prefix}.mask"

    call_kw: dict[str, Any] = {
        "imagename": prefix,
        "deconvolver": deconvolver,
        "niter": int(niter),
        "gain": float(gain),
        "threshold": threshold,
        "nsigma": float(nsigma),
        "restoration": bool(restoration),
        "restoringbeam": restoringbeam,
        "usemask": usemask,
        "interactive": interactive,
        "fullsummary": fullsummary,
        **kwargs,
    }
    if scales is not None:
        call_kw["scales"] = scales
    if mask is not None:
        call_kw["mask"] = mask
    elif usemask == "user" and Path(default_mask).exists():
        # CASA auto-loads imagename.mask; passing the same path raises an error.
        call_kw["mask"] = ""

    return casa_deconvolve(**call_kw)


def deconvolve_gridded_map(
    grid_result: dict[str, Any],
    beam_params: dict[str, Any],
    *,
    attribute: str = "AB_",
    imagename: str | Path | None = None,
    workdir: str | Path | None = None,
    keep_workdir: bool = False,
    ra_deg: np.ndarray | None = None,
    dec_deg: np.ndarray | None = None,
    freq_hz: float = _DEFAULT_FREQ_HZ,
    pad_value: float | str = "edge",
    taper_width_deg: float | None = None,
    apodize_fwhm_deg: float | None = None,
    taper_fwhm_deg: float | None = None,
    taper_kind: str = "cosine",
    deconvolver: str = "hogbom",
    niter: int = 100,
    gain: float = 0.1,
    threshold: float | str = 0.0,
    nsigma: float = 0.0,
    scales: list[int] | None = None,
    restoration: bool = True,
    restoringbeam: str = "",
    usemask: str = "user",
    interactive: bool = False,
    fullsummary: bool = False,
    deconvolve_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    CASA minor-cycle deconvolution of a beam-weighted gridded map.

    Parameters
    ----------
    grid_result : dict
        Output of ``grid_beam_obs_pointing``.
    beam_params : dict
        Fitted beam parameters (``FWHM_deg`` / ``sigma_deg``).
    attribute : str
        Polarization channel to deconvolve.
    imagename, workdir
        CASA image prefix. If ``imagename`` is None, a temp directory is used.
    keep_workdir
        If False and a temp workdir was created, remove CASA images after readback.
    ra_deg, dec_deg
        Optional pointings for interpolating the restored map.
    deconvolver, niter, gain, threshold, nsigma, scales, restoration, restoringbeam
        Passed to ``casatasks.deconvolve``.
    pad_value, taper_width_deg, apodize_fwhm_deg, taper_fwhm_deg, taper_kind
        Apodization applied before writing the ``.residual`` image.

    Returns
    -------
    dict
        ``grid``, ``observed_map``, ``deconvolved_map`` (restored ``.image``),
        ``model_map``, ``residual_map``, ``psf``, ``casa_summary``, ``method``
        (``\"casa_deconvolve\"``), plus optional ``deconvolved`` samples.
    """
    _, image_mod, _ = _require_casa()
    fwhm_deg = _fwhm_deg_from_beam_params(beam_params)
    pixel_size_deg = float(grid_result["pixel_size_deg"])

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if imagename is None:
        if workdir is not None:
            base = Path(workdir)
            base.mkdir(parents=True, exist_ok=True)
            imagename = base / "skymap_dec"
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="skymap_deconv_")
            imagename = Path(temp_dir.name) / "skymap_dec"
    elif workdir is not None:
        imagename = Path(workdir) / Path(imagename).name

    prefix = str(imagename)
    if restoringbeam == "" and restoration:
        restoringbeam = f"{fwhm_deg}deg"

    if scales is None and deconvolver in {"multiscale", "mtmfs"}:
        beam_pix = max(1, int(round(fwhm_deg / pixel_size_deg)))
        scales = [0, beam_pix, 3 * beam_pix]

    prep = write_casa_deconvolve_images(
        grid_result,
        beam_params,
        prefix,
        attribute=attribute,
        freq_hz=freq_hz,
        pad_value=pad_value,
        taper_width_deg=taper_width_deg,
        apodize_fwhm_deg=apodize_fwhm_deg,
        taper_fwhm_deg=taper_fwhm_deg,
        taper_kind=taper_kind,
    )

    dec_kw = dict(deconvolve_kwargs or {})
    casa_summary = run_casa_deconvolve(
        prefix,
        deconvolver=deconvolver,
        niter=niter,
        gain=gain,
        threshold=threshold,
        nsigma=nsigma,
        scales=scales,
        restoration=restoration,
        restoringbeam=restoringbeam,
        usemask=usemask,
        interactive=interactive,
        fullsummary=fullsummary,
        **dec_kw,
    )

    image_path = f"{prefix}.image"
    model_path = f"{prefix}.model"
    residual_path = f"{prefix}.residual"

    restored = (
        _read_casa_image(image_mod, image_path)
        if Path(image_path).exists()
        else np.full_like(prep["residual_map"], np.nan)
    )
    model = (
        _read_casa_image(image_mod, model_path)
        if Path(model_path).exists()
        else np.zeros_like(prep["residual_map"])
    )
    residual_out = (
        _read_casa_image(image_mod, residual_path)
        if Path(residual_path).exists()
        else prep["residual_map"].copy()
    )

    weight = grid_result["weight_sum"][attribute]
    for arr in (restored, model, residual_out):
        arr[weight <= 0] = np.nan

    out: dict[str, Any] = {
        "grid": grid_result,
        "observed_map": prep["observed_map"],
        "deconvolved_map": restored,
        "model_map": model,
        "residual_map": residual_out,
        "psf": prep["psf"],
        "psf_fwhm_deg": float(fwhm_deg),
        "fill_value": prep["fill_value"],
        "method": "casa_deconvolve",
        "casa_summary": casa_summary,
        "imagename": prefix,
        "deconvolver": deconvolver,
        "niter": int(niter),
        "deconvolved": {},
        "noise_spectra": {},
        "healpix_map": None,
    }

    if ra_deg is not None and dec_deg is not None:
        ra_a = np.atleast_1d(np.asarray(ra_deg, dtype=float))
        dec_a = np.atleast_1d(np.asarray(dec_deg, dtype=float))
        out["ra"] = ra_a.copy()
        out["dec"] = dec_a.copy()
        out["deconvolved"][attribute] = _interp_gridded_map_at_pointings(
            restored,
            grid_result["ra_centers"],
            grid_result["dec_centers"],
            ra_a,
            dec_a,
        )

    if not keep_workdir:
        _cleanup_imagename_prefix(prefix)
        if temp_dir is not None:
            temp_dir.cleanup()

    return out


def grid_and_deconvolve(
    beam_obs_pointing: object,
    beam_params: dict[str, Any],
    *,
    attribute: str | None = None,
    attributes: list[str] | None = None,
    freq_index: int | None = None,
    pixel_size_deg: float | None = None,
    kernel_sigma_deg: float | None = None,
    trunc_sigma: float = 3.0,
    padding_pixels: int = 2,
    deconvolve_attribute: str = "AB_",
    **deconvolve_kwargs: Any,
) -> dict[str, Any]:
    """
    Grid matched pointing data, then run CASA deconvolution on one channel.

    Gridding kwargs are forwarded to ``grid_beam_obs_pointing``; remaining
    kwargs go to ``deconvolve_gridded_map``.
    """
    grid = grid_beam_obs_pointing(
        beam_obs_pointing,
        beam_params,
        attribute=attribute,
        attributes=attributes,
        freq_index=freq_index,
        pixel_size_deg=pixel_size_deg,
        kernel_sigma_deg=kernel_sigma_deg,
        trunc_sigma=trunc_sigma,
        padding_pixels=padding_pixels,
    )
    dec = deconvolve_gridded_map(
        grid,
        beam_params,
        attribute=deconvolve_attribute,
        **deconvolve_kwargs,
    )
    dec["grid"] = grid
    return dec


def plot_deconvolution(
    result: dict[str, Any],
    *,
    attribute: str = "AB_",
    show_psf: bool = True,
    cmap: str = "viridis",
    show: bool = True,
) -> Any:
    """
    Compare observed, CASA restored, model, and residual maps.

    Parameters
    ----------
    result : dict
        Output of ``deconvolve_gridded_map``.
    """
    grid = result["grid"]
    ra_edges = grid["ra_edges"]
    dec_edges = grid["dec_edges"]
    panels: list[tuple[str, np.ndarray]] = [
        ("Observed", result["observed_map"]),
        ("Restored (CASA .image)", result["deconvolved_map"]),
        ("Model (CASA .model)", result["model_map"]),
        ("Residual (CASA .residual)", result["residual_map"]),
    ]
    if show_psf:
        panels.append(("PSF", result["psf"]))

    ncols = min(3, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, (title, z) in zip(axes_flat, panels):
        pc = ax.pcolormesh(ra_edges, dec_edges, z, shading="flat", cmap=cmap)
        ax.set_aspect("equal")
        ax.set_xlabel("RA (deg)")
        ax.set_ylabel("Dec (deg)")
        ax.set_title(title)
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(f"CASA deconvolution ({result.get('deconvolver', 'hogbom')}, "
                 f"niter={result.get('niter', '?')})", y=1.02)
    if show:
        plt.tight_layout()
        plt.show()
    return axes_flat


def plot_gridded_and_deconvolved(
    grid_result: dict[str, Any],
    result: dict[str, Any],
    attribute: str = "AB_",
    *,
    overlay_pointings: tuple[np.ndarray, np.ndarray] | None = None,
    show: bool = True,
) -> Any:
    """Side-by-side gridded map and CASA restored image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_gridded_map(
        grid_result,
        attribute,
        ax=axes[0],
        show_coverage=False,
        overlay_pointings=overlay_pointings,
        show=False,
    )
    axes[0].set_title("Gridded (observed)")

    z = result["deconvolved_map"]
    pc = axes[1].pcolormesh(
        grid_result["ra_edges"],
        grid_result["dec_edges"],
        z,
        shading="flat",
        cmap="viridis",
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("RA (deg)")
    axes[1].set_ylabel("Dec (deg)")
    axes[1].set_title("CASA restored")
    if overlay_pointings is not None:
        ra_ov, dec_ov = overlay_pointings
        axes[1].scatter(
            ra_ov, dec_ov, s=10, facecolors="none", edgecolors="crimson", linewidths=0.6
        )
    fig.colorbar(pc, ax=axes[1], label=f"{attribute} (K)")
    if show:
        plt.tight_layout()
        plt.show()
    return axes
