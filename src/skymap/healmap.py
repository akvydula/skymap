'''
Heal pix map module for 310 MHz observations
'''

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.text as mpltext
import numpy as np

from skymap.io import CAL_POL_NAMES

if TYPE_CHECKING:
    from skymap.io import PointingData

# Full sky solid angle in steradians; 4*pi*(180/pi)**2 gives full sky in sq.deg
FULL_SKY_STERADIANS = 4.0 * np.pi
FULL_SKY_SQ_DEG = FULL_SKY_STERADIANS * (180.0 / np.pi) ** 2
# Beam size: 1 sq.deg -> radius in deg (area = pi*r^2 => r = sqrt(1/pi))
BEAM_AREA_SQ_DEG = 1.0
BEAM_RADIUS_DEG = np.sqrt(BEAM_AREA_SQ_DEG / np.pi)


def _beam_circle_thetaphi_rad(
    center_ra_deg: float, center_dec_deg: float, npoints: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta, phi) in radians for a circle of angular radius BEAM_RADIUS_DEG.

    Exact calculation:
    1. Beam area = BEAM_AREA_SQ_DEG (1 sq.deg) -> radius = sqrt(area/pi) = BEAM_RADIUS_DEG deg.
    2. radius_rad = np.radians(BEAM_RADIUS_DEG) so angular radius is exactly BEAM_RADIUS_DEG.
    3. Center in healpy (theta, phi): theta = colatitude = pi/2 - dec, phi = ra (radians).
    4. Small circle (constant angular distance): theta = theta0 + radius_rad*cos(t),
       phi = phi0 + radius_rad*sin(t)/sin(theta0).
    Returns (theta, phi) in radians for use with projplot(..., lonlat=False).
    """
    radius_rad = np.radians(BEAM_RADIUS_DEG)
    ra_rad = np.radians(center_ra_deg)
    dec_rad = np.radians(center_dec_deg)
    theta0 = np.pi / 2.0 - dec_rad
    phi0 = ra_rad
    t = np.linspace(0, 2 * np.pi, npoints, endpoint=False)
    theta = theta0 + radius_rad * np.cos(t)
    phi = phi0 + radius_rad * np.sin(t) / np.sin(theta0)
    return theta, phi


def _beam_circle_radec_deg(center_ra_deg: float, center_dec_deg: float, npoints: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return (ra, dec) in degrees for a circle of angular radius BEAM_RADIUS_DEG (for RA/Dec axes plot)."""
    t = np.linspace(0, 2 * np.pi, npoints, endpoint=False)
    dec_rad = np.radians(center_dec_deg)
    ra_circle = center_ra_deg + (BEAM_RADIUS_DEG / np.cos(dec_rad)) * np.cos(t)
    dec_circle = center_dec_deg + BEAM_RADIUS_DEG * np.sin(t)
    return ra_circle, dec_circle


def _add_beam_indicator(region: tuple[float, float, float] | None) -> None:
    """Draw a circle showing the beam size (BEAM_AREA_SQ_DEG, radius BEAM_RADIUS_DEG).
    Circle angular size is always BEAM_RADIUS_DEG; position is chosen to lie within the plot region.
    Uses current axes (after mollview/gnomview)."""
    if region is not None:
        center_ra, center_dec, area_sqdeg = region
        extent_deg = np.sqrt(float(area_sqdeg))
        margin_deg = min(0.1, extent_deg / 8)
        inset = BEAM_RADIUS_DEG + margin_deg
        beam_ra = center_ra - extent_deg / 2 + inset
        beam_dec = center_dec - extent_deg / 2 + inset
    else:
        beam_ra, beam_dec = 260.0, -70.0
    theta, phi = _beam_circle_thetaphi_rad(beam_ra, beam_dec)
    try:
        hp.projplot(theta, phi, "k-", lonlat=False, lw=1.5)
    except Exception:
        pass


def _add_beam_indicator_radec(ax, region: tuple[float, float, float]) -> None:
    """Draw beam circle on axes where x=RA, y=Dec (data coordinates)."""
    center_ra, center_dec, area_sqdeg = region
    extent_deg = np.sqrt(float(area_sqdeg))
    margin_deg = min(0.1, extent_deg / 8)
    inset = BEAM_RADIUS_DEG + margin_deg
    beam_ra = center_ra - extent_deg / 2 + inset
    beam_dec = center_dec - extent_deg / 2 + inset
    ra_c, dec_c = _beam_circle_radec_deg(beam_ra, beam_dec)
    ax.plot(ra_c, dec_c, "k-", lw=1.5)


def _set_plot_labels_and_cbar_style() -> None:
    """
    Healpy uses axis('off') so set_xlabel/set_ylabel are hidden; we use text annotations instead."""
    ax = plt.gca()
    # Place "RA" and "Dec" as text (axis is off in healpy projection axes)
    ax.text(0.5, -0.06, "RA", transform=ax.transAxes, ha="center", va="top", fontsize=10)
    ax.text(-0.06, 0.5, "Dec", transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=10)
    fig = plt.gcf()



def _get_pol_source(data: PointingData | object) -> object | None:
    """Return calibrated_spec_mean or spec_mean if present, else None."""
    return getattr(data, "calibrated_spec_mean", None) or getattr(data, "spec_mean", None)


def _get_available_pol_names(data: object) -> list[str]:
    """Return pol channel names that exist and have data on the given object."""
    source = _get_pol_source(data)
    if source is None:
        return []
    return [n for n in CAL_POL_NAMES if hasattr(source, n) and getattr(source, n) is not None]


class HealPixMap:
    """
    HealPix map with pixel geometry derived from full-sky 4*pi and beam size.
    Pixel values are filled from pointing data (RA, Dec) using a 1 sq.deg beam.
    Accepts a pointing-data object (e.g. from match_data_and_pointing) with
    optional PolChannels; if attributes are not specified, all channels are used.
    """

    def __init__(self, nside: int):
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.map = np.zeros(self.npix, dtype=float)
        self._hit_count = np.zeros(self.npix, dtype=float)
        self._map_std = np.full(self.npix, np.nan, dtype=float)
        self._channel_maps: dict[str, np.ndarray] = {}
        self._channel_stds: dict[str, np.ndarray] = {}

    def get_map(self, attribute: str | None = None) -> np.ndarray:
        """Return the map: hit count if attribute is None and no channels, else the requested channel map."""
        if attribute is None:
            return self.map
        if attribute in self._channel_maps:
            return self._channel_maps[attribute]
        raise KeyError(f"No map for attribute {attribute!r}. Available: {list(self._channel_maps.keys())}")

    @property
    def pixel_area_steradians(self) -> float:
        """Solid angle per pixel in steradians (full sky 4*pi / npix)."""
        return FULL_SKY_STERADIANS / self.npix

    @property
    def pixel_area_sqdeg(self) -> float:
        """Pixel area in square degrees."""
        return FULL_SKY_SQ_DEG / self.npix

    @property
    def pixel_radius_deg(self) -> float:
        """Effective radius of each pixel in degrees (assuming circular area)."""
        return np.sqrt(self.pixel_area_sqdeg / np.pi)

    def fill_from_pointing(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        values: np.ndarray | None = None,
    ) -> None:
        """
        Fill the map from pointing data (RA, Dec in degrees).
        Each pointing is treated as a beam of 1 sq.deg. Every pixel whose center
        falls within the beam gets a contribution. Pixel value is the mean of
        all pointing values that cover it (or hit count if values is None).

        Parameters
        ----------
        ra : np.ndarray
            Right ascension in degrees (length n_pointings).
        dec : np.ndarray
            Declination in degrees (length n_pointings).
        values : np.ndarray, optional
            Per-pointing value (length n_pointings). If None, each pointing
            contributes 1 (hit count).
        """
        ra = np.atleast_1d(np.asarray(ra, dtype=float))
        dec = np.atleast_1d(np.asarray(dec, dtype=float))
        if ra.size != dec.size:
            raise ValueError("ra and dec must have the same length")
        if values is not None:
            values = np.atleast_1d(np.asarray(values, dtype=float))
            if values.size != ra.size:
                raise ValueError("values must have the same length as ra/dec")
        else:
            values = np.ones(ra.size, dtype=float)

        beam_radius_rad = np.radians(BEAM_RADIUS_DEG)
        map_sum = np.zeros(self.npix, dtype=float)
        map_count = np.zeros(self.npix, dtype=float)
        map_sum_sq = np.zeros(self.npix, dtype=float)

        for i in range(ra.size):
            # Healpy: theta = colatitude (0 at N pole), phi = longitude [0, 2*pi]
            theta = np.radians(90.0 - dec[i])
            phi = np.radians(ra[i])
            vec = hp.ang2vec(theta, phi)
            pix = hp.query_disc(self.nside, vec, beam_radius_rad)
            v = values[i]
            map_sum[pix] += v
            map_count[pix] += 1
            map_sum_sq[pix] += v * v

        self._hit_count = map_count
        self.map = np.where(map_count > 0, map_sum / map_count, 0.0)
        with np.errstate(invalid="ignore"):
            self._map_std = np.where(
                map_count > 0,
                np.sqrt(np.maximum(0, map_sum_sq / map_count - (map_sum / map_count) ** 2)),
                np.nan,
            )

    def fill_from_pointing_data(
        self,
        pointing_data: PointingData | object,
        attributes: list[str] | None = None,
        freq_index: int | None = None,
    ) -> None:
        """
        Fill the map from a pointing-data object (e.g. PointingData or result of match_data_and_pointing).

        The object must have .ra and .dec (degrees). If it also has .calibrated_spec_mean or
        .spec_mean (CalibratedSpec with PolChannels AA_, BB_, ...), then per-channel maps are
        computed. If attributes is None, all available PolChannels are used; otherwise only
        the listed attributes (e.g. ['AA_', 'BB_']) are used.

        Parameters
        ----------
        pointing_data : object with .ra, .dec and optionally .calibrated_spec_mean / .spec_mean
            e.g. PointingData (ra/dec only) or HDF5Data from match_data_and_pointing.
        attributes : list of str or None, optional
            PolChannel names to map (e.g. ['AA_', 'BB_']). If None and pointing_data has
            calibrated_spec_mean/spec_mean, all available channels are used. If None and
            no pol data, a single hit-count map is produced.
        freq_index : int or None, optional
            If per-channel values are (n_pointing, n_freq), use this frequency index for
            the map. If None, the mean over frequency is used.
        """
        ra = np.atleast_1d(np.asarray(getattr(pointing_data, "ra"), dtype=float))
        dec = np.atleast_1d(np.asarray(getattr(pointing_data, "dec"), dtype=float))
        if ra.size != dec.size:
            raise ValueError("pointing_data must have ra and dec of the same length")

        spec_source = _get_pol_source(pointing_data)
        if spec_source is None:
            self.fill_from_pointing(ra, dec)
            self._channel_maps.clear()
            self._channel_stds.clear()
            return

        available = _get_available_pol_names(pointing_data)
        if not available:
            self.fill_from_pointing(ra, dec)
            self._channel_maps.clear()
            self._channel_stds.clear()
            return

        attrs = attributes if attributes is not None else available
        for name in attrs:
            if name not in available:
                raise ValueError(f"Attribute {name!r} not available on pointing_data. Available: {available}")

        self._channel_maps.clear()
        self._channel_stds.clear()
        for attr in attrs:
            arr = np.asarray(getattr(spec_source, attr))
            if arr.ndim == 2:
                if freq_index is not None:
                    values = arr[:, freq_index]
                else:
                    values = np.nanmean(arr, axis=1)
            else:
                values = arr
            self.fill_from_pointing(ra, dec, values=values)
            self._channel_maps[attr] = self.map.copy()
            self._channel_stds[attr] = self._map_std.copy()

        self.map = self._hit_count.copy()
        self._map_std = np.full(self.npix, np.nan, dtype=float)
        self._map_std[self._hit_count > 0] = 0.0

    def get_hit_count(self) -> np.ndarray:
        """Return the number of pointings that contributed to each pixel."""
        return self._hit_count

    def get_std(self, attribute: str | None = None) -> np.ndarray:
        """Return per-pixel standard deviation (NaN where no hits). If attribute is set, return that channel's std."""
        if attribute is None:
            return self._map_std
        if attribute in self._channel_stds:
            return self._channel_stds[attribute]
        raise KeyError(f"No std for attribute {attribute!r}. Available: {list(self._channel_stds.keys())}")

    def _plot_one(
        self,
        map_array: np.ndarray,
        *,
        title: str | None = "HealPix map",
        unit: str | None = None,
        cmap: str = "viridis",
        mask_uncovered: bool = True,
        region: tuple[float, float, float] | None = None,
        sub: tuple[int, int, int] | None = None,
        show_beam: bool = True,
        ra_dec_map: bool = False,
    ) -> None:
        """Plot a single map: if ra_dec_map=True (and region given) use RA/Dec axes; else regular healpy (mollview or gnomview)."""
        to_plot = np.asarray(map_array, dtype=float).copy()
        to_plot[np.isnan(to_plot)] = hp.UNSEEN
        if mask_uncovered:
            to_plot[to_plot == 0] = hp.UNSEEN

        if ra_dec_map:
            if region is None:
                raise ValueError("region is required when ra_dec_map=True")
            # RA/Dec axes: get pixels in region via query_disc + pix2ang, plot with RA as X, Dec as Y
            if sub is not None:
                nrow, ncol, idx = sub
                plt.subplot(nrow, ncol, idx)
            center_ra, center_dec, area_sqdeg = region
            extent_deg = np.sqrt(float(area_sqdeg))
            radius_rad = np.radians(extent_deg / 2)
            vec = hp.ang2vec(np.radians(90.0 - center_dec), np.radians(center_ra))
            ipix = hp.query_disc(self.nside, vec, radius_rad)
            ra, dec = hp.pix2ang(self.nside, ipix, lonlat=True)
            vals = np.asarray(to_plot[ipix], dtype=float)
            bad = np.isnan(vals)
            if hp.UNSEEN is not None and np.isscalar(hp.UNSEEN):
                bad = bad | (vals == hp.UNSEEN)
            ok = ~bad
            ra, dec, vals = ra[ok], dec[ok], vals[ok]
            ax = plt.gca()
            pixel_deg = np.sqrt(self.pixel_area_sqdeg)
            s = max(1, 4 * (pixel_deg / extent_deg * 400) ** 2)
            sc = ax.scatter(ra, dec, c=vals, cmap=cmap, s=s, edgecolors="none")
            ax.set_xlim(center_ra - extent_deg / 2, center_ra + extent_deg / 2)
            ax.set_ylim(center_dec - extent_deg / 2, center_dec + extent_deg / 2)
            ax.set_xlabel("RA")
            ax.set_ylabel("Dec")
            ax.set_aspect("equal")
            ax.set_title(title or "")
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label(unit or "", fontweight="normal")
            if show_beam:
                _add_beam_indicator_radec(ax, region)
            return

        # Regular healpy plot: mollview (full sky) or gnomview (region)
        kwargs = dict(title=title, unit=unit, cbar=True, cmap=cmap)
        if sub is not None:
            kwargs["sub"] = sub
        if region is None:
            hp.mollview(to_plot, **kwargs)
        else:
            center_ra, center_dec, area_sqdeg = region
            extent_deg = np.sqrt(float(area_sqdeg))
            reso_arcmin = max(0.5, extent_deg * 60 / 256)
            xsize = max(64, int(extent_deg * 60 / reso_arcmin))
            hp.gnomview(
                to_plot,
                rot=(center_ra, center_dec),
                reso=reso_arcmin,
                xsize=xsize,
                **kwargs,
            )
        _set_plot_labels_and_cbar_style()
        if show_beam:
            _add_beam_indicator(region)

    def _cbar_label(self, attribute: str | None, stat: Literal["mean", "std"]) -> str:
        """Build color bar label: '{attribute} {stat} [K]' (e.g. 'AA_ Mean K'), not bold."""
        stat_str = stat.capitalize()
        parts = [attribute, stat_str, "K"] if attribute else [stat_str, "K"]
        return " ".join(parts)

    def plot(
        self,
        *,
        attribute: str | None = None,
        stat: Literal["mean", "std"] = "mean",
        title: str | None = "HealPix map",
        unit: str | None = None,
        cmap: str = "viridis",
        mask_uncovered: bool = True,
        show: bool = True,
        region: tuple[float, float, float] | None = None,
        show_beam: bool = True,
        ra_dec_map: bool = False,
        **kwargs,
    ) -> None:
        """
        Plot the map.

        If ra_dec_map=True (and region given): plot with RA and Dec as X/Y axes using
        healpix pixels in the region (query_disc + pix2ang). Otherwise: regular healpy
        plot (mollview for full sky, gnomview when region is given).

        Parameters
        ----------
        attribute : str or None, optional
            PolChannel name to plot (e.g. 'AA_', 'BB_'). If None and channel maps exist,
            plot all channels in a subplot grid; if None and no channel maps, plot the
            main (hit count) map.
        stat : {'mean', 'std'}, optional
            Plot mean map or standard-deviation map. Default 'mean'.
        title : str or None, optional
            Plot title. If None, no title is shown. For multiple channels, used as
            figure suptitle.
        unit : str or None, optional
            Unused; color bar title is always '{attribute} {stat} K'.
        cmap : str, optional
            Matplotlib colormap name (default "viridis").
        mask_uncovered : bool, optional
            If True (default), pixels with no hits (value 0) are masked as UNSEEN.
        show : bool, optional
            If True (default), call plt.show() after drawing.
        region : tuple (center_ra, center_dec, area_sqdeg) or None, optional
            Patch to plot. Required when ra_dec_map=True. When ra_dec_map=False and
            region is given, gnomview is used; when None, mollview (full sky).
        show_beam : bool, optional
            If True (default), draw the 1 sq.deg beam circle.
        ra_dec_map : bool, optional
            If True, plot with RA as X axis and Dec as Y axis (requires region).
            If False (default), use regular healpy projection (mollview or gnomview).
        **kwargs : dict, optional
            Additional keyword arguments passed to plt.figure().
        """
        if stat == "std":
            def _get_arr(attr: str | None) -> np.ndarray:
                return self.get_std(attr)
        else:
            def _get_arr(attr: str | None) -> np.ndarray:
                return self.get_map(attr)

        if attribute is not None:
            if attribute not in self._channel_maps:
                raise KeyError(f"No map for {attribute!r}. Available: {list(self._channel_maps.keys())}")
            self._plot_one(
                _get_arr(attribute),
                title=title or attribute,
                unit=self._cbar_label(attribute, stat),
                cmap=cmap,
                mask_uncovered=mask_uncovered,
                region=region,
                show_beam=show_beam,
                ra_dec_map=ra_dec_map,
            )
            if show:
                plt.show()
                plt.close()
            return

        if self._channel_maps:
            names = sorted(self._channel_maps.keys())
            n = len(names)
            ncol = min(n, 3)
            nrow = (n + ncol - 1) // ncol
            plt.figure(figsize=(5 * ncol, 4 * nrow), **kwargs)
            if title:
                plt.suptitle(title)
            for i, name in enumerate(names):
                self._plot_one(
                    _get_arr(name),
                    title=name,
                    unit=self._cbar_label(name, stat),
                    cmap=cmap,
                    mask_uncovered=mask_uncovered,
                    region=region,
                    sub=(nrow, ncol, i + 1),
                    show_beam=show_beam,
                    ra_dec_map=ra_dec_map,
                )
            plt.tight_layout()
            if show:
                plt.show()
                plt.close()
            return

        self._plot_one(
            _get_arr(None),
            title=title,
            unit=self._cbar_label(None, stat),
            cmap=cmap,
            mask_uncovered=mask_uncovered,
            region=region,
            show_beam=show_beam,
            ra_dec_map=ra_dec_map,
        )
        if show:
            plt.show()
            plt.close()