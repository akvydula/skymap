'''
Heal pix map module for 310 MHz observations
'''

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from skymap.io import get_available_pol_names, get_pol_source

if TYPE_CHECKING:
    from skymap.io import PointingData

# Full sky solid angle in steradians; 4*pi*(180/pi)**2 gives full sky in sq.deg
FULL_SKY_STERADIANS = 4.0 * np.pi
FULL_SKY_SQ_DEG = FULL_SKY_STERADIANS * (180.0 / np.pi) ** 2
# Beam size: 1 sq.deg -> radius in deg (area = pi*r^2 => r = sqrt(1/pi))
BEAM_AREA_SQ_DEG = 1.0
BEAM_RADIUS_DEG = np.sqrt(BEAM_AREA_SQ_DEG / np.pi)



def _add_beam_indicator_radec(ax, region: tuple[float, float, float]) -> None:
    """Draw beam circle on axes where x=RA, y=Dec (data coordinates)."""
    center_ra, center_dec, area_sqdeg = region
    extent_deg = np.sqrt(float(area_sqdeg))
    margin_deg = min(0.1, extent_deg / 8)
    inset = BEAM_RADIUS_DEG + margin_deg
    beam_ra = center_ra - extent_deg / 2 + inset
    beam_dec = center_dec - extent_deg / 2 + inset
    circle = mpatches.Circle(
        (beam_ra, beam_dec), BEAM_RADIUS_DEG, fill=False, edgecolor="k", lw=1.5
    )
    ax.add_patch(circle)


def _apply_axes_kwargs(ax, kwargs: dict) -> None:
    """Apply remaining matplotlib axes kwargs (e.g. title, xlabel, ylabel, xlim, ylim)."""
    # Explicit setters for common args (kwargs already stripped of clim, xlabel, ylabel, title if used)
    for key, val in list(kwargs.items()):
        setter = getattr(ax, f"set_{key}", None)
        if callable(setter):
            setter(val)
            kwargs.pop(key, None)
    # Any remaining (e.g. xlim, ylim) via ax.set()
    if kwargs:
        try:
            ax.set(**kwargs)
        except (TypeError, AttributeError):
            pass


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

        spec_source = get_pol_source(pointing_data, kind="mean")
        if spec_source is None:
            self.fill_from_pointing(ra, dec)
            self._channel_maps.clear()
            self._channel_stds.clear()
            return

        available = get_available_pol_names(pointing_data, kind="mean")
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

    def get_map_radec_values(
        self, attribute: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (ra_deg, dec_deg, values) for all pixels with valid data, for use
        with beam histogram functions in Beam.py.

        Only pixels with hit_count > 0 and finite map values are included.
        RA and Dec are in degrees (lonlat=True). Values are the map mean (e.g. in K).

        Parameters
        ----------
        attribute : str or None
            PolChannel to use (e.g. 'AA_', 'BB_'). If None, the main map is used.

        Returns
        -------
        ra_deg : np.ndarray
            Right ascension in degrees (length n_valid).
        dec_deg : np.ndarray
            Declination in degrees (length n_valid).
        values : np.ndarray
            Map values (e.g. mean in K) for each pixel (length n_valid).
        """
        map_vals = self.get_map(attribute)
        hit_count = self.get_hit_count()
        valid = (hit_count > 0) & np.isfinite(map_vals)
        ipix = np.where(valid)[0]
        if ipix.size == 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        ra_deg, dec_deg = hp.pix2ang(self.nside, ipix, lonlat=True)
        values = np.asarray(map_vals[ipix], dtype=float)
        return ra_deg, dec_deg, values

    def _plot_one(
        self,
        map_array: np.ndarray,
        *,
        unit: str | None = None,
        mask_uncovered: bool = True,
        region: tuple[float, float, float] | None = None,
        sub: tuple[int, int, int] | None = None,
        ra_dec_map: bool = False,
        **kwargs,
    ) -> None:
        """Plot a single map: if ra_dec_map=True (and region given) use RA/Dec axes; else full-sky mollview.
        When region is specified, ra_dec_map must be True. Extra kwargs (e.g. title, cmap, clim, xlabel, ylabel) are applied to the plot/axes."""
        kwargs = dict(kwargs)  # don't mutate caller's dict when used for multiple subplots
        to_plot = np.asarray(map_array, dtype=float).copy()
        to_plot[np.isnan(to_plot)] = hp.UNSEEN
        if mask_uncovered:
            to_plot[to_plot == 0] = hp.UNSEEN

        # Consume plot-level kwargs (clim, title, cmap; healpy uses min/max)
        clim = kwargs.pop("clim", None)
        title = kwargs.pop("title", "HealPix map")
        cmap = kwargs.pop("cmap", "viridis")
        xlabel = kwargs.pop("xlabel", None)
        ylabel = kwargs.pop("ylabel", None)

        if region is not None and not ra_dec_map:
            raise ValueError("When a region is specified, ra_dec_map must be True")
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
            if clim is not None:
                sc.set_clim(clim[0], clim[1])
            ax.set_xlim(center_ra - extent_deg / 2, center_ra + extent_deg / 2)
            ax.set_ylim(center_dec - extent_deg / 2, center_dec + extent_deg / 2)
            ax.set_xlabel(xlabel if xlabel is not None else "RA")
            ax.set_ylabel(ylabel if ylabel is not None else "Dec")
            ax.set_aspect("equal")
            ax.set_title(title or "")
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label(unit or "", fontweight="normal")
            _add_beam_indicator_radec(ax, region)
            _apply_axes_kwargs(ax, kwargs)
            return

        # Regular healpy plot: mollview (full sky)
        hp_kw = dict(title=title, unit=unit, cbar=True, cmap=cmap)
        if clim is not None:
            hp_kw["min"] = clim[0]
            hp_kw["max"] = clim[1]
        if sub is not None:
            hp_kw["sub"] = sub
        hp.mollview(to_plot, **hp_kw)
        ax = plt.gca()
        _apply_axes_kwargs(ax, kwargs)

    def plot(
        self,
        *,
        attribute: str | None = None,
        stat: Literal["mean", "std"] = "mean",
        unit: str | None = None,
        mask_uncovered: bool = True,
        region: tuple[float, float, float] | None = None,
        ra_dec_map: bool = False,
        **kwargs,
    ) -> None:
        """
        Plot the map.

        If ra_dec_map=True (and region given): plot with RA and Dec as X/Y axes using
        healpix pixels in the region (query_disc + pix2ang); beam circle is drawn automatically.
        Otherwise: full-sky mollview. When region is specified, ra_dec_map must be True.

        Parameters
        ----------
        attribute : str or None, optional
            PolChannel name to plot (e.g. 'AA_', 'BB_'). If None and channel maps exist,
            plot all channels in a subplot grid; if None and no channel maps, plot the
            main (hit count) map.
        stat : {'mean', 'std'}, optional
            Plot mean map or standard-deviation map. Default 'mean'.
        unit : str or None, optional
            Unused; color bar title is always '{attribute} {stat} K'.
        mask_uncovered : bool, optional
            If True (default), pixels with no hits (value 0) are masked as UNSEEN.
        region : tuple (center_ra, center_dec, area_sqdeg) or None, optional
            Patch to plot. Required when ra_dec_map=True. When region is given,
            ra_dec_map must be True. When None and ra_dec_map=False, mollview (full sky) is used.
        ra_dec_map : bool, optional
            If True, plot with RA as X axis and Dec as Y axis (requires region).
            If False (default), use regular healpy projection (mollview).
        **kwargs : dict, optional
            Matplotlib/plot arguments: e.g. title, cmap, clim (vmin, vmax), xlabel, ylabel,
            xlim, ylim. Figure-related (figsize, dpi, etc.) are used when creating the figure.
        """
        # Split figure kwargs (for plt.figure) from plot kwargs (for _plot_one)
        figure_keys = {"figsize", "dpi", "facecolor", "edgecolor", "frameon", "num", "clear"}
        figure_kw = {k: v for k, v in kwargs.items() if k in figure_keys}
        plot_kw = {k: v for k, v in kwargs.items() if k not in figure_keys}

        if stat == "std":
            def _get_arr(attr: str | None) -> np.ndarray:
                return self.get_std(attr)
        else:
            def _get_arr(attr: str | None) -> np.ndarray:
                return self.get_map(attr)

        stat_str = stat.capitalize()
        cbar_label = lambda attr: f"{attr} {stat_str} K" if attr else f"{stat_str} K"

        if attribute is not None:
            if attribute not in self._channel_maps:
                raise KeyError(f"No map for {attribute!r}. Available: {list(self._channel_maps.keys())}")
            if figure_kw:
                plt.figure(**figure_kw)
            self._plot_one(
                _get_arr(attribute),
                unit=cbar_label(attribute),
                mask_uncovered=mask_uncovered,
                region=region,
                ra_dec_map=ra_dec_map,
                **{**plot_kw, "title": plot_kw.get("title", attribute)},
            )
            plt.show()
            plt.close()
            return

        if self._channel_maps:
            names = sorted(self._channel_maps.keys())
            n = len(names)
            ncol = min(n, 3)
            nrow = (n + ncol - 1) // ncol
            plt.figure(figsize=(5 * ncol, 4 * nrow), **figure_kw)
            if plot_kw.get("title"):
                plt.suptitle(plot_kw["title"])
            for i, name in enumerate(names):
                self._plot_one(
                    _get_arr(name),
                    unit=cbar_label(name),
                    mask_uncovered=mask_uncovered,
                    region=region,
                    sub=(nrow, ncol, i + 1),
                    ra_dec_map=ra_dec_map,
                    **{**plot_kw, "title": name},
                )
            plt.tight_layout()
            plt.show()
            plt.close()
            return

        if figure_kw:
            plt.figure(**figure_kw)
        self._plot_one(
            _get_arr(None),
            unit=cbar_label(None),
            mask_uncovered=mask_uncovered,
            region=region,
            ra_dec_map=ra_dec_map,
            **plot_kw,
        )
        plt.show()
        plt.close()
