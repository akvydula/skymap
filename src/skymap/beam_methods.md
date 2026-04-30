# Beam Deconvolution Methods

This note describes the deconvolution pipeline implemented in `Beam.convolve_beam_with_fit` and related helpers in `Beam.py`.

## Model

For an axisymmetric beam, the observed sky in harmonic space is modeled as:

`d_lm = B_l s_lm + n_lm`

- `d_lm`: observed spherical-harmonic coefficients
- `s_lm`: true sky coefficients
- `B_l`: beam transfer function
- `n_lm`: noise coefficients

The goal is to recover an estimate of `s_lm` from `d_lm`.

## Step-by-step pipeline

1. **Build observed HealPix map from pointings**
   - For each selected channel (`AA_`, `AB_`, etc.), input pointings (`ra`, `dec`) and values are gridded with `HealPixMap.fill_from_pointing`.
   - This produces:
     - `raw_map` (observed map in RING ordering)
     - `hit_count` (coverage map)

2. **Mitigate partial-sky edge effects (padding + taper)**
   - If only a small patch of sky is observed, the hard mask/window (`hit_count > 0`) produces sinc-like ringing and leakage in harmonic space.
   - `Beam.convolve_beam_with_fit` supports three related options to suppress this:
     - `pad_unobserved=True`: fill pixels with `hit_count <= 0` before `hp.map2alm` (instead of setting them to `hp.UNSEEN`).
     - `pad_value=...`: sets the outside-of-patch baseline. 
       - `pad_value=params["baseline_k"]` (fit floor) or `pad_value="median"`, **not** the patch mean.
     - `gaussian_taper_fwhm_deg=...`: apply a Gaussian taper window derived from a smoothed hit-mask, blending the observed patch into the padding baseline before `hp.map2alm`.
       - This directly targets the sinc-like ringing from a hard-edged patch window.
     - `apodize_fwhm_deg=...`: an additional smooth blend of the observed/unobserved boundary (can be used together with the taper).

2. **Construct the beam in harmonic space**
   - Beam width is read from `beam_params` (`FWHM_deg` or derived from `sigma_deg`).
   - Convert to radians: `fwhm_rad`.
   - Compute beam window:
     - `B_l = hp.gauss_beam(fwhm_rad, lmax)`

3. **Map to spherical harmonics**
   - Default behavior: pixels with `hit_count <= 0` are masked as `hp.UNSEEN`.
   - If `pad_unobserved=True`, the map is first padded/tapered and **not** masked as `UNSEEN` before `hp.map2alm`.
   - Compute observed coefficients:
     - `alm_obs = hp.map2alm(raw_map, lmax=lmax, iter=map2alm_iter)`

4. **Build a stable inverse filter `F_l`**

   Two methods are supported:

   - **Regularized inversion** (`method="regularized"`):
     - `F_l = B_l / (B_l^2 + alpha)`
     - `alpha` is `regularization_alpha` (Tikhonov-like stabilization).

   - **Wiener filtering** (`method="wiener"` or `"weiner"`):
     - `F_l = (B_l C_l^S) / (B_l^2 C_l^S + C_l^N)`
     - Needs tests anc cheks [INCOMPLETE]

   Additional stabilization:
   - Modes with very small beam response are suppressed via `bl_floor`.
   - Only modes with `B_l > bl_floor * max(B_l)` are used.

5. **Apply filter in harmonic space**
   - `alm_dec = hp.almxfl(alm_obs, F_l)`

6. **Transform back to map space**
   - `dec_map = hp.alm2map(alm_dec, nside)`
   - Uncovered pixels are set back to zero in outputs.

7. **Sample at original pointing coordinates**
   - Interpolate deconvolved map back to input pointings:
     - `deconvolved[attr] = hp.get_interp_val(dec_map, theta, phi)`

8. **residual-based noise estimation**
   - If `estimate_noise_spectrum=True`, the code:
     1. Re-convolves `dec_map` with `B_l`
     2. Builds residual map: `residual = observed - reconvolved_model`
     3. Estimates noise spectrum with `hp.anafast(residual)`
   - This is returned in `noise_spectra`.

## Notes on practical behavior

- Deconvolution is an inverse problem and does not enforce positivity; negative pixels can appear.
- Output dynamic range can be larger than the input map, especially when regularization is weak.
- More stable results usually come from:
  - larger `regularization_alpha` (regularized mode),
  - larger `bl_floor`,
  - Wiener mode with realistic `signal_cl` and `noise_cl`.

### Partial-sky ringing (sinc-like artifacts)

If only a small patch is observed (e.g. ~10° around a bright source), the hard patch boundary behaves like a sharp window. In harmonic space this creates leakage/ringing that can reappear strongly after inverse-beam filtering.


Example (bright source, ~10° patch):

```python
nside = 1024
convolved_beam = Beam.convolve_beam_with_fit(
    beam_obs_pointing,
    params,
    nside=nside,
    attribute="AB_",
    lmax = nside,
    bl_floor = 1e-3,
    regularization_alpha = 1e-1,
    method = "regularized",
    pad_unobserved = True,
    pad_value = params.get("baseline_k", "median"),
    apodize_fwhm_deg = 5.0,
    gaussian_taper_fwhm_deg=15.0,
    gaussian_taper_normalize=True,
)

```

## What deconvolution and reconvolution mean here

In this code path:

- **Convolution** means the true sky is blurred by the telescope beam.
- **Deconvolution** means estimating a sharper sky by approximately undoing that blur.
- **Re-convolution** means blurring the recovered sky again with the same beam to validate the fit.

Using the harmonic model:

- `d_lm = B_l s_lm + n_lm`
  - `d_lm`: observed map coefficients
  - `s_lm`: true sky coefficients
  - `B_l`: beam response
  - `n_lm`: noise

Deconvolution computes an estimate `s_hat_lm` from `d_lm` using a stable inverse filter `F_l`:

- `s_hat_lm = F_l * d_lm`

with either:

- regularized inversion: `F_l = B_l / (B_l^2 + alpha)`, or
- Wiener filtering: `F_l = (B_l C_l^S) / (B_l^2 C_l^S + C_l^N)`.

Re-convolution is then:

- `d_hat_lm = B_l * s_hat_lm`

Comparing `d_hat` to the original observed map `d` (map domain or at original pointings) is the main consistency check.

