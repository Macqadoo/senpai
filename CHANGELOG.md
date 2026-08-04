# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.6.1] - 2026-07-27

### Fixed

- Decompose the streak pixel covariance in `analyze_source_shape_fwhm` with `np.linalg.eigh`
  instead of `np.linalg.eig`. The matrix is `[[xx, xy], [xy, yy]]` — symmetric by construction, so
  its eigenvalues are real by definition. numpy < 2.5 downcast a real `eig` result to `float64`,
  hiding the general solver's complex branch; **numpy 2.5.0 dropped that downcast**
  (`np.linalg.eig(np.eye(2))` is now `complex128`), so every streak length became complex and
  `extract_streak_dims_mapping`'s `round(length / (length_std * 0.5))` raised "type
  numpy.complex128 doesn't define `__round__`", aborting the entire collect — no frame in the
  observation returned a WCS. `eigh` is version-independent and faster, and is already what the
  other six eigendecompositions in the streak code use. This is a pre-existing v2.6.0 bug exposed
  by a dependency upgrade, not a regression on this branch; note `numpy>=2.2.4` is declared with no
  upper bound, so any fresh resolve picks up 2.5.x. Measurements are unchanged — only the dtype.
- Coerce `EXPTIME` to `float` in the rate-to-rate and sidereal-to-rate solvers; a string-valued
  FITS `EXPTIME` header previously crashed the exposure arithmetic.
- Route around a rate-to-rate frame pair that shares a timestamp instead of dividing the estimated
  track rate by a zero inter-frame gap (which crashed streak sizing via `int(inf)`); the pair is
  marked processed-but-invalid, as the missing-starfield guard already does.
- Catch the `ValueError` that `fit_wcs_from_points` raises on degenerate matched-star geometry in
  `fit_and_validate_wcs` and fall back to the provided WCS, rather than failing the whole collect.
- Return the declared 3-tuple `(mag, mag50, mag90)` from every `_estimate_simple_limiting_magnitude`
  fallback path. The `no results` and `< 3 catalog stars` branches previously returned a bare float,
  so `measure_rate_starfield_photometry` / `measure_simple_starfield_photometry` crashed with
  "cannot unpack non-iterable float" and aborted the whole photometry stage on any sparse or
  degenerate frame.

### Changed

- Rewrote `rectangle_pyramoid` to build the rotated streak kernel as the exact per-pixel area
  coverage of the rotated rectangle, evaluated directly on the output grid: no OpenCV in the kernel
  path, output-sized memory only, and a hard `MAX_KERNEL_FINE_ELEMENTS` cap that raises on a
  degenerate streak-length estimate instead of allocating a huge array (the previous
  supersample-rotate-downsample approach could OOM the worker). Reproduces the previous kernel to
  correlation >= 0.98; the call signature is unchanged (`upsample`/`halo_level` accepted as no-ops).
- Reclaim process memory (`gc` + glibc `malloc_trim`) after every collect so long-lived workers do
  not ratchet RSS upward across requests. No-op on non-glibc platforms.

### Dependencies

- Require `astroeasy>=1.2.1`, which releases the plate-solve index page cache after each solve
  (bounds resident memory in long-running solve loops).
