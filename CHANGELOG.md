# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.6.1] - 2026-07-27

### Fixed

- Coerce `EXPTIME` to `float` in the rate-to-rate and sidereal-to-rate solvers; a string-valued
  FITS `EXPTIME` header previously crashed the exposure arithmetic.
- Route around a rate-to-rate frame pair that shares a timestamp instead of dividing the estimated
  track rate by a zero inter-frame gap (which crashed streak sizing via `int(inf)`); the pair is
  marked processed-but-invalid, as the missing-starfield guard already does.
- Catch the `ValueError` that `fit_wcs_from_points` raises on degenerate matched-star geometry in
  `fit_and_validate_wcs` and fall back to the provided WCS, rather than failing the whole collect.

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
