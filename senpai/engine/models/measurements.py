"""Typed per-source measurements retained in full SENPAI results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhotometryStarMeasurement:
    """One catalog source and its measured photometry in a single frame.

    ``predicted_x`` and ``predicted_y`` are the final-WCS catalog projection
    used to centre the aperture. Independently measured centroids belong to
    the later astrometric-match phase and are intentionally not implied here.
    """

    source_key: str
    catalog: str | None
    catalog_id: str | None
    catalog_ra_deg: float
    catalog_dec_deg: float
    predicted_x: float | None
    predicted_y: float | None
    primary_catalog_magnitude: float | None
    catalog_magnitudes: dict[str, float] | None
    zero_point_catalog_magnitude: float | None
    zero_point_catalog_band: str | None
    flux: float
    flux_error: float
    instrumental_magnitude: float | None
    individual_zero_point_raw_flux: float | None
    individual_zero_point_per_second: float | None
    snr: float
    background_level: float
    background_std: float
    aperture_radius: float
    crowding_factor: float
    quality_flag: bool
    isolated: bool
    saturated: bool | None
    edge_clipped: bool
    used_for_zero_point: bool
    zero_point_rejection_reason: str | None
