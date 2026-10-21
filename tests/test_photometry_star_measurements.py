from dataclasses import asdict

import numpy as np
import pytest

from senpai.catalog.sstrc7_source import _stable_catalog_id
from senpai.core.config import PhotometryConfig
from senpai.engine.models.metadata import ImageMetadata
from senpai.engine.models.senpai import _compact_photometry_summary
from senpai.engine.models.starfield import StarField, StarInSpace
from senpai.engine.photometry.utils import (
    SimplePhotometryResult,
    _build_photometry_star_measurements,
    _calculate_simple_photometry_summary,
    _calculate_simple_zero_point,
    _calculate_simple_zero_point_details,
    _isolated_result_mask,
)


def _photometry_fixture():
    stars = [
        StarInSpace(
            ra=150.0 + index / 100,
            dec=-30.0 + index / 100,
            x=20.0 + index * 20,
            y=30.0 + index * 20,
            magnitude=12.0 + index,
            magnitudes={
                "Sloan_r": 12.0 + index,
                "Sloan_g": 12.5 + index,
                "Sloan_i": 11.5 + index,
            },
            catalog="SSTRC7",
            catalog_id=f"sstrc7:test-{index}",
        )
        for index in range(4)
    ]
    starfield = StarField(
        catalog_stars=stars,
        detections=[],
        image_metadata=ImageMetadata(width=100, height=120, exposure_time=10.0),
        fit=True,
        wcs=None,
    )

    results = []
    for index, star in enumerate(stars):
        flux = 10.0 * 10 ** ((25.0 - star.magnitudes["Sloan_r"]) / 2.5)
        results.append(
            SimplePhotometryResult(
                star=star,
                flux=flux,
                flux_err=flux / (50.0 if index < 3 else 10.0),
                snr=50.0 if index < 3 else 10.0,
                background_level=100.0,
                background_std=5.0,
                aperture_radius=3.0,
                crowding_factor=0.0,
                quality_flag=True,
                instrumental_magnitude=-2.5 * np.log10(flux),
            )
        )
    return starfield, results


def test_complete_measurements_preserve_linkage_and_zp_provenance():
    starfield, results = _photometry_fixture()
    config = PhotometryConfig(
        enable_color_terms=False,
        zp_min_snr=20.0,
        zp_min_stars=3,
    )

    fit = _calculate_simple_zero_point_details(results, starfield, config)
    scalar_zp, scalar_err = _calculate_simple_zero_point(results, starfield, config)
    isolated = _isolated_result_mask(results, starfield)
    measurements = _build_photometry_star_measurements(
        results, fit.selections, isolated, starfield
    )

    assert fit.zero_point == scalar_zp
    assert fit.zero_point_err == scalar_err
    assert fit.zero_point == pytest.approx(25.0)
    assert len(measurements) == len(results) == 4
    assert sum(measurement.used_for_zero_point for measurement in measurements) == 3
    assert measurements[-1].zero_point_rejection_reason == "low_snr"

    first = measurements[0]
    assert first.source_key == "SSTRC7:sstrc7:test-0"
    assert first.catalog_id == "sstrc7:test-0"
    assert first.predicted_x == 20.0
    assert first.predicted_y == 30.0
    assert first.zero_point_catalog_band == "Sloan_r"
    assert first.zero_point_catalog_magnitude == 12.0
    assert first.catalog_magnitudes == {
        "Sloan_r": 12.0,
        "Sloan_g": 12.5,
        "Sloan_i": 11.5,
    }
    assert first.individual_zero_point_per_second == 25.0
    assert first.saturated is None
    assert first.edge_clipped is False
    assert asdict(first)["flux"] == first.flux


@pytest.mark.parametrize(
    ("observation_filter", "catalog_band"),
    [("g", "Sloan_g"), ("r", "Sloan_r"), ("i", "Sloan_i")],
)
def test_observation_filter_selects_matching_catalog_band(
    observation_filter, catalog_band
):
    starfield, results = _photometry_fixture()
    for result in results:
        magnitude = result.star.magnitudes[catalog_band]
        result.flux = 10.0 * 10 ** ((25.0 - magnitude) / 2.5)

    config = PhotometryConfig(
        preferred_filters=["Sloan_r", "Sloan_g", "Sloan_i"],
        observation_filter_map={
            "g": "Sloan_g",
            "r": "Sloan_r",
            "i": "Sloan_i",
        },
        enable_color_terms=False,
        zp_min_snr=20.0,
        zp_min_stars=3,
    )

    fit = _calculate_simple_zero_point_details(
        results,
        starfield,
        config,
        observation_filter=observation_filter,
    )

    assert fit.zero_point == pytest.approx(25.0)
    assert fit.catalog_band == catalog_band
    assert fit.band_source == "observation_filter_map"
    assert {selection.catalog_band for selection in fit.selections} == {
        catalog_band
    }


def test_mapped_filter_does_not_substitute_another_band_per_star():
    starfield, results = _photometry_fixture()
    missing = results[-1]
    missing.star.magnitudes.pop("Sloan_g")
    for result in results[:-1]:
        magnitude = result.star.magnitudes["Sloan_g"]
        result.flux = 10.0 * 10 ** ((25.0 - magnitude) / 2.5)

    config = PhotometryConfig(
        preferred_filters=["Sloan_r", "Sloan_g"],
        observation_filter_map={"g": "Sloan_g", "r": "Sloan_r"},
        enable_color_terms=False,
        zp_min_snr=20.0,
        zp_min_stars=3,
    )
    fit = _calculate_simple_zero_point_details(
        results,
        starfield,
        config,
        observation_filter="g",
    )

    assert fit.zero_point == pytest.approx(25.0)
    assert fit.selections[-1].catalog_band == "Sloan_g"
    assert fit.selections[-1].catalog_magnitude is None
    assert fit.selections[-1].rejection_reason == "missing_catalog_magnitude"


def test_unknown_filter_uses_one_common_preferred_band():
    starfield, results = _photometry_fixture()
    config = PhotometryConfig(
        preferred_filters=["Sloan_r", "Sloan_g"],
        observation_filter_map={"g": "Sloan_g", "r": "Sloan_r"},
        enable_color_terms=False,
        zp_min_snr=20.0,
        zp_min_stars=3,
    )

    fit = _calculate_simple_zero_point_details(
        results,
        starfield,
        config,
        observation_filter="open",
    )

    assert fit.zero_point == pytest.approx(25.0)
    assert fit.catalog_band == "Sloan_r"
    assert fit.band_source == "preferred_common_band"
    assert {selection.catalog_band for selection in fit.selections} == {
        "Sloan_r"
    }


def test_summary_preserves_filter_and_multiband_provenance():
    starfield, results = _photometry_fixture()
    for result in results:
        magnitude = result.star.magnitudes["Sloan_g"]
        result.flux = 10.0 * 10 ** ((25.0 - magnitude) / 2.5)

    config = PhotometryConfig(
        preferred_filters=["Sloan_r", "Sloan_g"],
        observation_filter_map={"g": "Sloan_g", "r": "Sloan_r"},
        target_bands=["Sloan_g", "Sloan_r"],
        zp_min_snr=20.0,
        zp_min_stars=3,
    )
    summary = _calculate_simple_photometry_summary(
        results,
        starfield,
        config,
        observation_filter="g",
    )

    assert summary.observation_filter == "g"
    assert summary.zero_point_catalog_band == "Sloan_g"
    assert summary.zero_point_band_source == "observation_filter_map"
    assert summary.multiband_calibration is not None
    assert summary.multiband_calibration.observation_filter == "g"
    assert "Sloan_g" in summary.multiband_calibration.bands


def test_compact_summary_drops_only_complete_measurement_records():
    summary = {
        "zero_point": 25.0,
        "n_stars": 4,
        "star_measurements_schema_version": 1,
        "star_measurements": [{"source_key": "SSTRC7:1"}],
    }

    compact = _compact_photometry_summary(summary)

    assert "star_measurements" not in compact
    assert compact["star_measurements_schema_version"] == 1
    assert summary["star_measurements"] == [{"source_key": "SSTRC7:1"}]


def test_sstrc7_catalog_id_is_deterministic_at_catalog_epoch():
    first = _stable_catalog_id(1.0, -0.5, 1e-15, -2e-15, 3e-9)
    repeat = _stable_catalog_id(1.0, -0.5, 1e-15, -2e-15, 3e-9)
    other = _stable_catalog_id(1.0, -0.5, 1e-15, -2e-15, 4e-9)

    assert first == repeat
    assert first.startswith("sstrc7:")
    assert first != other
