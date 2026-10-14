"""Hermetic tests for the thin Huntsman ingestion adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from senpai.cli.common import write_json
from senpai.cli.huntsman import build_parser
from senpai.core import config as config_module
from senpai.core.config import AppConfig, load_yaml
from senpai.engine.models.metadata import TrackMode
from senpai.engine.utils.file_io import load_fits_file
from senpai.integrations.huntsman.batches import (
    HuntsmanBatch,
    HuntsmanDiscovery,
    HuntsmanFrameRecord,
    discover_huntsman_batches,
)
from senpai.integrations.huntsman.filenames import (
    ParsedHuntsmanFilename,
    fits_stem,
    is_fits_file,
    parse_huntsman_filename,
)
from senpai.integrations.huntsman.quality import (
    SiderealQualityResult,
    select_batch_frames,
)
from senpai.integrations.huntsman.runner import (
    batch_output_directory,
    batch_relative_directory,
    has_pending_batches,
    run_huntsman_batches,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "resources" / "config"


@pytest.fixture
def huntsman_config(monkeypatch) -> AppConfig:
    config = AppConfig(**load_yaml(CONFIG_DIR / "huntsman.yaml"))
    monkeypatch.setattr(config_module, "_config_instance", config)
    return config


def _write_fits(path: Path, **header_values) -> None:
    header = fits.Header()
    for key, value in header_values.items():
        header[key] = value
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32), header=header).writeto(path)


def _write_compressed_fits(path: Path, **header_values) -> None:
    header = fits.Header()
    for key, value in header_values.items():
        header[key] = value
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.CompImageHDU(
                np.zeros((8, 8), dtype=np.uint16),
                header=header,
                compression_type="RICE_1",
            ),
        ]
    ).writeto(path)


def _record(path: str, mode: TrackMode) -> HuntsmanFrameRecord:
    parsed = ParsedHuntsmanFilename(
        path=Path(path),
        convention="new",
        filename_task_id="019fb948",
        filename_filter="g",
        filename_track_mode=mode,
    )
    return HuntsmanFrameRecord(
        path=parsed.path,
        parsed=parsed,
        task_id="task-full",
        task_id_source="SKTASKID",
        filter_name="g",
        sensor="camera-1",
        track_mode=mode,
        track_mode_source="TRKMODE",
        norad_id="25371",
    )


@pytest.mark.parametrize(
    "name",
    [
        "image.fits",
        "image.fit",
        "image.fts",
        "image.fits.fz",
        "image.fit.fz",
        "image.fts.fz",
        "IMAGE.FITS.FZ",
    ],
)
def test_supported_fits_suffixes(name):
    assert is_fits_file(name)
    assert fits_stem(name).casefold() == "image"


def test_parse_old_huntsman_filename():
    parsed = parse_huntsman_filename("20260506T081156614_rate_0968299a_F0of2_r.fits.fz")
    assert parsed.recognized
    assert parsed.convention == "old"
    assert parsed.filename_task_id == "0968299a"
    assert parsed.filename_filter == "r"
    assert parsed.filename_track_mode == TrackMode.RATE
    assert parsed.frame_index == 0
    assert parsed.frame_count == 2
    assert parsed.timestamp.isoformat() == "2026-05-06T08:11:56.614000+00:00"


def test_parse_new_huntsman_filename():
    parsed = parse_huntsman_filename("260731T174744_25371_019fb948_g_F1of2_sidereal.fits")
    assert parsed.recognized
    assert parsed.convention == "new"
    assert parsed.filename_norad_id == "25371"
    assert parsed.filename_task_id == "019fb948"
    assert parsed.filename_filter == "g"
    assert parsed.filename_track_mode == TrackMode.SIDEREAL
    assert parsed.frame_index == 1
    assert parsed.frame_count == 2


def test_unknown_filename_is_non_raising():
    parsed = parse_huntsman_filename("unstructured.fits")
    assert not parsed.recognized
    assert parsed.filename_task_id is None


def test_discovery_prefers_full_task_id_and_separates_filters(tmp_path, huntsman_config):
    full_id = "019fb948-aaaa-bbbb-cccc-0123456789ab"
    common = {
        "SKTASKID": full_id,
        "TASKID": "different-task-id",
        "NORAD_ID": 25371,
        "INSTRUME": "Huntsman-1",
    }
    _write_fits(
        tmp_path / "260731T174744_25371_019fb948_g_F0of2_sidereal.fits",
        **common,
        FILTER="g",
        TRKMODE="sidereal",
    )
    _write_fits(
        tmp_path / "260731T174745_25371_019fb948_g_F1of2_rate.fits",
        **common,
        FILTER="g",
        TRKMODE="rate",
    )
    _write_fits(
        tmp_path / "260731T174746_25371_019fb948_r_F0of1_rate.fits",
        **common,
        FILTER="r",
        TRKMODE="rate",
    )

    discovery = discover_huntsman_batches(tmp_path, huntsman_config)

    assert not discovery.skipped
    assert len(discovery.batches) == 2
    by_filter = {batch.filter_name: batch for batch in discovery.batches}
    assert by_filter["g"].task_id == full_id
    assert by_filter["g"].task_id_source == "SKTASKID"
    assert by_filter["g"].sktask_id == full_id
    assert by_filter["g"].controller_task_id == "different-task-id"
    assert len(by_filter["g"].sidereal_frames) == 1
    assert len(by_filter["g"].rate_frames) == 1
    assert len(by_filter["r"].rate_frames) == 1


def test_discovery_preserves_header_identity_and_separates_cameras(tmp_path, huntsman_config):
    common = {
        "SKTASKID": "sensorkit-task",
        "TASKID": "controller-task",
        "SENSORID": "asi183-serial",
        "TARGET": "25371",
        "TARGTYPE": "NORAD",
        "NORAD_ID": 25371,
        "FILTER": "r",
        "INSTRUME": "ZWO ASI183MM Pro",
        "EGAIN": 0.1,
        "RDNOISE": 1.2,
    }
    for camera_id in ("C3", "C4"):
        camera_dir = tmp_path / camera_id
        camera_dir.mkdir()
        _write_fits(
            camera_dir / f"{camera_id}_rate.fits",
            **common,
            CAMERAID=camera_id,
            TRKMODE="rate",
        )
        _write_fits(
            camera_dir / f"{camera_id}_sidereal.fits",
            **common,
            CAMERAID=camera_id,
            TRKMODE="sidereal",
        )

    discovery = discover_huntsman_batches(tmp_path, huntsman_config)

    assert not discovery.skipped
    assert len(discovery.batches) == 2
    by_camera = {batch.camera_id: batch for batch in discovery.batches}
    assert set(by_camera) == {"C3", "C4"}
    for camera_id, batch in by_camera.items():
        assert f"_{camera_id}_" in batch.output_label
        assert batch.sktask_id == "sensorkit-task"
        assert batch.controller_task_id == "controller-task"
        assert batch.sensor_id == "asi183-serial"
        assert batch.filter_source == "FILTER"
        assert batch.target_id == "25371"
        assert batch.target_id_source == "TARGET"
        assert batch.target_type == "NORAD"
        assert batch.target_type_source == "TARGTYPE"
        assert batch.norad_id == "25371"
        assert batch.norad_id_source == "NORAD_ID"
        assert batch.gain_e_per_adu == pytest.approx(0.1)
        assert batch.gain_source == "EGAIN"
        assert batch.read_noise_electrons == pytest.approx(1.2)
        assert batch.read_noise_source == "RDNOISE"
        assert len(batch.rate_frames) == 1
        assert len(batch.sidereal_frames) == 1


def test_discovery_accepts_deployed_huntsman_identity_cards(tmp_path, huntsman_config):
    common = {
        "SKTASKID": "sensorkit-task",
        "TASKID": "controller-task",
        "CAMSERNO": "camera-serial",
        "TARGET": "42917",
        "FILTER": "r",
        "INSTRUME": "ZWO ASI183MM Pro",
    }
    for unit_id in ("C5", "C7"):
        unit_dir = tmp_path / unit_id
        unit_dir.mkdir()
        _write_compressed_fits(
            unit_dir / f"{unit_id}_rate.fits",
            **common,
            UNITID=unit_id,
            TRKMODE="rate",
        )
        _write_compressed_fits(
            unit_dir / f"{unit_id}_sidereal.fits",
            **common,
            UNITID=unit_id,
            TRKMODE="sidereal",
        )

    discovery = discover_huntsman_batches(tmp_path, huntsman_config)

    assert not discovery.skipped
    assert len(discovery.batches) == 2
    by_unit = {batch.camera_id: batch for batch in discovery.batches}
    assert set(by_unit) == {"C5", "C7"}
    for unit_id, batch in by_unit.items():
        assert batch.camera_id_source == "UNITID"
        assert batch.sensor_id == "camera-serial"
        assert batch.sensor_id_source == "CAMSERNO"
        assert batch.target_id == "42917"
        assert len(batch.rate_frames) == 1
        assert len(batch.sidereal_frames) == 1
        manifest = batch.to_manifest()
        assert manifest["camera_id"] == unit_id
        assert manifest["camera_id_source"] == "UNITID"
        assert manifest["sensor_id_source"] == "CAMSERNO"


def test_loader_reads_sensorkit_compressed_image_hdu(tmp_path):
    path = tmp_path / "sensorkit-compressed.fits"
    _write_compressed_fits(
        path,
        SKTASKID="sensorkit-task",
        UNITID="C8",
        FILTER="r",
        TRKMODE="rate",
        EXPTIME=10.0,
    )

    image = load_fits_file(path)

    assert image.data.shape == (8, 8)
    assert image.header["SKTASKID"] == "sensorkit-task"
    assert image.header["UNITID"] == "C8"
    assert image.metadata.exposure_time == pytest.approx(10.0)


def test_trkmode_overrides_filename_label(tmp_path, huntsman_config):
    path = tmp_path / "20260506T081156614_rate_0968299a_F0of2_r.fits"
    _write_fits(
        path,
        TASKID="full-task-id",
        FILTER="r",
        TRKMODE="sidereal",
        INSTRUME="Huntsman-1",
    )

    discovery = discover_huntsman_batches(tmp_path, huntsman_config)

    record = discovery.batches[0].frames[0]
    assert record.track_mode == TrackMode.SIDEREAL
    assert record.track_mode_source == "TRKMODE"
    assert record.mode_disagreement is True


def test_discovery_accepts_compound_fz_suffix(tmp_path, huntsman_config):
    path = tmp_path / "260731T174745_25371_019fb948_g_F1of2_rate.fits.fz"
    _write_fits(path, SKTASKID="full-task", FILTER="g", TRKMODE="rate")

    discovery = discover_huntsman_batches(tmp_path, huntsman_config)

    assert len(discovery.batches) == 1
    assert discovery.batches[0].rate_frames[0].path == path


def test_output_label_contains_norad_filter_sensor_and_full_task():
    batch = HuntsmanBatch(
        task_id="full-task-id",
        task_id_source="SKTASKID",
        filter_name="g",
        sensor="Huntsman-1",
        norad_id="25371",
        frames=[_record("rate.fits", TrackMode.RATE)],
    )
    assert batch.output_label == "obsunknown_25371_g_Huntsman-1_full-task-id"
    assert batch_relative_directory(batch) == Path("batches/full-task-id/g")


def test_sharpest_policy_selects_lowest_fwhm():
    first = _record("first.fits", TrackMode.SIDEREAL)
    second = _record("second.fits", TrackMode.SIDEREAL)
    rate = _record("rate.fits", TrackMode.RATE)
    batch = HuntsmanBatch(
        task_id="task-full",
        task_id_source="SKTASKID",
        filter_name="g",
        sensor="camera-1",
        norad_id="25371",
        frames=[first, second, rate],
    )
    fwhm = {first.path: 8.0, second.path: 4.0}

    def evaluate(path, **_kwargs):
        return SiderealQualityResult(
            path=path,
            passed=True,
            conclusive=True,
            median_fwhm=fwhm[path],
            detected_source_count=20,
        )

    selection = select_batch_frames(batch, policy="sharpest", evaluator=evaluate)

    assert selection.selected_sidereal == (second,)
    assert selection.selected_files == (second.path, rate.path)
    assert not selection.rejected_for_blur


def test_default_quality_gate_rejects_blurry_sidereal_batch():
    sidereal = _record("sidereal.fits", TrackMode.SIDEREAL)
    rate = _record("rate.fits", TrackMode.RATE)
    batch = HuntsmanBatch(
        task_id="task-full",
        task_id_source="SKTASKID",
        filter_name="g",
        sensor="camera-1",
        norad_id="25371",
        frames=[sidereal, rate],
    )

    def evaluate(path, **_kwargs):
        return SiderealQualityResult(
            path=path,
            passed=False,
            conclusive=True,
            median_fwhm=25.0,
            detected_source_count=20,
            reason="too blurry",
        )

    selection = select_batch_frames(batch, policy="first", evaluator=evaluate)

    assert selection.rejected_for_blur
    assert selection.selected_files == ()


@pytest.mark.parametrize("name", ["huntsman.yaml", "huntsman_containerized.yaml"])
def test_huntsman_profiles_use_required_headers(name):
    config = AppConfig(**load_yaml(CONFIG_DIR / name))
    assert config.headers.pointing.target_ra_keys == ["RA"]
    assert config.headers.pointing.target_dec_keys == ["DEC"]
    assert config.headers.pointing.ra_dec_format == "float"
    assert config.headers.pointing.ra_units == "degrees"
    assert config.headers.pointing.dec_units == "degrees"
    assert config.headers.site.positional_format == "float"
    assert config.headers.site.positional_unit == "degrees"
    assert config.headers.site.site_altitude_keys == ["SITEELEV"]
    assert config.headers.tracking.track_mode_keys == ["TRKMODE"]
    assert config.headers.tracking.track_ra_rate_keys == ["RA_RATE"]
    assert config.headers.tracking.track_dec_rate_keys == ["DEC_RATE"]
    assert config.headers.tracking.track_ra_rate_unit == "degrees/second"
    assert config.headers.tracking.track_dec_rate_unit == "degrees/second"
    assert config.headers.filter_keys == ["FILTER"]
    assert config.runtime.save_processed_fits is False


def test_minimal_cli_defaults_to_safe_skips():
    args = build_parser().parse_args(["input"])
    assert args.sidereal_policy == "sharpest"
    assert args.reprocess is False
    assert args.allow_blurry_sidereal is False


def test_completed_batch_needs_no_pipeline_preflight(tmp_path, huntsman_config):
    batch = HuntsmanBatch(
        task_id="full-task-id",
        task_id_source="SKTASKID",
        filter_name="g",
        sensor="Huntsman-1",
        norad_id="25371",
        frames=[_record("rate.fits", TrackMode.RATE)],
    )
    output_dir = batch_output_directory(tmp_path, batch)
    output_dir.mkdir(parents=True)
    write_json(
        output_dir / "huntsman_manifest.json",
        {
            "status": "success",
            "science_report_id": "report-1",
            "science_report_path": str(output_dir / "science_report.json"),
            "science_report_document_count": 1,
        },
    )

    discovery = HuntsmanDiscovery(batches=[batch])

    assert not has_pending_batches(discovery, tmp_path)
    assert has_pending_batches(discovery, tmp_path, skip_existing=False)
    results = run_huntsman_batches(discovery, tmp_path, huntsman_config)
    assert results[0]["status"] == "skipped_existing"
    assert results[0]["existing_status"] == "success"
    assert results[0]["science_report_id"] == "report-1"


def test_v2_layout_rejects_mixed_cameras_with_same_task_and_filter(tmp_path, huntsman_config):
    batches = [
        HuntsmanBatch(
            task_id="sensor-task",
            task_id_source="SKTASKID",
            filter_name="r",
            sensor="shared-model",
            norad_id="25371",
            frames=[_record(f"{camera_id}.fits", TrackMode.RATE)],
            sktask_id="sensor-task",
            camera_id=camera_id,
        )
        for camera_id in ("C3", "C4")
    ]

    with pytest.raises(ValueError, match="separate task/camera product root"):
        run_huntsman_batches(
            HuntsmanDiscovery(batches=batches),
            tmp_path,
            huntsman_config,
        )


def test_shared_json_writer_is_pretty_and_round_trips(tmp_path):
    path = tmp_path / "result.json"
    payload = {"group": {"task": "abc"}, "frames": [1, 2]}

    write_json(path, payload)

    text = path.read_text()
    assert text.endswith("\n")
    assert '\n  "group"' in text
    assert json.loads(text) == payload
