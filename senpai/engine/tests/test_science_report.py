"""Science-report contract tests for the Huntsman adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from senpai.core.config import AppConfig, load_yaml
from senpai.engine.models.metadata import CollectionMetadata, TrackMode
from senpai.engine.models.senpai import (
    FrameSummary,
    SenpaiRunResult,
    SenpaiRunSummary,
    SiderealFrameSerializable,
)
from senpai.engine.models.starfield import SatelliteInImage
from senpai.integrations.huntsman import runner as huntsman_runner
from senpai.integrations.huntsman import science_report as science_report_module
from senpai.integrations.huntsman.batches import HuntsmanBatch, HuntsmanFrameRecord
from senpai.integrations.huntsman.filenames import ParsedHuntsmanFilename
from senpai.integrations.huntsman.quality import HuntsmanBatchSelection
from senpai.integrations.huntsman.runner import batch_output_directory
from senpai.integrations.huntsman.science_report import (
    SCIENCE_REPORT_FILENAME,
    build_science_report,
    write_science_report,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "resources" / "config" / "huntsman.yaml"


def _source_fits(path: Path) -> None:
    header = fits.Header()
    header["SKTASKID"] = "sensorkit-task"
    header["TASKID"] = "controller-task"
    header["UNITID"] = "C5"
    header["CAMSERNO"] = "camera-serial"
    header["TARGET"] = "42917"
    header["TARGTYPE"] = "NORAD"
    header["NORAD_ID"] = "42917"
    header["FILTER"] = "r"
    header["TRKMODE"] = "sidereal"
    header["DATE-OBS"] = "2026-10-14T10:00:00+00:00"
    header["EXPTIME"] = 10.0
    header["SATURATE"] = 65535
    fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32), header=header).writeto(path)


def _products(tmp_path: Path, n_measurements: int = 1):
    source = tmp_path / "frame.fits"
    _source_fits(source)
    parsed = ParsedHuntsmanFilename(
        path=source,
        convention="unknown",
        filename_track_mode=TrackMode.SIDEREAL,
    )
    record = HuntsmanFrameRecord(
        path=source,
        parsed=parsed,
        task_id="sensorkit-task",
        task_id_source="SKTASKID",
        filter_name="r",
        filter_source="FILTER",
        sensor="ZWO ASI183MM Pro",
        track_mode=TrackMode.SIDEREAL,
        track_mode_source="TRKMODE",
        norad_id="42917",
        sktask_id="sensorkit-task",
        controller_task_id="controller-task",
        camera_id="C5",
        camera_id_source="UNITID",
        sensor_id="camera-serial",
        sensor_id_source="CAMSERNO",
        target_id="42917",
        target_id_source="TARGET",
        target_type="NORAD",
        target_type_source="TARGTYPE",
        norad_id_source="NORAD_ID",
        gain_e_per_adu=0.1,
        gain_source="EGAIN",
        read_noise_electrons=1.2,
        read_noise_source="RDNOISE",
    )
    batch = HuntsmanBatch(
        task_id=record.task_id,
        task_id_source=record.task_id_source,
        filter_name=record.filter_name,
        filter_source=record.filter_source,
        sensor=record.sensor,
        norad_id=record.norad_id,
        frames=[record],
        sktask_id=record.sktask_id,
        controller_task_id=record.controller_task_id,
        camera_id=record.camera_id,
        camera_id_source=record.camera_id_source,
        sensor_id=record.sensor_id,
        sensor_id_source=record.sensor_id_source,
        target_id=record.target_id,
        target_id_source=record.target_id_source,
        target_type=record.target_type,
        target_type_source=record.target_type_source,
        norad_id_source=record.norad_id_source,
        gain_e_per_adu=record.gain_e_per_adu,
        gain_source=record.gain_source,
        read_noise_electrons=record.read_noise_electrons,
        read_noise_source=record.read_noise_source,
    )
    selection = HuntsmanBatchSelection(
        batch=batch,
        policy="first",
        selected_sidereal=(record,),
        skipped_sidereal=(),
        quality=(),
    )
    measurements = [
        {
            "source_key": f"SSTRC7:sstrc7:test-{index}",
            "catalog": "SSTRC7",
            "catalog_id": f"sstrc7:test-{index}",
            "catalog_ra_deg": 150.0 + index / 1000,
            "catalog_dec_deg": -30.0,
            "catalog_magnitudes": {"Sloan_r": 12.0},
            "flux": 1000.0,
            "flux_error": 10.0,
            "snr": 100.0,
            "used_for_zero_point": True,
        }
        for index in range(n_measurements)
    ]
    photometry = {
        "n_stars": n_measurements,
        "median_background": 100.0,
        "zero_point": 25.0,
        "stars_mag": [12.0] * n_measurements,
        "stars_snr": [100.0] * n_measurements,
        "star_measurements_schema_version": 1,
        "star_measurements": measurements,
    }
    result = SenpaiRunResult(
        id=batch.batch_id,
        num_frames=1,
        collect_metadata=CollectionMetadata(),
        completed=True,
        compute_seconds=2.5,
        sidereal_frames=[
            SiderealFrameSerializable(
                index=0,
                timestamp="2026-10-14T10:00:00+00:00",
                original_frame_path=str(source),
                original_frame_header=dict(fits.getheader(source)),
                photometry_summary=photometry,
            )
        ],
    )
    summary = SenpaiRunSummary(
        id=batch.batch_id,
        num_frames=1,
        collect_metadata=CollectionMetadata(),
        completed=True,
        compute_seconds=2.5,
        frames=[
            FrameSummary(
                index=0,
                timestamp="2026-10-14T10:00:00+00:00",
                track_mode="sidereal",
                original_frame_path=str(source),
                photometry_summary={"zero_point": 25.0},
                limiting_magnitude=18.5,
                detections=[SatelliteInImage(x=10, y=20, snr=8, ra=150, dec=-30)],
            )
        ],
    )
    config = AppConfig(**load_yaml(CONFIG_PATH))
    output = tmp_path / "products"
    output.mkdir()
    return batch, selection, result, summary, config, output


def test_science_report_contains_linked_science_and_complete_headers(tmp_path):
    batch, selection, result, summary, config, output = _products(tmp_path)

    report = build_science_report(
        batch=batch,
        selection=selection,
        config=config,
        batch_dir=output,
        status="success",
        error=None,
        elapsed_seconds=3.0,
        result=result,
        summary=summary,
        terminal_manifest={"status": "success"},
    )
    written = write_science_report(report, output)
    payload = json.loads(written.root_path.read_text())

    assert written.root_path.name == SCIENCE_REPORT_FILENAME
    assert written.document_count == 1
    assert payload["schema_version"] == 1
    assert payload["layout_version"] == 2
    assert payload["document_type"] == "run"
    assert payload["batch"]["metadata_sources"]["camera_id"] == "UNITID"
    assert payload["batch"]["metadata_sources"]["sensor_id"] == "CAMSERNO"
    assert payload["observation"]["time_end_utc"] == "2026-10-14T10:00:10+00:00"
    assert payload["input_frames"][0]["sha256"]
    cards = payload["input_frames"][0]["fits_headers"][0]["cards"]
    assert any(card["keyword"] == "SKTASKID" for card in cards)

    frame = payload["frames"][0]
    assert frame["frame_id"].startswith(payload["science_report_id"])
    assert frame["quality"]["saturation_level_adu"] == 65535
    assert frame["detections"][0]["measurement_id"].startswith("detection:")
    measurement = frame["photometry"]["star_measurements"][0]
    assert measurement["frame_id"] == frame["frame_id"]
    assert measurement["measurement_id"].startswith("measurement:")
    assert "stars_mag" not in frame["photometry"]
    assert "NaN" not in written.root_path.read_text()


def test_large_calibration_results_use_linked_same_name_parts(tmp_path):
    batch, selection, result, summary, config, output = _products(tmp_path, n_measurements=200)
    report = build_science_report(
        batch=batch,
        selection=selection,
        config=config,
        batch_dir=output,
        status="success",
        error=None,
        elapsed_seconds=3.0,
        result=result,
        summary=summary,
    )

    written = write_science_report(report, output, max_document_bytes=20_000)
    root = json.loads(written.root_path.read_text())
    parts = [json.loads(path.read_text()) for path in written.part_paths]

    assert written.document_count > 1
    assert all(path.name == SCIENCE_REPORT_FILENAME for path in written.part_paths)
    assert root["calibration_measurements_embedded"] is False
    assert root["parts_total"] == written.document_count
    assert len(root["measurement_parts"]) == len(written.part_paths)
    assert sum(part["measurement_count"] for part in parts) == 200
    assert all(part["science_report_id"] == root["science_report_id"] for part in parts)
    assert all(len(path.read_bytes()) <= 20_000 for path in written.part_paths)


def test_calibration_chunk_sizing_is_linear(tmp_path, monkeypatch):
    n_measurements = 500
    batch, selection, result, summary, config, output = _products(tmp_path, n_measurements=n_measurements)
    report = build_science_report(
        batch=batch,
        selection=selection,
        config=config,
        batch_dir=output,
        status="success",
        error=None,
        elapsed_seconds=3.0,
        result=result,
        summary=summary,
    )
    payload = report.model_dump(mode="json")
    measurements = science_report_module._extract_calibration_measurements(payload)

    encoded_measurement_slots = 0
    original_encoded_json = science_report_module._encoded_json

    def counting_encoded_json(value):
        nonlocal encoded_measurement_slots
        if isinstance(value, dict):
            encoded_measurement_slots += len(value.get("measurements", []))
        return original_encoded_json(value)

    monkeypatch.setattr(science_report_module, "_encoded_json", counting_encoded_json)
    chunks = science_report_module._chunk_measurements(
        report,
        measurements,
        max_document_bytes=20_000,
        generation="performance-test",
    )

    assert sum(map(len, chunks)) == n_measurements
    assert encoded_measurement_slots == n_measurements
    assert all(
        len(
            original_encoded_json(
                science_report_module._part_payload(
                    report,
                    chunk,
                    part_number=index,
                    parts_total=len(chunks) + 1,
                    generation="performance-test",
                ).model_dump(mode="json")
            )
        )
        <= 20_000
        for index, chunk in enumerate(chunks, start=1)
    )


def test_huntsman_runner_publishes_science_report_and_manifest_link(tmp_path, monkeypatch):
    batch, selection, result, summary, config, output = _products(tmp_path)

    class CompletedRun:
        completed = True
        error_message = None

        @staticmethod
        def to_result():
            return result

        @staticmethod
        def to_summary():
            return summary

    monkeypatch.setattr(
        huntsman_runner,
        "select_batch_frames",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(huntsman_runner, "load_fits_files", lambda _paths: [])
    monkeypatch.setattr(
        huntsman_runner,
        "process_senpai_collect",
        lambda _images, **_kwargs: CompletedRun(),
    )
    monkeypatch.setattr(huntsman_runner, "final_plots", lambda *_args: None)

    manifest = huntsman_runner._process_batch(
        batch,
        output,
        config,
        huntsman_runner.HuntsmanRunOptions(
            sidereal_policy="first",
            skip_blurry_sidereal=False,
        ),
    )

    report_path = Path(manifest["science_report_path"])
    batch_dir = batch_output_directory(output, batch)
    stored_manifest = json.loads((batch_dir / "huntsman_manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["layout_version"] == 2
    assert report_path.is_file()
    assert report_path == batch_dir / SCIENCE_REPORT_FILENAME
    assert Path(manifest["result_path"]) == batch_dir / "senpai" / "result.json"
    assert Path(manifest["summary_path"]) == batch_dir / "senpai" / "summary.json"
    assert (batch_dir / "senpai" / "frames" / "frame_0_sidereal.json").is_file()
    assert (batch_dir / "provenance" / "config.yaml").is_file()
    assert (batch_dir / "provenance" / "command.txt").is_file()
    assert stored_manifest["science_report_id"] == manifest["science_report_id"]
    assert json.loads(report_path.read_text())["status"] == "success"
