"""Build MongoDB-ready science reports for Huntsman SENPAI batches.

The report is intentionally a derived data product: it reorganises the full
SENPAI result, compact summary, Huntsman manifest and source FITS headers into
a stable schema without recomputing any science measurements.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy.io import fits
from pydantic import BaseModel, ConfigDict, Field

from senpai.core.config import AppConfig
from senpai.engine.models.senpai import SenpaiRunResult, SenpaiRunSummary
from senpai.integrations.huntsman.batches import HuntsmanBatch, HuntsmanFrameRecord
from senpai.integrations.huntsman.quality import HuntsmanBatchSelection

logger = logging.getLogger(__name__)

SCIENCE_REPORT_SCHEMA_VERSION = 1
SCIENCE_REPORT_FILENAME = "science_report.json"
# MongoDB's BSON document limit is 16 MiB. JSON and BSON sizes are not
# identical, so retain a conservative margin for NiFi/BSON conversion.
MONGO_SAFE_DOCUMENT_BYTES = 14_000_000
_HASH_CHUNK_BYTES = 1024 * 1024
_LEGACY_CALIBRATION_ARRAYS = frozenset(
    {
        "stars_mag",
        "stars_snr",
        "stars_zp_offset",
        "stars_isolated",
        "stars_catalog_id",
    }
)
ScienceStatus = Literal["success", "error", "skipped_blurry", "skipped_empty"]


class ScienceReportError(RuntimeError):
    """Raised when a science report cannot satisfy its output contract."""


class ScienceReport(BaseModel):
    """One run-level document for a Huntsman SENPAI batch."""

    model_config = ConfigDict(extra="forbid")

    report_type: Literal["senpai_science"] = "senpai_science"
    schema_version: Literal[1] = SCIENCE_REPORT_SCHEMA_VERSION
    document_type: Literal["run"] = "run"
    document_key: str
    science_report_id: str
    run_key: str
    generated_at_utc: str
    status: ScienceStatus
    success: bool
    error: str | None = None
    batch: dict[str, Any]
    observation: dict[str, Any]
    detector: dict[str, Any]
    processing: dict[str, Any]
    input_fingerprint: str
    input_frames: list[dict[str, Any]] = Field(default_factory=list)
    frames: list[dict[str, Any]] = Field(default_factory=list)
    correlated_streaks: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    product_fingerprint: str
    calibration_measurement_count: int = 0
    calibration_measurements_embedded: bool = True
    measurement_parts: list[dict[str, Any]] = Field(default_factory=list)
    part_number: int = Field(default=0, ge=0)
    parts_total: int = Field(default=1, ge=1)


class ScienceMeasurementPart(BaseModel):
    """A MongoDB-size-safe continuation document for calibration stars."""

    model_config = ConfigDict(extra="forbid")

    report_type: Literal["senpai_science"] = "senpai_science"
    schema_version: Literal[1] = SCIENCE_REPORT_SCHEMA_VERSION
    document_type: Literal["calibration_measurements"] = "calibration_measurements"
    document_key: str
    science_report_id: str
    run_key: str
    generated_at_utc: str
    status: ScienceStatus
    part_number: int = Field(ge=1)
    parts_total: int = Field(ge=2)
    measurement_count: int = Field(ge=1)
    measurements: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ScienceReportWriteResult:
    root_path: Path
    part_paths: tuple[Path, ...]
    science_report_id: str
    run_key: str
    document_count: int
    calibration_measurement_count: int

    def to_manifest(self) -> dict[str, Any]:
        return {
            "science_report_path": str(self.root_path),
            "science_report_part_paths": [str(path) for path in self.part_paths],
            "science_report_schema_version": SCIENCE_REPORT_SCHEMA_VERSION,
            "science_report_id": self.science_report_id,
            "science_report_run_key": self.run_key,
            "science_report_document_count": self.document_count,
            "calibration_measurement_count": self.calibration_measurement_count,
        }


def _json_safe(value: Any) -> Any:
    """Convert common scientific/Python values into strict JSON values."""

    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return moment.astimezone(UTC).isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _encoded_json(payload: Any) -> bytes:
    """Serialize strict JSON and fail if any unsafe value survived conversion."""

    return (
        json.dumps(
            _json_safe(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    """Resolve immutable source provenance without requiring the git binary."""

    for key in ("SENPAI_GIT_COMMIT", "GIT_COMMIT", "CI_COMMIT_SHA"):
        if value := os.environ.get(key):
            return value.strip() or None

    repository = Path(__file__).resolve().parents[3]
    head_path = repository / ".git" / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head or None
        reference = head.removeprefix("ref: ")
        loose_ref = repository / ".git" / reference
        if loose_ref.is_file():
            return loose_ref.read_text(encoding="utf-8").strip() or None
        packed_refs = repository / ".git" / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if not line.startswith(("#", "^")):
                    commit, _, name = line.partition(" ")
                    if name == reference:
                        return commit
    except OSError:
        return None
    return None


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encoded_json(payload)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _header_value(hdus: list[dict[str, Any]], keyword: str) -> Any:
    target = keyword.casefold()
    for hdu in hdus:
        for card in hdu["cards"]:
            if str(card["keyword"]).casefold() == target:
                return card["value"]
    return None


def _read_fits_headers(path: Path) -> list[dict[str, Any]]:
    """Retain every HDU header card, including comments and duplicates."""

    documents: list[dict[str, Any]] = []
    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdus:
        for index, hdu in enumerate(hdus):
            cards = [
                {
                    "keyword": str(card.keyword),
                    "value": _json_safe(card.value),
                    "comment": str(card.comment) if card.comment else None,
                }
                for card in hdu.header.cards
            ]
            documents.append(
                {
                    "hdu_index": index,
                    "hdu_name": str(hdu.name),
                    "hdu_type": type(hdu).__name__,
                    "cards": cards,
                }
            )
    return documents


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _input_frame_document(
    record: HuntsmanFrameRecord,
    *,
    selected: bool,
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    path = record.path.resolve()
    stat = path.stat()
    headers = _read_fits_headers(path)
    sha256 = _file_sha256(path)
    exposure = _header_value(headers, "EXPTIME")
    try:
        exposure = float(exposure) if exposure is not None else None
    except (TypeError, ValueError):
        exposure = None
    date_obs = _header_value(headers, "DATE-OBS")
    date_end = _header_value(headers, "DATE-END")
    archive_id = _header_value(headers, "ARCHIVEID") or _header_value(headers, "ARCHIVE_ID")
    return {
        "input_frame_id": f"fits:{sha256}",
        "filename": path.name,
        "source_path": str(path),
        "archive_id": str(archive_id) if archive_id is not None else None,
        "size_bytes": stat.st_size,
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "sha256": sha256,
        "selected_for_processing": selected,
        "identity": record.to_manifest(),
        "observation": {
            "date_obs": str(date_obs) if date_obs is not None else None,
            "date_end": str(date_end) if date_end is not None else None,
            "exposure_seconds": exposure,
            "ra_degrees": _json_safe(_header_value(headers, "RA")),
            "dec_degrees": _json_safe(_header_value(headers, "DEC")),
            "ra_rate": _json_safe(_header_value(headers, "RA_RATE")),
            "dec_rate": _json_safe(_header_value(headers, "DEC_RATE")),
        },
        "quality_precheck": quality,
        "fits_headers": headers,
    }


def _observation_window(inputs: list[dict[str, Any]]) -> dict[str, Any]:
    starts: list[datetime] = []
    ends: list[datetime] = []
    for frame in inputs:
        observation = frame["observation"]
        start = _parse_utc(observation.get("date_obs"))
        end = _parse_utc(observation.get("date_end"))
        if start is not None:
            starts.append(start)
            if end is None and observation.get("exposure_seconds") is not None:
                end = start + timedelta(seconds=float(observation["exposure_seconds"]))
        if end is not None:
            ends.append(end)
    first = min(starts) if starts else None
    last = max(ends or starts) if (ends or starts) else None
    return {
        "night": first.date().isoformat() if first is not None else None,
        "time_start_utc": first.isoformat() if first is not None else None,
        "time_end_utc": last.isoformat() if last is not None else None,
    }


def _artifact_type(path: Path) -> str:
    name = path.name.casefold()
    if name.startswith("senpai_") and name.endswith("_summary.json"):
        return "senpai_summary"
    if name.startswith("senpai_") and name.endswith(".json"):
        return "senpai_result"
    if name == "huntsman_manifest.json":
        return "huntsman_manifest"
    if name == "config.yaml":
        return "configuration"
    if name == "command.txt":
        return "command"
    if name.startswith("frame_") and name.endswith(".json"):
        return "frame_quicklook"
    if name.endswith((".fits", ".fit", ".fts", ".fits.fz")):
        return "processed_fits"
    if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".pdf"}:
        return "plot"
    return "artifact"


def _artifact_documents(batch_dir: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(batch_dir.rglob("*")):
        if (
            not path.is_file()
            or path.name in {SCIENCE_REPORT_FILENAME, "huntsman_manifest.json"}
            or path.name.endswith(".tmp")
        ):
            continue
        artifacts.append(
            {
                "artifact_type": _artifact_type(path),
                "relative_path": str(path.relative_to(batch_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return artifacts


def _input_for_result_frame(
    result_frame: dict[str, Any],
    input_frames: list[dict[str, Any]],
) -> dict[str, Any] | None:
    original = result_frame.get("original_frame_path")
    if original:
        wanted = Path(original)
        for input_frame in input_frames:
            candidate = Path(input_frame["source_path"])
            if candidate == wanted or candidate.name == wanted.name:
                return input_frame
    return None


def _measurement_id(frame_id: str, source_key: Any, ordinal: int) -> str:
    digest = _canonical_hash({"frame_id": frame_id, "source_key": source_key, "ordinal": ordinal})
    return f"measurement:{digest[:32]}"


def _detection_documents(
    frame_id: str,
    detections: list[dict[str, Any]],
    detection_type: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for ordinal, detection in enumerate(detections):
        document = dict(detection)
        key = {
            "frame_id": frame_id,
            "type": detection_type,
            "ordinal": ordinal,
            "x": detection.get("x"),
            "y": detection.get("y"),
        }
        document.update(
            {
                "measurement_id": f"detection:{_canonical_hash(key)[:32]}",
                "frame_id": frame_id,
                "measurement_type": detection_type,
            }
        )
        output.append(document)
    return output


def _science_frames(
    result: SenpaiRunResult | None,
    summary: SenpaiRunSummary | None,
    input_frames: list[dict[str, Any]],
    science_report_id: str,
) -> tuple[list[dict[str, Any]], int]:
    if result is None or summary is None:
        return [], 0

    result_payload = result.model_dump()
    result_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for frame in result_payload.get("sidereal_frames", []):
        result_by_key[("sidereal", int(frame["index"]))] = frame
    for frame in result_payload.get("rate_track_frames", []):
        result_by_key[("rate", int(frame["index"]))] = frame

    science_frames: list[dict[str, Any]] = []
    calibration_count = 0
    for summary_frame in summary.model_dump().get("frames", []):
        track_mode = summary_frame.get("track_mode") or "unknown"
        index = int(summary_frame["index"])
        full = result_by_key.get((track_mode, index), {})
        frame_id = f"{science_report_id}:frame:{index}:{track_mode}"
        input_frame = _input_for_result_frame(full, input_frames)

        photometry = dict(full.get("photometry_summary") or summary_frame.get("photometry_summary") or {})
        measurements = list(photometry.pop("star_measurements", None) or [])
        if measurements:
            for legacy_key in _LEGACY_CALIBRATION_ARRAYS:
                photometry.pop(legacy_key, None)
        enriched_measurements: list[dict[str, Any]] = []
        for ordinal, measurement in enumerate(measurements):
            item = dict(measurement)
            item.update(
                {
                    "measurement_id": _measurement_id(frame_id, measurement.get("source_key"), ordinal),
                    "frame_id": frame_id,
                    "frame_index": index,
                    "track_mode": track_mode,
                }
            )
            enriched_measurements.append(item)
        calibration_count += len(enriched_measurements)
        if enriched_measurements:
            photometry["star_measurements"] = enriched_measurements
        photometry["calibration_measurement_count"] = len(enriched_measurements)

        detections = _detection_documents(
            frame_id,
            list(summary_frame.get("detections") or []),
            "science_detection",
        )
        streak_candidates = _detection_documents(
            frame_id,
            list(summary_frame.get("streak_candidates") or []),
            "streak_candidate",
        )
        starfield = full.get("starfield") or {}
        fwhm_stats = starfield.get("fwhm_stats") or {}
        seeing = summary_frame.get("seeing")
        input_observation = input_frame.get("observation", {}) if input_frame else {}
        quality = {
            "wcs_status": summary_frame.get("wcs_status"),
            "wcs_quality": summary_frame.get("wcs_quality"),
            "solver_tier": summary_frame.get("solver_tier"),
            "solve_ms": summary_frame.get("solve_ms"),
            "seeing": seeing,
            "median_fwhm_pixels": fwhm_stats.get("median_fwhm"),
            "detected_source_count": len(starfield.get("detections") or []),
            "catalog_source_count": summary_frame.get("num_catalog_stars"),
            "astrometric_fit_source_count": summary_frame.get("num_astrometric_fit_stars"),
            "median_background_adu": photometry.get("median_background"),
            "limiting_magnitude": summary_frame.get("limiting_magnitude"),
            "saturation_level_adu": (_header_value(input_frame["fits_headers"], "SATURATE") if input_frame else None),
            "precheck": input_frame.get("quality_precheck") if input_frame else None,
        }

        science_frame = dict(summary_frame)
        science_frame.pop("photometry_summary", None)
        science_frame["detections"] = detections
        science_frame["streak_candidates"] = streak_candidates
        science_frame.update(
            {
                "frame_id": frame_id,
                "input_frame_id": input_frame.get("input_frame_id") if input_frame else None,
                "archive_id": input_frame.get("archive_id") if input_frame else None,
                "observation": input_observation,
                "quality": quality,
                "photometry": photometry or None,
                "processing": {
                    "original_frame_path": full.get("original_frame_path"),
                    "processed_frame_path": full.get("processed_frame_path"),
                    "history": full.get("processing_history"),
                },
            }
        )
        science_frames.append(_json_safe(science_frame))

    science_frames.sort(key=lambda frame: (frame["index"], frame["track_mode"] or ""))
    return science_frames, calibration_count


def _correlated_streak_documents(
    summary: SenpaiRunSummary | None,
    science_report_id: str,
    requested_norad_id: str | None,
) -> list[dict[str, Any]]:
    if summary is None:
        return []
    output: list[dict[str, Any]] = []
    for ordinal, streak in enumerate(summary.model_dump().get("correlated_streaks", [])):
        document = dict(streak)
        streak_key = streak.get("streak_id") or ordinal
        document.update(
            {
                "measurement_id": (
                    f"correlated-streak:{_canonical_hash({'report': science_report_id, 'streak': streak_key})[:32]}"
                ),
                "requested_norad_id": requested_norad_id,
                "matched_norad_id": streak.get("matched_norad_id"),
            }
        )
        output.append(_json_safe(document))
    return output


def build_science_report(
    *,
    batch: HuntsmanBatch,
    selection: HuntsmanBatchSelection,
    config: AppConfig,
    batch_dir: Path,
    status: ScienceStatus,
    error: str | None,
    elapsed_seconds: float | None,
    result: SenpaiRunResult | None = None,
    summary: SenpaiRunSummary | None = None,
    terminal_manifest: dict[str, Any] | None = None,
) -> ScienceReport:
    """Build and validate a complete run-level science report."""

    selected_paths = {path.resolve() for path in selection.selected_files}
    quality_by_path = {quality.path.resolve(): quality.to_manifest() for quality in selection.quality}
    input_frames = [
        _input_frame_document(
            frame,
            selected=frame.path.resolve() in selected_paths,
            quality=quality_by_path.get(frame.path.resolve()),
        )
        for frame in batch.frames
    ]
    input_identity = [
        {
            "filename": frame["filename"],
            "size_bytes": frame["size_bytes"],
            "sha256": frame["sha256"],
        }
        for frame in input_frames
    ]
    input_fingerprint = f"sha256:{_canonical_hash(input_identity)}"
    config_payload = config.model_dump(mode="json")
    config_hash = f"sha256:{_canonical_hash(config_payload)}"
    git_commit = _git_commit()
    senpai_version = result.senpai_version if result is not None else None
    run_identity = {
        "batch_id": batch.batch_id,
        "task_id": batch.task_id,
        "controller_task_id": batch.controller_task_id,
        "sktask_id": batch.sktask_id,
        "camera_id": batch.camera_id,
        "sensor_id": batch.sensor_id,
        "filter": batch.filter_name,
        "target_id": batch.target_id,
        "norad_id": batch.norad_id,
        "input_fingerprint": input_fingerprint,
        "senpai_version": senpai_version,
        "senpai_git_commit": git_commit,
        "configuration_hash": config_hash,
    }
    run_digest = _canonical_hash(run_identity)
    run_key = f"sha256:{run_digest}"
    science_report_id = f"senpai-science:{run_digest[:32]}"

    frames, calibration_count = _science_frames(result, summary, input_frames, science_report_id)
    artifacts = _artifact_documents(batch_dir)
    product_fingerprint = f"sha256:{_canonical_hash(artifacts)}"
    batch_manifest = batch.to_manifest()
    batch_manifest.pop("frames", None)
    batch_manifest["metadata_sources"] = {
        "task_id": batch.task_id_source,
        "controller_task_id": "TASKID" if batch.controller_task_id else None,
        "sktask_id": "SKTASKID" if batch.sktask_id else None,
        "camera_id": batch.camera_id_source,
        "sensor_id": batch.sensor_id_source,
        "filter": batch.filter_source,
        "target_id": batch.target_id_source,
        "target_type": batch.target_type_source,
        "norad_id": batch.norad_id_source,
        "gain_e_per_adu": batch.gain_source,
        "read_noise_electrons": batch.read_noise_source,
    }
    generated = datetime.now(UTC).isoformat()
    report = ScienceReport(
        document_key=f"{science_report_id}:run",
        science_report_id=science_report_id,
        run_key=run_key,
        generated_at_utc=generated,
        status=status,
        success=status == "success" and bool(result and result.completed),
        error=error,
        batch=batch_manifest,
        observation=_observation_window(input_frames),
        detector={
            "sensor": batch.sensor,
            "camera_id": batch.camera_id,
            "camera_id_source": batch.camera_id_source,
            "sensor_id": batch.sensor_id,
            "sensor_id_source": batch.sensor_id_source,
            "gain_e_per_adu": batch.gain_e_per_adu,
            "gain_source": batch.gain_source,
            "read_noise_electrons": batch.read_noise_electrons,
            "read_noise_source": batch.read_noise_source,
        },
        processing={
            "senpai_version": senpai_version,
            "senpai_git_commit": git_commit,
            "container_digest": os.environ.get("SENPAI_CONTAINER_DIGEST"),
            "configuration_hash": config_hash,
            "configuration_artifact": "config.yaml",
            "command_artifact": "command.txt",
            "completed": result.completed if result is not None else False,
            "compute_seconds": result.compute_seconds if result is not None else None,
            "elapsed_seconds": elapsed_seconds,
            "scale_factor": result.scale_factor if result is not None else None,
            "collection_metadata": (result.collect_metadata.model_dump(mode="json") if result is not None else None),
            "selection": selection.to_manifest(),
            "terminal_manifest": terminal_manifest,
        },
        input_fingerprint=input_fingerprint,
        input_frames=input_frames,
        frames=frames,
        correlated_streaks=_correlated_streak_documents(summary, science_report_id, batch.norad_id),
        artifacts=artifacts,
        product_fingerprint=product_fingerprint,
        calibration_measurement_count=calibration_count,
    )
    # A strict serialization pass is part of the contract: no NaN/Infinity or
    # values that NiFi's JSON reader cannot represent are allowed through.
    _encoded_json(report.model_dump(mode="json"))
    return report


def _extract_calibration_measurements(
    root_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for frame in root_payload["frames"]:
        photometry = frame.get("photometry")
        if not photometry:
            continue
        measurements.extend(photometry.pop("star_measurements", None) or [])
    return measurements


def _part_payload(
    report: ScienceReport,
    measurements: list[dict[str, Any]],
    part_number: int,
    parts_total: int,
    generation: str,
) -> ScienceMeasurementPart:
    return ScienceMeasurementPart(
        document_key=(f"{report.science_report_id}:calibration:{generation}:{part_number:04d}"),
        science_report_id=report.science_report_id,
        run_key=report.run_key,
        generated_at_utc=report.generated_at_utc,
        status=report.status,
        part_number=part_number,
        parts_total=parts_total,
        measurement_count=len(measurements),
        measurements=measurements,
    )


def _chunk_measurements(
    report: ScienceReport,
    measurements: list[dict[str, Any]],
    max_document_bytes: int,
    generation: str,
) -> list[list[dict[str, Any]]]:
    """Split measurements without repeatedly serializing a growing chunk.

    A one-measurement probe gives the encoded contribution of that
    measurement to a continuation document.  Summing those contributions is
    conservative: pretty-printed JSON is two bytes smaller for every item
    after the first because adjacent list items share a separator.  The old
    implementation serialized the entire candidate list for every item,
    making large catalog results quadratic in both copied and encoded data.
    """

    if not measurements:
        return []

    # Continuation documents require at least one measurement, so derive the
    # fixed wrapper size from a valid one-item model and then empty its dumped
    # representation solely for byte accounting.
    empty_payload = _part_payload(report, [measurements[0]], 999_999, 999_999, generation).model_dump(mode="json")
    empty_payload["measurement_count"] = 0
    empty_payload["measurements"] = []
    empty_size = len(_encoded_json(empty_payload))
    if empty_size > max_document_bytes:
        raise ScienceReportError("science report part metadata exceeds the safe MongoDB document size")

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    estimated_size = empty_size
    for measurement in measurements:
        single_probe = _part_payload(report, [measurement], 999_999, 999_999, generation)
        contribution = len(_encoded_json(single_probe.model_dump(mode="json"))) - empty_size
        if empty_size + contribution > max_document_bytes:
            raise ScienceReportError("one calibration measurement exceeds the safe MongoDB document size")

        if current and estimated_size + contribution > max_document_bytes:
            chunks.append(current)
            current = []
            estimated_size = empty_size

        current.append(measurement)
        estimated_size += contribution
    if current:
        chunks.append(current)
    return chunks


def _clear_old_parts(parts_root: Path, *, keep: Path | None = None) -> None:
    if not parts_root.exists():
        return
    for path in parts_root.rglob(SCIENCE_REPORT_FILENAME):
        if keep is not None and keep in path.parents:
            continue
        path.unlink()
    for directory in sorted(
        (path for path in parts_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        with suppress(OSError):
            directory.rmdir()
    with suppress(OSError):
        parts_root.rmdir()


def write_science_report(
    report: ScienceReport,
    batch_dir: str | Path,
    *,
    max_document_bytes: int = MONGO_SAFE_DOCUMENT_BYTES,
) -> ScienceReportWriteResult:
    """Atomically publish a report and size-safe linked continuation parts."""

    if max_document_bytes <= 0:
        raise ValueError("max_document_bytes must be positive")
    output_dir = Path(batch_dir)
    root_path = output_dir / SCIENCE_REPORT_FILENAME
    parts_root = output_dir / "science_report_parts"
    full_payload = report.model_dump(mode="json")
    full_size = len(_encoded_json(full_payload))

    if full_size <= max_document_bytes:
        _atomic_write(root_path, full_payload)
        _clear_old_parts(parts_root)
        return ScienceReportWriteResult(
            root_path=root_path,
            part_paths=(),
            science_report_id=report.science_report_id,
            run_key=report.run_key,
            document_count=1,
            calibration_measurement_count=report.calibration_measurement_count,
        )

    root_payload = report.model_dump(mode="json")
    measurements = _extract_calibration_measurements(root_payload)
    if not measurements:
        raise ScienceReportError(f"science report is {full_size} bytes without splittable calibration measurements")
    logger.info(
        "Science report is %.1f MiB; splitting %d calibration measurements into MongoDB-safe documents",
        full_size / (1024 * 1024),
        len(measurements),
    )
    generation = report.product_fingerprint.removeprefix("sha256:")[:16]
    chunks = _chunk_measurements(report, measurements, max_document_bytes, generation)
    parts_total = len(chunks) + 1
    logger.info("Writing science report root and %d calibration measurement parts", len(chunks))
    part_models = [
        _part_payload(report, chunk, index, parts_total, generation) for index, chunk in enumerate(chunks, start=1)
    ]
    generation_root = parts_root / generation
    part_paths = tuple(
        generation_root / f"part_{part.part_number:04d}" / SCIENCE_REPORT_FILENAME for part in part_models
    )
    root_payload.update(
        {
            "calibration_measurements_embedded": False,
            "measurement_parts": [
                {
                    "document_key": part.document_key,
                    "part_number": part.part_number,
                    "measurement_count": part.measurement_count,
                    "relative_path": str(path.relative_to(output_dir)),
                }
                for part, path in zip(part_models, part_paths, strict=True)
            ],
            "parts_total": parts_total,
        }
    )
    root_size = len(_encoded_json(root_payload))
    if root_size > max_document_bytes:
        raise ScienceReportError(f"science report root remains {root_size} bytes after calibration chunking")

    for part, path in zip(part_models, part_paths, strict=True):
        payload = part.model_dump(mode="json")
        if len(_encoded_json(payload)) > max_document_bytes:
            raise ScienceReportError(f"science report part {part.part_number} exceeds size limit")
        _atomic_write(path, payload)
    # Publish the root last; its presence means every referenced part exists.
    _atomic_write(root_path, root_payload)
    _clear_old_parts(parts_root, keep=generation_root)
    return ScienceReportWriteResult(
        root_path=root_path,
        part_paths=part_paths,
        science_report_id=report.science_report_id,
        run_key=report.run_key,
        document_count=parts_total,
        calibration_measurement_count=report.calibration_measurement_count,
    )
