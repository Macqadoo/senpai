"""Discover Huntsman FITS files and group them into SENPAI frame batches."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from astropy.io import fits
from astropy.io.fits import Header

from senpai.core.config import AppConfig
from senpai.engine.models.metadata import TrackMode
from senpai.integrations.huntsman.filenames import (
    ParsedHuntsmanFilename,
    fits_stem,
    frame_sort_key,
    is_fits_file,
    parse_huntsman_filename,
)

logger = logging.getLogger(__name__)

DEFAULT_SENSOR_HEADER_KEYS: tuple[str, ...] = (
    "SENSOR",
    "SENSORID",
    "CAMERA",
    "CAMID",
    "INSTRUME",
    "DETECTOR",
)

_RATE_UNIT_TO_ARCSEC_PER_SECOND: dict[str, float] = {
    "arcseconds/second": 1.0,
    "arcsec/second": 1.0,
    "arcsec/s": 1.0,
    "degrees/second": 3600.0,
    "deg/second": 3600.0,
    "deg/s": 3600.0,
    "radians/second": 206264.80624709636,
    "rad/s": 206264.80624709636,
}


@dataclass(frozen=True, slots=True)
class HuntsmanFrameRecord:
    """One FITS path enriched with grouping and routing metadata."""

    path: Path
    parsed: ParsedHuntsmanFilename
    task_id: str
    task_id_source: str
    filter_name: str
    sensor: str
    track_mode: TrackMode
    track_mode_source: str
    norad_id: str | None = None
    mode_disagreement: bool = False

    def to_manifest(self) -> dict:
        return {
            "path": str(self.path),
            "filename_convention": self.parsed.convention,
            "task_id": self.task_id,
            "task_id_source": self.task_id_source,
            "filter": self.filter_name,
            "sensor": self.sensor,
            "track_mode": self.track_mode.value,
            "track_mode_source": self.track_mode_source,
            "filename_track_mode": (
                self.parsed.filename_track_mode.value
                if self.parsed.filename_track_mode is not None
                else None
            ),
            "mode_disagreement": self.mode_disagreement,
            "norad_id": self.norad_id,
        }


@dataclass(slots=True)
class HuntsmanBatch:
    """A task/filter/sensor group suitable for one collect-pipeline call."""

    task_id: str
    task_id_source: str
    filter_name: str
    sensor: str
    norad_id: str | None
    frames: list[HuntsmanFrameRecord] = field(default_factory=list)

    @property
    def sidereal_frames(self) -> list[HuntsmanFrameRecord]:
        return [frame for frame in self.frames if frame.track_mode == TrackMode.SIDEREAL]

    @property
    def rate_frames(self) -> list[HuntsmanFrameRecord]:
        return [frame for frame in self.frames if frame.track_mode == TrackMode.RATE]

    @property
    def output_label(self) -> str:
        timestamps = [frame.parsed.timestamp for frame in self.frames if frame.parsed.timestamp]
        observation = f"obs{min(timestamps).strftime('%y%m%d')}" if timestamps else "obsunknown"
        parts = [
            observation,
            safe_label(self.norad_id, "none"),
            safe_label(self.filter_name, "none"),
        ]
        if self.sensor.casefold() != "unknown":
            parts.append(safe_label(self.sensor, "unknown"))
        parts.append(safe_label(self.task_id, "unknown-task"))
        return "_".join(parts)

    @property
    def batch_id(self) -> str:
        return self.output_label

    def to_manifest(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "task_id": self.task_id,
            "task_id_source": self.task_id_source,
            "filter": self.filter_name,
            "sensor": self.sensor,
            "norad_id": self.norad_id,
            "num_sidereal": len(self.sidereal_frames),
            "num_rate": len(self.rate_frames),
            "mode_disagreements": sum(frame.mode_disagreement for frame in self.frames),
            "frames": [frame.to_manifest() for frame in self.frames],
        }


@dataclass(frozen=True, slots=True)
class DiscoveryIssue:
    path: Path
    reason: str

    def to_manifest(self) -> dict[str, str]:
        return {"path": str(self.path), "reason": self.reason}


@dataclass(slots=True)
class HuntsmanDiscovery:
    batches: list[HuntsmanBatch] = field(default_factory=list)
    skipped: list[DiscoveryIssue] = field(default_factory=list)

    def to_manifest(self) -> dict:
        return {
            "num_batches": len(self.batches),
            "num_skipped_files": len(self.skipped),
            "batches": [batch.to_manifest() for batch in self.batches],
            "skipped_files": [issue.to_manifest() for issue in self.skipped],
        }


def safe_label(value: object, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return re.sub(r"[^A-Za-z0-9.-]+", "-", text or default)


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_value(header: Header, keys: list[str] | tuple[str, ...]) -> str | None:
    for key in keys:
        value = _clean(header.get(key))
        if value is not None:
            return value
    return None


def _header_keys(config: AppConfig, sensor_header_keys: tuple[str, ...]) -> set[str]:
    tracking = config.headers.tracking
    return {
        "SKTASKID",
        "TASKID",
        "NORAD_ID",
        *config.headers.filter_keys,
        *tracking.track_mode_keys,
        *tracking.track_ra_rate_keys,
        *tracking.track_dec_rate_keys,
        *sensor_header_keys,
    }


def _read_discovery_header(
    path: Path,
    config: AppConfig,
    sensor_header_keys: tuple[str, ...],
) -> Header:
    """Read only grouping-related keywords, searching HDUs in file order."""

    keys = _header_keys(config, sensor_header_keys)
    header = Header()
    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdus:
        for hdu in hdus:
            for key in keys:
                if key in header:
                    continue
                value = hdu.header.get(key)
                if _clean(value) is not None:
                    header[key] = value
    return header


def _normalize_header_track_mode(value: str | None) -> TrackMode | None:
    if value is None:
        return None
    lowered = value.casefold()
    has_rate = "rate" in lowered
    has_sidereal = "sidereal" in lowered
    if has_rate and not has_sidereal:
        return TrackMode.RATE
    if has_sidereal and not has_rate:
        return TrackMode.SIDEREAL
    return None


def _rate_value(header: Header, keys: list[str], unit: str) -> float | None:
    value = _first_value(header, keys)
    if value is None:
        return None
    try:
        numeric = float(value)
    except ValueError:
        return None
    factor = _RATE_UNIT_TO_ARCSEC_PER_SECOND.get(unit.strip().casefold())
    if factor is None:
        logger.warning("Unknown tracking-rate unit %r; treating it as arcsec/s", unit)
        factor = 1.0
    return numeric * factor


def _discover_track_mode(
    header: Header,
    parsed: ParsedHuntsmanFilename,
    config: AppConfig,
) -> tuple[TrackMode, str] | None:
    tracking = config.headers.tracking
    header_mode = _normalize_header_track_mode(
        _first_value(header, tracking.track_mode_keys or ["TRKMODE"])
    )
    if header_mode is not None:
        return header_mode, "TRKMODE"

    ra_rate = _rate_value(header, tracking.track_ra_rate_keys, tracking.track_ra_rate_unit)
    dec_rate = _rate_value(header, tracking.track_dec_rate_keys, tracking.track_dec_rate_unit)
    if ra_rate is not None and dec_rate is not None:
        magnitude = math.hypot(ra_rate, dec_rate)
        mode = (
            TrackMode.SIDEREAL
            if magnitude <= tracking.sidereal_rate_threshold_arcsec_per_second
            else TrackMode.RATE
        )
        return mode, "rates"

    if parsed.filename_track_mode is not None:
        return parsed.filename_track_mode, "filename"
    return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _frame_record(
    path: Path,
    config: AppConfig,
    sensor_override: str | None,
    sensor_header_keys: tuple[str, ...],
) -> HuntsmanFrameRecord:
    parsed = parse_huntsman_filename(path)
    header = _read_discovery_header(path, config, sensor_header_keys)

    sktaskid = _first_value(header, ("SKTASKID",))
    taskid = _first_value(header, ("TASKID",))
    if sktaskid is not None:
        resolved_task_id, task_source = sktaskid, "SKTASKID"
    elif taskid is not None:
        resolved_task_id, task_source = taskid, "TASKID"
    elif parsed.filename_task_id is not None:
        resolved_task_id, task_source = parsed.filename_task_id, "filename"
    else:
        raise ValueError("no SKTASKID, TASKID, or recognized filename task ID")

    mode_result = _discover_track_mode(header, parsed, config)
    if mode_result is None:
        raise ValueError("no usable TRKMODE, tracking rates, or filename track mode")
    track_mode, track_source = mode_result

    filter_name = _first_value(header, config.headers.filter_keys) or parsed.filename_filter or "unknown"
    sensor = sensor_override or _first_value(header, sensor_header_keys) or "unknown"
    norad_id = _first_value(header, ("NORAD_ID",)) or parsed.filename_norad_id
    disagreement = (
        parsed.filename_track_mode is not None
        and parsed.filename_track_mode != track_mode
        and track_source != "filename"
    )
    if disagreement:
        logger.warning(
            "Tracking-mode disagreement for %s: filename=%s, %s=%s; trusting %s",
            path,
            parsed.filename_track_mode.value,
            track_source,
            track_mode.value,
            track_source,
        )

    return HuntsmanFrameRecord(
        path=path,
        parsed=parsed,
        task_id=resolved_task_id,
        task_id_source=task_source,
        filter_name=filter_name,
        sensor=sensor,
        track_mode=track_mode,
        track_mode_source=track_source,
        norad_id=norad_id,
        mode_disagreement=disagreement,
    )


def discover_huntsman_batches(
    data_directory: str | Path,
    config: AppConfig,
    *,
    output_directory: str | Path | None = None,
    sensor: str | None = None,
    sensor_header_keys: tuple[str, ...] = DEFAULT_SENSOR_HEADER_KEYS,
) -> HuntsmanDiscovery:
    """Recursively discover and group Huntsman frames.

    Groups are separated by task identity, filter and sensor. Filename-only
    task identities also include date and NORAD in their grouping key because
    the legacy eight-character token is not globally unique.
    """

    data_root = Path(data_directory).resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Huntsman input directory not found: {data_root}")
    output_root = Path(output_directory).resolve() if output_directory is not None else None

    records_by_key: dict[tuple[str, ...], list[HuntsmanFrameRecord]] = defaultdict(list)
    skipped: list[DiscoveryIssue] = []

    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or not is_fits_file(path):
            continue
        if output_root is not None and output_root != data_root and _is_relative_to(path, output_root):
            continue
        parsed = parse_huntsman_filename(path)
        if parsed.path.name.startswith(".") or fits_stem(path).casefold().endswith("_processed"):
            continue

        try:
            record = _frame_record(path, config, sensor, sensor_header_keys)
        except Exception as exc:
            skipped.append(DiscoveryIssue(path=path, reason=str(exc)))
            logger.warning("Skipping Huntsman FITS %s: %s", path, exc)
            continue

        key: tuple[str, ...] = (
            record.task_id.casefold(),
            record.filter_name.casefold(),
            record.sensor.casefold(),
        )
        if record.task_id_source == "filename":
            date_scope = record.parsed.timestamp.strftime("%Y%m%d") if record.parsed.timestamp else "unknown-date"
            key += (date_scope, (record.norad_id or "unknown-norad").casefold())
        records_by_key[key].append(record)

    source_rank = {"SKTASKID": 0, "TASKID": 1, "filename": 2}
    batches: list[HuntsmanBatch] = []
    for records in records_by_key.values():
        records.sort(key=lambda record: frame_sort_key(record.parsed))
        strongest = min(records, key=lambda record: source_rank[record.task_id_source])
        batches.append(
            HuntsmanBatch(
                task_id=strongest.task_id,
                task_id_source=strongest.task_id_source,
                filter_name=records[0].filter_name,
                sensor=records[0].sensor,
                norad_id=next((record.norad_id for record in records if record.norad_id), None),
                frames=records,
            )
        )

    batches.sort(key=lambda batch: batch.output_label)
    return HuntsmanDiscovery(batches=batches, skipped=skipped)
