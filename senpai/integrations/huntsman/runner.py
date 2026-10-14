"""Batch execution for the Huntsman adapter.

The CLI is intentionally thin; discovery, quality selection, persistence and
status handling live here so a future service worker can call the same code.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from senpai.cli.common import save_run_metadata, write_frame_quicklooks, write_json
from senpai.core.config import AppConfig
from senpai.engine.processing.collect import final_plots, process_senpai_collect
from senpai.engine.utils.file_io import load_fits_files
from senpai.integrations.huntsman.batches import HuntsmanBatch, HuntsmanDiscovery, safe_label
from senpai.integrations.huntsman.quality import (
    DEFAULT_MAX_SIDEREAL_FWHM,
    DEFAULT_MIN_SIDEREAL_SOURCES,
    SiderealPolicy,
    select_batch_frames,
)
from senpai.integrations.huntsman.science_report import (
    build_science_report,
    write_science_report,
)

logger = logging.getLogger(__name__)
HUNTSMAN_LAYOUT_VERSION = 2


@dataclass(frozen=True, slots=True)
class HuntsmanRunOptions:
    sidereal_policy: SiderealPolicy = "sharpest"
    skip_existing: bool = True
    skip_blurry_sidereal: bool = True
    max_sidereal_fwhm: float = DEFAULT_MAX_SIDEREAL_FWHM
    min_sidereal_sources: int = DEFAULT_MIN_SIDEREAL_SOURCES

    def to_manifest(self) -> dict:
        return {
            "sidereal_policy": self.sidereal_policy,
            "skip_existing": self.skip_existing,
            "skip_blurry_sidereal": self.skip_blurry_sidereal,
            "max_sidereal_fwhm": self.max_sidereal_fwhm,
            "min_sidereal_sources": self.min_sidereal_sources,
        }


def batch_relative_directory(batch: HuntsmanBatch) -> Path:
    """Return the stable v2 directory for one SensorKit/filter batch."""

    sensorkit_task_id = batch.sktask_id or batch.task_id
    return (
        Path("batches")
        / safe_label(sensorkit_task_id, "unknown-task")
        / safe_label(batch.filter_name, "unknown-filter")
    )


def batch_output_directory(output_root: str | Path, batch: HuntsmanBatch) -> Path:
    """Resolve a batch below a task/camera product root."""

    return Path(output_root) / batch_relative_directory(batch)


def _validate_unique_batch_directories(discovery: HuntsmanDiscovery) -> None:
    """Reject mixed-camera inputs that would share a task/filter directory."""

    claimed: dict[Path, HuntsmanBatch] = {}
    for batch in discovery.batches:
        relative = batch_relative_directory(batch)
        previous = claimed.get(relative)
        if previous is not None:
            raise ValueError(
                "Multiple Huntsman batches map to the same output directory "
                f"{relative}: cameras {previous.camera_id!r} and "
                f"{batch.camera_id!r}. Use a separate task/camera product root "
                "for each invocation."
            )
        claimed[relative] = batch


def _completed_manifest(batch_dir: Path) -> dict | None:
    """Load a successful terminal manifest, or return ``None``."""

    manifest_path = batch_dir / "huntsman_manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("status") == "success":
                return manifest
        except (OSError, ValueError):
            pass
    return None


def completed_product_exists(batch: HuntsmanBatch, batch_dir: Path) -> bool:
    """Return whether a batch directory contains a completed Huntsman product."""

    if _completed_manifest(batch_dir) is not None:
        return True
    return (batch_dir / "senpai" / "result.json").is_file() or (batch_dir / f"senpai_{batch.batch_id}.json").is_file()


def has_pending_batches(
    discovery: HuntsmanDiscovery,
    output_directory: str | Path,
    *,
    skip_existing: bool = True,
) -> bool:
    """Check whether an invocation will need the science-pipeline dependencies."""

    if not discovery.batches:
        return False
    if not skip_existing:
        return True
    output_root = Path(output_directory)
    return any(
        not completed_product_exists(batch, batch_output_directory(output_root, batch)) for batch in discovery.batches
    )


def _publish_science_report(
    *,
    batch: HuntsmanBatch,
    selection,
    batch_dir: Path,
    config: AppConfig,
    manifest: dict,
    elapsed_seconds: float | None,
    result=None,
    summary=None,
) -> dict:
    """Write the stable NiFi/MongoDB science product and enrich the manifest."""

    logger.info("Building science report for Huntsman batch %s", batch.batch_id)
    report = build_science_report(
        batch=batch,
        selection=selection,
        config=config,
        batch_dir=batch_dir,
        status=manifest["status"],
        error=manifest.get("error"),
        elapsed_seconds=elapsed_seconds,
        result=result,
        summary=summary,
        terminal_manifest=manifest,
    )
    logger.info(
        "Publishing science report for Huntsman batch %s (%d calibration measurements)",
        batch.batch_id,
        report.calibration_measurement_count,
    )
    written = write_science_report(report, batch_dir)
    logger.info(
        "Published science report for Huntsman batch %s as %d linked document(s)",
        batch.batch_id,
        written.document_count,
    )
    return {**manifest, **written.to_manifest()}


def _process_batch(
    batch: HuntsmanBatch,
    output_root: Path,
    config: AppConfig,
    options: HuntsmanRunOptions,
) -> dict:
    batch_dir = batch_output_directory(output_root, batch)
    provenance_dir = batch_dir / "provenance"
    senpai_dir = batch_dir / "senpai"
    frames_dir = senpai_dir / "frames"
    artifacts_dir = batch_dir / "artifacts"
    base_manifest = {
        **batch.to_manifest(),
        "layout_version": HUNTSMAN_LAYOUT_VERSION,
        "relative_output_dir": str(batch_relative_directory(batch)),
        "legacy_output_label": batch.output_label,
        "output_dir": str(batch_dir),
        "options": options.to_manifest(),
    }

    existing_manifest = _completed_manifest(batch_dir) if options.skip_existing else None
    if options.skip_existing and (existing_manifest or completed_product_exists(batch, batch_dir)):
        logger.info("Skipping existing Huntsman batch %s", batch.batch_id)
        return {
            **(existing_manifest or {}),
            **base_manifest,
            "existing_status": (existing_manifest or {}).get("status"),
            "status": "skipped_existing",
            "error": None,
        }

    # RuntimeConfig declares this field as a string. Keep that contract when
    # assigning a Path-derived batch directory so model_dump(mode="json") does
    # not emit a Pydantic unexpected-value warning during report generation.
    config.runtime.output_dir = str(batch_dir)
    config.runtime.run_id = batch.batch_id
    selection = select_batch_frames(
        batch,
        policy=options.sidereal_policy,
        skip_blurry_sidereal=options.skip_blurry_sidereal,
        max_fwhm=options.max_sidereal_fwhm,
        min_sources=options.min_sidereal_sources,
    )
    selection_manifest = selection.to_manifest()
    if selection.rejected_for_blur:
        logger.warning("Skipping blurry Huntsman batch %s", batch.batch_id)
        batch_dir.mkdir(parents=True, exist_ok=True)
        provenance_dir.mkdir(parents=True, exist_ok=True)
        save_run_metadata(provenance_dir, "senpai.cli.huntsman", config)
        manifest = {
            **base_manifest,
            **selection_manifest,
            "status": "skipped_blurry",
            "error": "no sidereal frame passed the FWHM quality gate",
        }
        try:
            manifest = _publish_science_report(
                batch=batch,
                selection=selection,
                batch_dir=batch_dir,
                config=config,
                manifest=manifest,
                elapsed_seconds=0.0,
            )
        except Exception as exc:
            logger.exception("Science report failed for skipped batch %s", batch.batch_id)
            manifest["science_report_error"] = str(exc)
        write_json(batch_dir / "huntsman_manifest.json", manifest)
        return manifest
    if not selection.selected_files:
        batch_dir.mkdir(parents=True, exist_ok=True)
        provenance_dir.mkdir(parents=True, exist_ok=True)
        save_run_metadata(provenance_dir, "senpai.cli.huntsman", config)
        manifest = {
            **base_manifest,
            **selection_manifest,
            "status": "skipped_empty",
            "error": "batch contains no usable frames",
        }
        try:
            manifest = _publish_science_report(
                batch=batch,
                selection=selection,
                batch_dir=batch_dir,
                config=config,
                manifest=manifest,
                elapsed_seconds=0.0,
            )
        except Exception as exc:
            logger.exception("Science report failed for empty batch %s", batch.batch_id)
            manifest["science_report_error"] = str(exc)
        write_json(batch_dir / "huntsman_manifest.json", manifest)
        return manifest

    for directory in (batch_dir, provenance_dir, senpai_dir, frames_dir, artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pending_manifest = {
        **base_manifest,
        **selection_manifest,
        "status": "running",
        "error": None,
    }
    write_json(batch_dir / "huntsman_manifest.json", pending_manifest)

    save_run_metadata(provenance_dir, "senpai.cli.huntsman", config)

    started = time.monotonic()
    result = None
    summary = None
    try:
        images = load_fits_files(list(selection.selected_files))
        senpai_run = process_senpai_collect(images, id=batch.batch_id)

        result = senpai_run.to_result()
        result_path = senpai_dir / "result.json"
        write_json(result_path, result.model_dump())

        summary = senpai_run.to_summary()
        summary_path = senpai_dir / "summary.json"
        write_json(summary_path, summary.model_dump())
        write_frame_quicklooks(summary, frames_dir)

        try:
            final_plots(senpai_run, artifacts_dir)
        except Exception as exc:
            logger.warning("Plots failed for Huntsman batch %s: %s", batch.batch_id, exc)

        status = "success" if senpai_run.completed else "error"
        manifest = {
            **pending_manifest,
            "status": status,
            "error": senpai_run.error_message,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "result_path": str(result_path),
            "summary_path": str(summary_path),
        }
    except Exception as exc:
        logger.exception("Huntsman batch %s failed", batch.batch_id)
        manifest = {
            **pending_manifest,
            "status": "error",
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }

    elapsed_seconds = manifest["elapsed_seconds"]
    try:
        manifest = _publish_science_report(
            batch=batch,
            selection=selection,
            batch_dir=batch_dir,
            config=config,
            manifest=manifest,
            elapsed_seconds=elapsed_seconds,
            result=result,
            summary=summary,
        )
    except Exception as exc:
        logger.exception("Science report failed for Huntsman batch %s", batch.batch_id)
        manifest["science_report_error"] = str(exc)
        if manifest["status"] == "success":
            manifest["status"] = "error"
            manifest["error"] = f"science report generation failed: {exc}"

    write_json(batch_dir / "huntsman_manifest.json", manifest)
    return manifest


def run_huntsman_batches(
    discovery: HuntsmanDiscovery,
    output_directory: str | Path,
    config: AppConfig,
    options: HuntsmanRunOptions | None = None,
) -> list[dict]:
    """Run discovered batches sequentially and maintain a resumable summary."""

    options = options or HuntsmanRunOptions()
    output_root = Path(output_directory)
    _validate_unique_batch_directories(discovery)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "huntsman_batch_summary.json"

    results: list[dict] = []
    for batch in discovery.batches:
        results.append(_process_batch(batch, output_root, config, options))
        write_json(
            summary_path,
            {
                "layout_version": HUNTSMAN_LAYOUT_VERSION,
                "options": options.to_manifest(),
                "num_discovery_issues": len(discovery.skipped),
                "discovery_issues": [issue.to_manifest() for issue in discovery.skipped],
                "results": results,
            },
        )
    if not discovery.batches:
        write_json(
            summary_path,
            {
                "layout_version": HUNTSMAN_LAYOUT_VERSION,
                "options": options.to_manifest(),
                "num_discovery_issues": len(discovery.skipped),
                "discovery_issues": [issue.to_manifest() for issue in discovery.skipped],
                "results": [],
            },
        )
    return results
