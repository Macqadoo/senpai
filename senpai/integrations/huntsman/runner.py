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
from senpai.integrations.huntsman.batches import HuntsmanBatch, HuntsmanDiscovery
from senpai.integrations.huntsman.quality import (
    DEFAULT_MAX_SIDEREAL_FWHM,
    DEFAULT_MIN_SIDEREAL_SOURCES,
    SiderealPolicy,
    select_batch_frames,
)

logger = logging.getLogger(__name__)


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


def completed_product_exists(batch: HuntsmanBatch, batch_dir: Path) -> bool:
    """Return whether a batch directory contains a completed Huntsman product."""

    manifest_path = batch_dir / "huntsman_manifest.json"
    if manifest_path.is_file():
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                return json.load(handle).get("status") == "success"
        except (OSError, ValueError):
            return False
    return (batch_dir / f"senpai_{batch.batch_id}.json").is_file()


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
        not completed_product_exists(batch, output_root / batch.output_label)
        for batch in discovery.batches
    )


def _process_batch(
    batch: HuntsmanBatch,
    output_root: Path,
    config: AppConfig,
    options: HuntsmanRunOptions,
) -> dict:
    batch_dir = output_root / batch.output_label
    base_manifest = {
        **batch.to_manifest(),
        "output_dir": str(batch_dir),
        "options": options.to_manifest(),
    }

    if options.skip_existing and completed_product_exists(batch, batch_dir):
        logger.info("Skipping existing Huntsman batch %s", batch.batch_id)
        return {**base_manifest, "status": "skipped_existing", "error": None}

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
        return {
            **base_manifest,
            **selection_manifest,
            "status": "skipped_blurry",
            "error": "no sidereal frame passed the FWHM quality gate",
        }
    if not selection.selected_files:
        return {
            **base_manifest,
            **selection_manifest,
            "status": "skipped_empty",
            "error": "batch contains no usable frames",
        }

    batch_dir.mkdir(parents=True, exist_ok=True)
    pending_manifest = {
        **base_manifest,
        **selection_manifest,
        "status": "running",
        "error": None,
    }
    write_json(batch_dir / "huntsman_manifest.json", pending_manifest)

    config.runtime.output_dir = batch_dir
    config.runtime.run_id = batch.batch_id
    save_run_metadata(batch_dir, "senpai.cli.huntsman", config)

    started = time.monotonic()
    try:
        images = load_fits_files(list(selection.selected_files))
        senpai_run = process_senpai_collect(images, id=batch.batch_id)

        result = senpai_run.to_result()
        result_path = batch_dir / f"senpai_{result.id}.json"
        write_json(result_path, result.model_dump())

        summary = senpai_run.to_summary()
        summary_path = batch_dir / f"senpai_{summary.id}_summary.json"
        write_json(summary_path, summary.model_dump())
        write_frame_quicklooks(summary, batch_dir)

        try:
            final_plots(senpai_run, batch_dir)
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
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "huntsman_batch_summary.json"

    results: list[dict] = []
    for batch in discovery.batches:
        results.append(_process_batch(batch, output_root, config, options))
        write_json(
            summary_path,
            {
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
                "options": options.to_manifest(),
                "num_discovery_issues": len(discovery.skipped),
                "discovery_issues": [issue.to_manifest() for issue in discovery.skipped],
                "results": [],
            },
        )
    return results
