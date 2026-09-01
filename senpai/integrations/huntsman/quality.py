"""Sidereal FWHM quality checks and anchor selection for Huntsman batches."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from senpai.integrations.huntsman.batches import HuntsmanBatch, HuntsmanFrameRecord

logger = logging.getLogger(__name__)

SiderealPolicy = Literal["first", "sharpest", "all"]
DEFAULT_MAX_SIDEREAL_FWHM = 20.0
DEFAULT_MIN_SIDEREAL_SOURCES = 10


@dataclass(frozen=True, slots=True)
class SiderealQualityResult:
    path: Path
    passed: bool
    conclusive: bool
    median_fwhm: float | None
    detected_source_count: int
    reason: str | None = None

    def to_manifest(self) -> dict:
        return {
            "path": str(self.path),
            "passed": self.passed,
            "conclusive": self.conclusive,
            "median_fwhm": self.median_fwhm,
            "detected_source_count": self.detected_source_count,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class HuntsmanBatchSelection:
    batch: HuntsmanBatch
    policy: SiderealPolicy
    selected_sidereal: tuple[HuntsmanFrameRecord, ...]
    skipped_sidereal: tuple[HuntsmanFrameRecord, ...]
    quality: tuple[SiderealQualityResult, ...]
    rejected_for_blur: bool = False

    @property
    def selected_files(self) -> tuple[Path, ...]:
        if self.rejected_for_blur:
            return ()
        return tuple(frame.path for frame in self.selected_sidereal) + tuple(
            frame.path for frame in self.batch.rate_frames
        )

    def to_manifest(self) -> dict:
        return {
            "sidereal_policy": self.policy,
            "selected_sidereal": [str(frame.path) for frame in self.selected_sidereal],
            "skipped_sidereal": [str(frame.path) for frame in self.skipped_sidereal],
            "rate_frames": [str(frame.path) for frame in self.batch.rate_frames],
            "rejected_for_blur": self.rejected_for_blur,
            "quality": [result.to_manifest() for result in self.quality],
        }


def measure_sidereal_quality(
    path: Path,
    *,
    max_fwhm: float = DEFAULT_MAX_SIDEREAL_FWHM,
    min_sources: int = DEFAULT_MIN_SIDEREAL_SOURCES,
) -> SiderealQualityResult:
    """Measure detection-stage FWHM without a solve, catalog query or photometry."""

    try:
        from senpai.engine.models.metadata import TrackMode
        from senpai.engine.processing.collect import process_senpai_collect
        from senpai.engine.utils.file_io import load_fits_file

        image = load_fits_file(path)
        run = process_senpai_collect(
            [image],
            id=f"quality_{path.name}",
            force_track_mode=TrackMode.SIDEREAL,
            pipeline_mode="detect",
        )
        frame = run.sidereal_frames[0] if run.sidereal_frames else None
        starfield = frame.starfield if frame is not None else None
        fwhm_stats = starfield.fwhm_stats if starfield is not None else None
        median_fwhm = float(fwhm_stats.median_fwhm) if fwhm_stats is not None else None
        source_count = len(starfield.detections) if starfield is not None else 0
    except Exception as exc:
        logger.warning("Sidereal quality check was inconclusive for %s: %s", path, exc)
        return SiderealQualityResult(
            path=path,
            passed=True,
            conclusive=False,
            median_fwhm=None,
            detected_source_count=0,
            reason=f"quality check failed: {exc}",
        )

    if median_fwhm is None:
        return SiderealQualityResult(
            path=path,
            passed=True,
            conclusive=False,
            median_fwhm=None,
            detected_source_count=source_count,
            reason="no valid detection-stage FWHM",
        )
    if source_count < min_sources:
        return SiderealQualityResult(
            path=path,
            passed=True,
            conclusive=False,
            median_fwhm=median_fwhm,
            detected_source_count=source_count,
            reason=f"only {source_count} detected sources; need {min_sources}",
        )
    if median_fwhm > max_fwhm:
        return SiderealQualityResult(
            path=path,
            passed=False,
            conclusive=True,
            median_fwhm=median_fwhm,
            detected_source_count=source_count,
            reason=f"median FWHM {median_fwhm:.1f}px exceeds {max_fwhm:.1f}px",
        )
    return SiderealQualityResult(
        path=path,
        passed=True,
        conclusive=True,
        median_fwhm=median_fwhm,
        detected_source_count=source_count,
    )


def select_batch_frames(
    batch: HuntsmanBatch,
    *,
    policy: SiderealPolicy = "sharpest",
    skip_blurry_sidereal: bool = True,
    max_fwhm: float = DEFAULT_MAX_SIDEREAL_FWHM,
    min_sources: int = DEFAULT_MIN_SIDEREAL_SOURCES,
    evaluator: Callable[..., SiderealQualityResult] = measure_sidereal_quality,
) -> HuntsmanBatchSelection:
    """Select sidereal anchors while always retaining the batch's rate frames."""

    if policy not in {"first", "sharpest", "all"}:
        raise ValueError(f"unsupported sidereal policy: {policy}")

    sidereal = batch.sidereal_frames
    if not sidereal:
        return HuntsmanBatchSelection(
            batch=batch,
            policy=policy,
            selected_sidereal=(),
            skipped_sidereal=(),
            quality=(),
        )

    to_measure = sidereal if policy in {"sharpest", "all"} else sidereal[:1]
    if not skip_blurry_sidereal and policy != "sharpest":
        to_measure = []
    quality = tuple(
        evaluator(frame.path, max_fwhm=max_fwhm, min_sources=min_sources)
        for frame in to_measure
    )
    quality_by_path = {result.path: result for result in quality}

    def eligible(frame: HuntsmanFrameRecord) -> bool:
        result = quality_by_path.get(frame.path)
        return not skip_blurry_sidereal or result is None or result.passed

    if policy == "first":
        selected = tuple(sidereal[:1]) if eligible(sidereal[0]) else ()
    elif policy == "all":
        selected = tuple(frame for frame in sidereal if eligible(frame))
    else:
        candidates = [frame for frame in sidereal if eligible(frame)]

        def sharpness(frame: HuntsmanFrameRecord) -> tuple[bool, float, str]:
            result = quality_by_path[frame.path]
            return (
                result.median_fwhm is None,
                result.median_fwhm if result.median_fwhm is not None else float("inf"),
                str(frame.path),
            )

        selected = (min(candidates, key=sharpness),) if candidates else ()

    selected_paths = {frame.path for frame in selected}
    skipped = tuple(frame for frame in sidereal if frame.path not in selected_paths)
    return HuntsmanBatchSelection(
        batch=batch,
        policy=policy,
        selected_sidereal=selected,
        skipped_sidereal=skipped,
        quality=quality,
        rejected_for_blur=bool(sidereal and not selected),
    )
