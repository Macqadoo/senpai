"""Minimal CLI for processing Huntsman FITS collections."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from senpai.core.config import initialize_config
from senpai.core.constants import CONFIG_DIR
from senpai.core.logging import set_log_level
from senpai.integrations.huntsman.batches import discover_huntsman_batches

logger = logging.getLogger(__name__)

HUNTSMAN_CONFIG = CONFIG_DIR / "huntsman.yaml"
DEFAULT_MAX_SIDEREAL_FWHM = 20.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="senpai-huntsman",
        description="Discover and process Huntsman sidereal/rate FITS batches.",
    )
    parser.add_argument("input", type=Path, help="Directory containing Huntsman FITS files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("huntsman_runs"),
        help="Output root (default: huntsman_runs/).",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=HUNTSMAN_CONFIG,
        help=f"SENPAI config (default: {HUNTSMAN_CONFIG}).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report grouping without processing.")
    parser.add_argument(
        "--sidereal-policy",
        choices=("first", "sharpest", "all"),
        default="sharpest",
        help="Sidereal anchor selection (default: sharpest).",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Process batches with completed products (existing products are skipped by default).",
    )
    parser.add_argument(
        "--allow-blurry-sidereal",
        action="store_true",
        help="Disable the default sidereal FWHM rejection gate.",
    )
    parser.add_argument(
        "--max-sidereal-fwhm",
        type=float,
        default=DEFAULT_MAX_SIDEREAL_FWHM,
        metavar="PIXELS",
        help=f"Default blur-gate threshold (default: {DEFAULT_MAX_SIDEREAL_FWHM:g}px).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_sidereal_fwhm <= 0:
        raise SystemExit("--max-sidereal-fwhm must be positive")

    config = initialize_config(args.config)
    set_log_level(config.logging.level)
    discovery = discover_huntsman_batches(
        args.input,
        config,
        output_directory=args.output,
    )
    logger.info(
        "Discovered %d Huntsman batch(es); skipped %d unusable FITS file(s)",
        len(discovery.batches),
        len(discovery.skipped),
    )

    if args.dry_run:
        print(json.dumps(discovery.to_manifest(), indent=2))
        return 0
    if not discovery.batches:
        logger.error("No usable Huntsman batches found in %s", args.input)
        return 1

    from senpai.integrations.huntsman.runner import (
        HuntsmanRunOptions,
        has_pending_batches,
        run_huntsman_batches,
    )

    options = HuntsmanRunOptions(
        sidereal_policy=args.sidereal_policy,
        skip_existing=not args.reprocess,
        skip_blurry_sidereal=not args.allow_blurry_sidereal,
        max_sidereal_fwhm=args.max_sidereal_fwhm,
    )
    if has_pending_batches(
        discovery,
        args.output,
        skip_existing=options.skip_existing,
    ):
        from senpai.astrometry import enforce_indices, require_astrometry_install
        from senpai.catalog.runner import enforce_catalog

        require_astrometry_install()
        enforce_indices()
        enforce_catalog()
    results = run_huntsman_batches(discovery, args.output, config, options)
    errors = sum(result["status"] == "error" for result in results)
    logger.info(
        "Huntsman processing complete: %d batch(es), %d error(s)",
        len(results),
        errors,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
