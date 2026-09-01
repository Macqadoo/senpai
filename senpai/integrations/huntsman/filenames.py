"""Filename parsing and FITS suffix handling for Huntsman observations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from senpai.engine.models.metadata import TrackMode

FITS_SUFFIXES: tuple[str, ...] = (
    ".fits.fz",
    ".fit.fz",
    ".fts.fz",
    ".fits",
    ".fit",
    ".fts",
)

_OLD_RE = re.compile(
    r"^(?P<timestamp>\d{8}T\d{6,9})_"
    r"(?P<mode>sidereal|rate)_"
    r"(?P<task>[^_]+)_F(?P<index>\d+)of(?P<count>\d+)_"
    r"(?P<filter>.+)$",
    re.IGNORECASE,
)
_NEW_RE = re.compile(
    r"^(?P<timestamp>\d{6}T\d{6,9})_"
    r"(?P<norad>[^_]+)_(?P<task>[^_]+)_(?P<filter>[^_]+)_"
    r"F(?P<index>\d+)of(?P<count>\d+)_(?P<mode>sidereal|rate)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParsedHuntsmanFilename:
    """Structured fields available from an old or new Huntsman filename."""

    path: Path
    convention: str | None = None
    timestamp: datetime | None = None
    filename_task_id: str | None = None
    filename_filter: str | None = None
    filename_track_mode: TrackMode | None = None
    frame_index: int | None = None
    frame_count: int | None = None
    filename_norad_id: str | None = None

    @property
    def recognized(self) -> bool:
        return self.convention is not None


def is_fits_file(path: str | Path) -> bool:
    """Return whether a path has a supported FITS/fpack filename suffix."""

    return Path(path).name.lower().endswith(FITS_SUFFIXES)


def fits_stem(path: str | Path) -> str:
    """Strip one logical FITS suffix, including compound ``.fits.fz`` names."""

    name = Path(path).name
    lower_name = name.lower()
    for suffix in FITS_SUFFIXES:
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _parse_timestamp(token: str) -> datetime | None:
    date_token, time_token = token.split("T", maxsplit=1)
    if len(time_token) < 6:
        return None
    base_format = "%Y%m%d%H%M%S" if len(date_token) == 8 else "%y%m%d%H%M%S"
    try:
        value = datetime.strptime(date_token + time_token[:6], base_format)
    except ValueError:
        return None
    fractional = time_token[6:]
    if fractional:
        value = value.replace(microsecond=int(fractional[:6].ljust(6, "0")))
    return value.replace(tzinfo=UTC)


def _track_mode(value: str) -> TrackMode:
    return TrackMode.SIDEREAL if value.lower() == "sidereal" else TrackMode.RATE


def parse_huntsman_filename(path: str | Path) -> ParsedHuntsmanFilename:
    """Parse the two supported Huntsman conventions without reading headers.

    Unknown names produce an unrecognized record rather than raising. Header
    discovery may still accept such a file when it carries a complete task ID,
    filter and track mode.
    """

    parsed_path = Path(path)
    stem = fits_stem(parsed_path)

    match = _OLD_RE.fullmatch(stem)
    if match:
        return ParsedHuntsmanFilename(
            path=parsed_path,
            convention="old",
            timestamp=_parse_timestamp(match["timestamp"]),
            filename_task_id=match["task"],
            filename_filter=match["filter"],
            filename_track_mode=_track_mode(match["mode"]),
            frame_index=int(match["index"]),
            frame_count=int(match["count"]),
        )

    match = _NEW_RE.fullmatch(stem)
    if match:
        return ParsedHuntsmanFilename(
            path=parsed_path,
            convention="new",
            timestamp=_parse_timestamp(match["timestamp"]),
            filename_task_id=match["task"],
            filename_filter=match["filter"],
            filename_track_mode=_track_mode(match["mode"]),
            frame_index=int(match["index"]),
            frame_count=int(match["count"]),
            filename_norad_id=match["norad"],
        )

    return ParsedHuntsmanFilename(path=parsed_path)


def frame_sort_key(parsed: ParsedHuntsmanFilename) -> tuple[datetime, int, str]:
    """Deterministic observation ordering for parsed and header-only files."""

    earliest = datetime.min.replace(tzinfo=UTC)
    return parsed.timestamp or earliest, parsed.frame_index or 0, str(parsed.path)
