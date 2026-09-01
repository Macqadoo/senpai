"""Thin Huntsman/SensorKit ingestion adapter for the SENPAI v2.8 engine."""

from senpai.integrations.huntsman.batches import (
    HuntsmanBatch,
    HuntsmanDiscovery,
    HuntsmanFrameRecord,
    discover_huntsman_batches,
)
from senpai.integrations.huntsman.filenames import (
    ParsedHuntsmanFilename,
    parse_huntsman_filename,
)

__all__ = [
    "HuntsmanBatch",
    "HuntsmanDiscovery",
    "HuntsmanFrameRecord",
    "ParsedHuntsmanFilename",
    "discover_huntsman_batches",
    "parse_huntsman_filename",
]

