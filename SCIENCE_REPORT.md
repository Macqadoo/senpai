# Huntsman science report

Every terminal Huntsman batch now publishes a fixed-name
`science_report.json` in its batch output directory. This is the stable,
MongoDB-ready science contract; `pipeline_report.json` remains the controller
execution/input report and is not replaced.

## Ingestion contract

- Recursively discover files whose basename is exactly `science_report.json`.
- Decode each file as one JSON document and route it to the
  `senpai_science_results` collection.
- Upsert by `document_key`. Do not use unconditional inserts: a retry of the
  same inputs, SENPAI commit and configuration deliberately has the same key.
- Treat the root document (`document_type: run`) as the completion boundary.
  SENPAI writes continuation documents first and atomically publishes the root
  last.
- Accept only supported `schema_version` values. Version 1 is the initial
  contract.

No MongoDB URI or write permission belongs in SENPAI. SENPAI only writes files;
NiFi owns transport, authentication and collection writes.

## Root document

The run document includes:

- `science_report_id`, `run_key`, `document_key`, status and timestamps;
- resolved `TASKID`/`SKTASKID`, camera, sensor, filter, target and NORAD
  identities, plus the FITS card or filename that supplied each value;
- input filename, path, optional archive ID, SHA-256, selection decision,
  pre-check result and every header card from every FITS HDU;
- per-frame WCS, WCS quality, seeing/FWHM, background, limiting magnitude,
  zero point, processing history and artifact paths;
- deterministic point/streak detection IDs and correlated-streak results;
- complete per-star calibration measurements when the document remains below
  the MongoDB-safe size threshold;
- SENPAI version/commit, configuration hash, input/product fingerprints and
  detector provenance.

`archive_id` is populated only when it is present in the data supplied to
SENPAI. Current SensorKit FITS files generally do not carry the NiFi archive
ID, so an ingester should fall back to the exact filename/path and then the
task-plus-camera identifiers when linking to `nifi_test`.

All timestamps are UTC strings and all values are strict JSON. Non-finite
scientific values are represented as `null`, never non-standard `NaN` or
`Infinity` tokens.

## Large calibration results

MongoDB has a 16 MiB BSON document limit. SENPAI conservatively starts
splitting at 14,000,000 encoded JSON bytes. The root then contains
`calibration_measurements_embedded: false` and a `measurement_parts` list.

Each continuation lives below the batch directory at a path such as:

```text
science_report_parts/<product-generation>/part_0001/science_report.json
```

It uses `document_type: calibration_measurements`, carries the same
`science_report_id` and `run_key`, and has its own deterministic
`document_key`. Keeping the same basename lets a simple recursive NiFi
filename matcher ingest root and continuation documents through one route.
Consumers should select the part keys listed by the root rather than assuming
that every historical part with the same report ID is current.

Generation-specific paths and document keys keep the previous root and its
parts internally consistent until the replacement root is atomically
published. Obsolete on-disk generations are removed only after publication.

## Publication behavior

Reports are generated for successful, failed, `skipped_blurry` and
`skipped_empty` batches. Each document is written to a temporary file, flushed
and atomically renamed. If a required report cannot be generated after an
otherwise successful reduction, the Huntsman manifest is marked `error`
instead of advertising an incomplete science product.

Existing successful outputs are still honoured by the adapter's
`skip_existing` behavior. Use the Huntsman CLI's reprocess option when a
science report must be backfilled for an older result.
