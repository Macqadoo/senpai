# Releasing astro-senpai

Publishing to PyPI is automated by `.github/workflows/python-publish.yml`. It
runs **only when a GitHub Release is published** — never on a merge to `main`,
never for draft releases, never for prereleases.

The PyPI project is **`astro-senpai`**; the import package is `senpai`. Built
artifacts normalize the dash to an underscore (`astro_senpai-2.6.0-py3-none-any.whl`).

Authentication uses **PyPI Trusted Publishing** (OIDC). There is no
`PYPI_API_TOKEN` secret, and there should never be one.

## One-time setup

Both steps are required before the first automated release. Until they are
done, the publish job will fail with a 403 or hang waiting on a missing
environment.

### 1. Register the trusted publisher on PyPI

On <https://pypi.org/manage/project/astro-senpai/settings/publishing/>, add a
GitHub publisher with **exactly** these values:

| Field             | Value                |
| ----------------- | -------------------- |
| Owner             | `ssc-ai`             |
| Repository        | `senpai`             |
| Workflow name     | `python-publish.yml` |
| Environment name  | `pypi`               |

The workflow name is the filename only — not the full path. The environment
name must match the workflow's `environment.name` exactly or PyPI will reject
the OIDC token.

### 2. Create the `pypi` environment with required reviewers

In **Settings → Environments → New environment**, name it `pypi`, then:

- Enable **Required reviewers** and add the maintainers who are allowed to
  approve a PyPI upload.
- Optionally restrict **Deployment branches and tags** to tags matching `v*`.

This is the human gate. Anyone with write access can publish a GitHub Release,
which starts the build — but the upload to PyPI pauses at "Waiting for review"
until a listed reviewer approves it in the Actions run. Approval is per-run, so
each release is approved individually.

## Cutting a release

1. Bump `version` in `pyproject.toml`, and in `CHANGELOG.md` rename the
   `[Unreleased]` heading to `[<version>] - <YYYY-MM-DD>`, adding a fresh empty
   `[Unreleased]` above it. Merge to `main`.
2. Tag the release commit as `v<version>` — e.g. `v2.6.1` for version `2.6.1`.
   `make tag` does this from `pyproject.toml` and pushes it.
3. On GitHub, create a Release pointing at that tag and click **Publish
   release**. Leave "Set as a pre-release" unchecked — prereleases are skipped
   by design.
4. The `release-build` job runs unattended: it verifies the tag, lints, tests,
   builds, and checks the artifacts.
5. A reviewer approves the `pypi` environment in the Actions run. The upload
   then happens.

## What the workflow verifies before uploading

- The release tag equals `v` + the `version` in `pyproject.toml`. A mismatch
  fails loudly instead of silently republishing the old version.
- That version is not already on PyPI (PyPI rejects re-uploads, and it is
  better to find out before the release is public).
- `ruff check senpai --select E9,F63,F7,F82,F401,F541,F811` passes — the same
  narrow rule set `ci.yml` uses. The full `make lint` config currently reports
  ~850 pre-existing findings, so gating releases on it would block every
  release until that backlog is cleared. Worth fixing, but not as part of the
  release path.
- `pytest -m "not requires_astrometry and not requires_catalog"` passes — the
  same subset as `make test-ci`. Astrometry.net, index files, and local star
  catalogs are unavailable on GitHub runners.
- The built wheel and sdist filenames carry the expected version.
- `twine check` passes on both artifacts.

## If something goes wrong

**Tag/version mismatch.** Fix `pyproject.toml` or the tag, delete the release
and tag, and redo. The `published` event does not re-fire when you edit an
existing release — you must publish a new one.

**Version already on PyPI.** Bump the version. PyPI never allows re-uploading a
filename, even after a delete; a bad release can only be *yanked*, not
unpublished.

**Publish job stuck on "Waiting for review".** That is the environment gate
working as intended. A required reviewer approves it from the run page.

## Manual fallback

`make publish` still exists (clean tree → version check → test → build →
`twine upload` with a token from `~/.pypirc`), as does `make publish-test` for
TestPyPI. Prefer the workflow; the manual path bypasses the reviewer gate.
