# Public release publication

Public publication is deliberately separate from candidate construction and formal
authorization. The validation workflow builds immutable candidate artifacts; the
detached execution manifest and formal release gate decide whether a candidate is
authorized; only then should a maintainer run the manual **Publish release** workflow.

## Stable workflow identity

The workflow is stored at `.github/workflows/release-v0.6.3.yml`. The filename is
intentionally retained even though the implementation is version-agnostic because
PyPI Trusted Publishing binds the GitHub OIDC identity to the workflow filename.
Renaming it requires updating the PyPI publisher configuration first.

The workflow has no push trigger. A normal commit, merge, tag, or version bump cannot
publish a package accidentally.

## Required inputs

- `version`: package version without the leading `v`.
- `source_run`: successful **Tests and release validation** workflow run ID that
  produced the frozen `release-candidate` artifact.
- `source_commit`: full commit SHA authorized by the detached release manifest.
- `release_notes_file`: optional repository-relative Markdown file at that frozen
  commit; when omitted GitHub generates release notes.
- `prerelease`: optional GitHub prerelease flag.
- `publish_pypi`: defaults to true; disable only when intentionally making a
  GitHub-only release.

## What publication verifies

Before creating a tag or release the workflow checks that:

1. the requested version equals `project.version` at `source_commit`;
2. `source_run` completed successfully, ran `.github/workflows/ci.yml`, and its
   `head_sha` equals `source_commit`;
3. the candidate artifact contains exactly one wheel, sdist, Skill archive, source
   archive and release bundle with names matching the requested version;
4. a fresh `SHA256SUMS.txt` over those immutable downloaded bytes verifies;
5. an existing tag, if any, points to the exact authorized commit; an existing
   GitHub Release is never overwritten.

The release job uploads those exact candidate bytes. It does not rebuild the
package after tagging.

The PyPI job downloads the same candidate again and verifies the wheel/sdist against
the SHA256 values produced by the GitHub Release job. Publishing uses OIDC Trusted
Publishing only; no repository API-token fallback is kept.

## Recovery and retry

If GitHub Release succeeds but PyPI fails, use **Re-run failed jobs** on the same
workflow run. The successful GitHub Release job is not repeated and the PyPI job
re-downloads the same frozen candidate.

Published tags, GitHub Release assets and PyPI files are immutable. A corrected
artifact requires a new version rather than replacing bytes under an existing
version.

This workflow is an execution mechanism, not an authorization mechanism. Do not
dispatch it until the detached formal acceptance record reports `release_ready=true`
for the same commit and artifacts.
