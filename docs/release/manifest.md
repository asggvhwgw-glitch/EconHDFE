# Release artifact manifest

This page describes current artifact roles. A source version is not proof of
publication or formal acceptance; see [validation status](../development/test-status.md).
The historical [0.6.1 local manifest](history/manifest-0.6.1.md) is retained verbatim.

## Frozen candidate set

For version `<version>`, the `release-candidate` CI artifact contains exactly:

| File | Purpose |
| --- | --- |
| `econhdfe-<version>-py3-none-any.whl` | Installable runtime and canonical support templates |
| `econhdfe-<version>.tar.gz` | Python source distribution used for isolated wheel construction |
| `econhdfe-skill-v<version>.zip` | Complete portable Agent Skill |
| `econhdfe-v<version>-source.zip` | Repository source, tests, documentation and validation tooling |
| `econhdfe-v<version>-release-bundle.zip` | Combined auditable delivery with recursive hashes |

Technical manuscripts remain source/release assets and are excluded from the wheel.
Archive timestamps may differ between builds; do not claim byte reproducibility
across hosts. Publication uses already-frozen bytes, not a fresh rebuild.

## Records and authority

- `maintenance.json`: review/compatibility responsibilities.
- `execution.json`: candidate execution schema; `not_run` is not success.
- Detached final execution manifest: real source SHA, runtime/validation fingerprint,
  hashed execution evidence, and final wheel/sdist/source/bundle hashes. It stays
  outside the immutable bundle to avoid self-referential hashing.
- `SHA256SUMS.txt`: added by the publication workflow for GitHub Release assets.
  PyPI receives the exact same wheel and sdist checked against these hashes.

Run the [acceptance gate](acceptance.md) and follow [publishing.md](publishing.md).
Only an explicitly authorized publication changes tags, GitHub Releases or PyPI;
normal CI and a PR do not publish. Public statistical/mathematical support claims
remain limited to their actual evidence and assumptions.
