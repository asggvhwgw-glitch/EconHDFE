# EconHDFE release manifest

This file describes the **current release state**. Version-specific candidate manifests and closeout records are retained under [`history/`](history/).

## Published release

The latest published release is **v0.6.3**, frozen at commit
`2782d93f901ec1eed28ac0f963834701e0a44b09`.

Release evidence is retained under [`evidence/0.6.3/`](evidence/0.6.3/). Historical evidence is immutable: later development validation does not rewrite the evidence attached to an already published release.

## Current development line

The repository currently contains the **unreleased 0.6.5 development line**. It includes the reorganized contracts/behavior/numerics/tooling test structure, later documentation/formal-verification work, and other changes merged after v0.6.3.

Passing ordinary development CI does not create a release. Until an explicit publication step is authorized, there is no 0.6.5 tag, detached GitHub Release, or PyPI publication to infer from the repository state.

Current development priorities and validation boundaries are tracked in:

- [Unified TODO / roadmap](../../TODO.md)
- [Validation status](../development/test-status.md)
- [Release acceptance](acceptance.md)
- [Release checklist](checklist.md)
- [Publishing workflow](publishing.md)

## Mathematical technical documents

The repository maintains three theorem-backed project technical documents:

1. exact multiway categorical FE structural rank / absorbed DoF;
2. exact multiway categorical residual-core reduction;
3. exact partition-refinement structural design reduction.

Their core mathematical statements are machine-checked in the separate Lean 4 library under `formal/`. This establishes mathematical correctness under stated assumptions; it does not certify the production Python/Numba implementation or historical originality.

## Publication rule

A future release must be tied to a frozen source commit and pass the release checklist, isolated source/wheel builds, clean-install validation, required platform/dependency matrix, artifact verification, and explicit publication authorization. Release artifacts must be bound to that frozen commit rather than reused from an earlier ordinary CI run.
