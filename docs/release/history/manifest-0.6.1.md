# econhdfe 0.6.1 Local Release-Candidate Manifest

Baseline: the user-supplied 0.6.0 source and release bundle. This PATCH candidate
repairs confirmed numerical/inference failures and audits the assumptions,
proofs and implementations behind three mathematical contributions.

## Contents

The bundle contains a full GitHub-ready source ZIP (including CI, tests,
compatibility history, documentation and scripts), a standard Python sdist, a
runtime wheel, the portable Skill, three revised LaTeX/PDF manuscripts, the
Chinese mathematical review and recursive SHA256 hashes. Manuscripts are source
and review artifacts, not runtime wheel payload. Build timestamps mean this is
content-parity validation, not a claim of byte-identical archives across hosts.

## Local status versus external prerequisites

Actual counts/environment/evidence are in `docs/development/test-status.md`.
The full local suite, original audit regressions, mathematical oracles, target
wheel smoke, extracted-source retest, source/wheel parity, and hash verification
are local gates. Ordinary build-release execution also requires isolated builds
and clean dependency installation. Offline diagnostic execution explicitly does
not satisfy those two gates. A reviewed maintenance area may document a deferred
external test; it is not proof that the deferred test passed.

The remote GitHub matrix, clean dependency resolution against package indexes,
lowest supported dependency combinations, Stata/upstream certification, optional
GPU/IO/FLINT paths, and historic 10-million-row benchmarks are not silently
inferred from this local candidate. Do not describe this archive as externally
certified. No GitHub repository, tag, Release or PyPI project was changed.

## Mathematical scope

Exact rank is for finite categorical intercept dummy designs with observed
levels. Numerical residual-core equivalence requires strictly positive finite
weights and preserves raw observation multiplicity. Structural simplification
requires true partition dependencies, correct active/reference columns and the
same slope multiplier. These are not blanket results for arbitrary varying
slopes, group-individual/multimembership designs or all named coefficients.
The default pairwise DoF setting remains unchanged; exact DoF remains explicit.

Registered results have proof/implementation review under stated assumptions,
not independent certification of historical originality.
