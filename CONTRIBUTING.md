# Contributing to gwreparam

Contributions are welcome, including bug reports, focused tests, documentation,
and reusable gravitational-wave parameter transformations. This guide follows
Sequince's development conventions.

## Community and responsibility

Treat other contributors with respect and follow the
[Python Community Code of Conduct](https://www.python.org/psf/codeofconduct/).
Contributions must comply with the project's [license](LICENSE). Preserve
required attribution when incorporating code from other projects.

AI-assisted contributions are welcome under our [AI policy](AI_POLICY.md).
Contributors are responsible for understanding, validating, and explaining
their changes, including numerical correctness and performance.

## Scope of contributions

Keep each pull request focused on one problem or a closely related set of
changes. Discuss substantial API changes, dependencies, or algorithms with
maintainers before investing significant implementation effort. An agreed
implementation plan can provide that scope.

gwreparam provides sampler-independent coordinate mappings, Jacobians, and
deterministic branch handling for gravitational-wave inference. Keep the core
independent of Sequince, Nessai, Bilby, and BBHx. Sampler adapters, mutation
steps, acceptance logic, random branch selection, and application-specific
likelihoods belong in downstream packages.

Prefer existing interfaces and helpers. Introduce abstractions only for a
concrete shared need, and avoid unrelated cleanup during extraction or fixes.

## Development setup

Fork [mj-will/gwreparam](https://github.com/mj-will/gwreparam), then clone your
fork and configure local main to track upstream:

```bash
git clone git@github.com:YOUR_USERNAME/gwreparam.git
cd gwreparam
git remote add upstream https://github.com/mj-will/gwreparam.git
git fetch upstream
git switch main
git branch --set-upstream-to=upstream/main main
```

Keep local main for upstream changes and use separate development branches.
With a clean working tree, start a contribution with:

```bash
git switch main
git pull --ff-only
git switch -c fix-descriptive-name
```

Here, upstream is the gwreparam repository and origin is your fork. Preserve
existing local work before switching branches or updating the checkout.

Use [Pixi](https://pixi.prefix.dev/latest/installation/) for development.
Dependencies, environments, and tasks belong in pyproject.toml. Install the
editable package and development tools, then install the hooks:

```bash
pixi install
pixi run pre-commit install
```

Configure both pre-commit and commit-msg hooks. When changing dependencies,
update pyproject.toml and rerun pixi install. Keep generated lock files,
environments, caches, and build outputs out of commits.

## Code style and documentation

- Use Ruff for formatting and import ordering, four-space indentation, and
  a 79-character Python line limit.
- Use descriptive names; conventional mathematical names are appropriate
  when their meaning is clear locally.
- Keep public docstrings concise and NumPy-style. Include essential shapes,
  units, conventions, return values, and domain restrictions. Avoid repeating
  the implementation or adding long derivations; put those in docs.
- Explain non-obvious numerical choices in comments rather than restating code.
- Use NumPy initially and document supported shapes and dtypes. Do not imply
  support for untested array backends or silently change conversion behavior.
- Avoid unnecessary copies and input mutation. Support performance claims
  with reproducible measurements.

## Typing

Add parameter and return annotations to new functions and methods, including
private helpers and tests where practical. Update annotations and affected
callers together when changing an interface.

- Target Python 3.11+ and use modern syntax such as list[str] and float | None.
  Prefer Mapping or Sequence when callers need not supply a concrete container.
- Use numpy.typing.NDArray for NumPy arrays, with an appropriate dtype when
  known. Use ArrayLike only when the function accepts and converts such inputs.
- Document shape and scalar-versus-batch behavior; annotations alone do not
  express these constraints. Validate them at appropriate boundaries.
- Give callbacks explicit signatures and prefer precise unions or protocols
  to Any. Keep abstractions local until they have a concrete shared use.
- Use TYPE_CHECKING for annotation-only imports where needed. Type annotations
  must not introduce sampler dependencies.
- Keep casts and type-ignore comments narrowly scoped and explain non-obvious
  reasons. Fix inaccurate annotations rather than broadly suppressing errors.

Use `pixi run typecheck`. Static typing complements numerical
and runtime tests; it does not replace them.

## Scientific correctness

Specify whether an operation is a coordinate change, symmetry fold, or proposal
mapping. Document the forward and inverse log-absolute-Jacobian conventions and
retain the auxiliary branch labels required for inversion.

Preserve angle periods, phase signs, distance units, branch multiplicities,
prior support, and invalid-proposal behavior during extraction. Do not silently
clip, regularize singularities, or alter branch probabilities. Keep deterministic
mappings separate from downstream RNG and acceptance decisions.

Explain intentional changes affecting transformed densities or downstream
inference. Check consumer compatibility, including import paths and serialized
objects where relevant. Python-version requirements must also be compatible
with the consumer's declared support.

## Checks and tests

Run before requesting review:

```bash
pixi run pre-commit run --all-files
pixi run test
```

Pre-commit covers Ruff linting and formatting, mypy, spelling, whitespace,
YAML validation, and large-file checks. Run pytest separately. Review any
automatic fixes and rerun affected checks. The actual pyproject.toml and
.pre-commit-config.yaml configuration is authoritative.

Tests use pytest and live in tests/. Keep them focused on observable behavior
and scientific invariants, rather than private implementation details. Add
regression tests for fixes and reuse existing numerical tests when extracting
code. Use conftest.py for genuinely shared fixtures; keep one-off setup local.
Parameterize related cases without creating redundant coverage.

Select checks appropriate to the change:

- Forward/inverse round trips, with angular comparisons modulo their periods.
- Discrete branches, seam behavior, and preservation of branch labels.
- Jacobian signs, forward/inverse cancellation, and independent numerical
  Jacobian checks away from singularities.
- Relevant singular boundaries, invalid inputs, and rejected proposals.
- Prior preservation, including branch multiplicity, for folded coordinates.
- Agreement with the original implementation and focused downstream adapter
  tests during migration.

Use fixed seeds where tests require randomness and justify numerical tolerances.
Verify that the shared package can be imported and used without either sampler.
Run relevant configured Python environments and wheel/sdist clean-install
checks when those areas change:

```bash
pixi run -e py311 test
pixi run -e py312 test
pixi run -e py313 test
pixi run -e build check-dist
```

CI follows Sequince's generated-lock cache/restore workflow and runs the same
Pixi tasks with `--locked`. `check-dist` builds both distributions, checks their
metadata, and installs each into a fresh environment without either sampler.
Keep only the current distributions in `dist/` when running it locally.
Report the commands actually run, failures, skips, and unavailable checks. Do not bypass hooks.

## Commit messages

Use uppercase SciPy-style prefixes for commit subjects and pull request titles:

```text
PREFIX(scope): short imperative summary
PREFIX: short imperative summary
PREFIX/PREFIX(scope): short imperative summary
```

Use a short scope such as phase, lisa, or amplitudes. Start it with a lowercase
letter and use only lowercase letters, digits, dots, underscores, or hyphens.
Omit the scope when it adds little information.

| Prefix | Use |
| --- | --- |
| BUG | Fix incorrect behavior |
| ENH | Add or extend functionality |
| PERF | Improve performance without changing intended behavior |
| API | Change a public interface or introduce an incompatibility |
| DEP | Deprecate functionality |
| MAINT | Refactor or maintain code without changing intended behavior |
| TST | Add or improve tests |
| DOC | Update documentation |
| BLD | Change packaging, dependencies, or build configuration |
| CI | Change continuous integration workflows |
| REL | Prepare or maintain a release |
| STY | Make formatting-only changes |

Use BUG consistently rather than FIX. Use one prefix by default, or at most
two when both describe the change. Begin the summary with a lowercase imperative
verb, omit a trailing period and whitespace, and aim for 72 characters or fewer.

```text
BUG(phase): preserve branch labels at angular seams
ENH(amplitudes): add circular-amplitude coordinates
DOC: clarify Jacobian conventions
```

Make commits in coherent units. Explain significant numerical or compatibility
changes in the body. Use Closes #123 only when the change fully resolves the
issue. The commit-message hook and PR-title workflow should use the same
validator; commit-message checks run on git commit, not pre-commit run --all-files.

## Pull requests and review

Push your branch and open a pull request targeting mj-will/gwreparam:main:

```bash
git push --set-upstream origin fix-descriptive-name
```

Use a draft pull request for work in progress. Explain the problem and resulting
behavior, relevant API and numerical changes, and actual validation results.
Include reproducible measurements for performance claims. Keep the title and
description aligned with the final implementation.

Respond to review feedback directly. Raise unrelated findings separately.
Maintainers should wait for applicable checks and review discussions before
merging and ensure the final commit message follows the conventions above.

## Releases

Releases are coordinated by maintainers. The build uses setuptools-scm
to derive versions from Git tags such as v0.1.0, without a separate hard-coded
version. Release notes should describe user-visible and compatibility changes.

Before publishing, verify the intended commit, checks, and built distributions
in a clean environment. The .github/workflows/publish.yml workflow will
publish to PyPI when a GitHub release is published. Verify the tag and artifacts
first, since published distributions cannot be overwritten.
