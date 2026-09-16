# Agent instructions

Before making changes, read and follow [CONTRIBUTING.md](CONTRIBUTING.md) and
[AI_POLICY.md](AI_POLICY.md). CONTRIBUTING.md defines the development setup,
code style, typing, testing, commit-message, and pull request conventions.

## Working in the repository

- Inspect the working tree before editing. Preserve existing work and do not
  overwrite or revert unrelated changes. When committing, stage only files
  relevant to the requested task.
- Keep changes focused. Follow existing interfaces and helpers, and avoid
  unrelated refactors. Add dependencies or abstractions only when they address
  a concrete need within the task.
- Keep generated lock files, local environments, caches, and generated outputs
  out of commits unless explicitly requested. Preserve existing local files.
- Keep the shared library independent of Sequince, Nessai, Bilby, and BBHx.
  Sampler adapters, branch sampling, and acceptance logic belong downstream.
- Keep docstrings concise. Put extended mathematical explanations in docs.
- Write focused tests of observable behavior and numerical invariants. Use
  conftest.py for shared fixtures and keep one-off setup local.

## Scientific correctness and validation

- Identify changes that affect Jacobians, branch multiplicities, parameter
  conventions, prior support, RNG behavior, or downstream checkpoint
  compatibility. Explain intentional behavior changes and test them.
- Preserve phase signs, angular periods, units, and invalid-proposal behavior
  during extraction. Do not introduce clipping or regularization silently.
- Distinguish coordinate changes, symmetry folds, and proposal mappings;
  do not assume every mapping preserves the waveform or target density.
- Follow CONTRIBUTING.md for validation. Report the checks actually run and
  their results, including failures, skips, and unavailable checks. Do not
  describe skipped checks as passing.
- Do not bypass hooks or disable checks to make a commit succeed. Address the
  failure, or report the blocker when it cannot be resolved within the task.
