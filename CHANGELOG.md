# Changelog

All notable changes to this repository will be documented in this file.

The project follows a lightweight, research-friendly approach to versioning:
major milestones are recorded as tagged releases when appropriate.

## Unreleased

## kk-ready-2026-02-05
- Phase0 shared params are read via `tools/shared_env.py` (no wrapper copy-paste).
- `core_calib_check.py` enforces Phase0 presence + cross-module consistency.
- `rel` Phase0 `c0` sourced from env; wrapper_version aligned.
- Ringdown engine no longer crashes on empty bootstrap; outputs quantiles as `None` with warn.
- `.gitattributes` added for stable line endings.

## 2026-01-11

### Added
- Initial repository bootstrap (README, MIT License, Python .gitignore)
- Governance documents: CONTRIBUTING.md, SECURITY.md
- Citation support: CITATION.cff (GitHub “Cite this repository” enabled)
- Documentation: RU/EN methodological manifestos and docs index
- Modules directory with initial rotation-curves module and external links
- README navigation to docs/ and modules/
- .gitattributes for LF normalization (Windows-friendly diffs)
