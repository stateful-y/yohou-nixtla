# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0-alpha.6] - 2026-08-10

This **minor release** includes 20 commits.


### Bug Fixes
- Pin exact uv version in setup-uv steps (template v0.29.6)  ([#46](https://github.com/stateful-y/yohou-nixtla/pull/46)) by @gtauzin
- Pin ossf/scorecard-action to the existing v2.4.4 tag by @gtauzin
- Stop the nightly coverage upload from silently uploading the wrong report  ([#64](https://github.com/stateful-y/yohou-nixtla/pull/64)) by @gtauzin

### Refactoring
- Move build output to .artifacts/, config into .github/  ([#63](https://github.com/stateful-y/yohou-nixtla/pull/63)) by @gtauzin

### Testing
- Isolate lightning_logs per test to fix xdist race  ([#36](https://github.com/stateful-y/yohou-nixtla/pull/36)) by @gtauzin

### Miscellaneous Tasks
- Move the docs build out of hooks.py and widen the ruff ignores (template v0.27.3)  ([#37](https://github.com/stateful-y/yohou-nixtla/pull/37)) by @gtauzin
- Render API page structure from mkdocstrings templates (template v0.28.1)  ([#40](https://github.com/stateful-y/yohou-nixtla/pull/40)) by @gtauzin
- Discover the API surface with Griffe (template v0.28.3)  ([#42](https://github.com/stateful-y/yohou-nixtla/pull/42)) by @gtauzin
- Replace stale git hooks by installing with prek install -f (template v0.28.4)  ([#43](https://github.com/stateful-y/yohou-nixtla/pull/43)) by @gtauzin
- Make the generated docs build engine-independent (template v0.29.3)  ([#44](https://github.com/stateful-y/yohou-nixtla/pull/44)) by @gtauzin
- Migrate the docs engine from MkDocs to Zensical (template v0.30.1)  ([#47](https://github.com/stateful-y/yohou-nixtla/pull/47)) by @gtauzin
- Add pre-push gates and a single CI roll-up check (template v0.32.1)  ([#50](https://github.com/stateful-y/yohou-nixtla/pull/50)) by @gtauzin
- Restrict workflow permissions and add secret scanning (template v0.35.0)  ([#51](https://github.com/stateful-y/yohou-nixtla/pull/51)) by @gtauzin
- Switch Codecov to OIDC and pin the Scorecard action (template v0.36.0)  ([#52](https://github.com/stateful-y/yohou-nixtla/pull/52)) by @gtauzin
- Document signing release tags with gitsign (template v0.37.0)  ([#53](https://github.com/stateful-y/yohou-nixtla/pull/53)) by @gtauzin
- Add a CLAUDE.md project-instructions file for AI assistants (template v0.38.0)  ([#54](https://github.com/stateful-y/yohou-nixtla/pull/54)) by @gtauzin
- Fix three release-pipeline defects (template v0.39.0)  ([#55](https://github.com/stateful-y/yohou-nixtla/pull/55)) by @gtauzin
- Let Renovate see the SBOM tool's version pin (template v0.39.1)  ([#56](https://github.com/stateful-y/yohou-nixtla/pull/56)) by @gtauzin
- Add a nightly job that exercises the release path (template v0.40.0)  ([#57](https://github.com/stateful-y/yohou-nixtla/pull/57)) by @gtauzin
- Fix a shell injection in the release publish job (template v0.40.1)  ([#62](https://github.com/stateful-y/yohou-nixtla/pull/62)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.5] - 2026-07-20

This **minor release** includes 6 commits and one breaking API change.


### Breaking Changes
- Rename the forecaster slot `feature_transformer` to `actual_transformer`, following yohou; no deprecation alias, so callers must rename at the call site  ([#31](https://github.com/stateful-y/yohou-nixtla/pull/31)) by @gtauzin
- Move the yohou pin to `>=0.1.0a11,<0.2.0`, the first release carrying the rename  ([#31](https://github.com/stateful-y/yohou-nixtla/pull/31)) by @gtauzin

### Bug Fixes
- Honor `predict(groups=...)` for panel data; non-requested group columns were mis-classified as globals, so every group was returned  ([#26](https://github.com/stateful-y/yohou-nixtla/pull/26)) by @gtauzin

### Documentation
- Rename `BaseTransformer` to `BaseActualTransformer` in the transformer parameter docstrings, matching yohou's forecast-transformer hierarchy  ([#26](https://github.com/stateful-y/yohou-nixtla/pull/26)) by @gtauzin

### Testing
- Register the `Pandas4Warning` filter from a root `conftest.py` only when pandas 3 is installed; pytest resolved the static `filterwarnings` entry at startup and aborted on pandas 2.x  ([#27](https://github.com/stateful-y/yohou-nixtla/pull/27)) by @gtauzin
- Xfail `check_clone_preserves_forecaster_params` for the neural wrappers, whose torch loss default compares by identity  ([#26](https://github.com/stateful-y/yohou-nixtla/pull/26)) by @gtauzin

### Dependencies
- Bump pandas 2.3.3 -> 3.0.3 and mkdocs-material 9.7.6 -> 9.7.7; statsforecast resolves back to 2.0.x, since 2.1.0 caps pandas below 3  ([#33](https://github.com/stateful-y/yohou-nixtla/pull/33)) by @dependabot

### Miscellaneous Tasks
- Update from template v0.26.1: API cross-referencing across docs pages, `uv.lock` as the single source of truth for lint tooling, restored yohou[nixtla] logos  ([#28](https://github.com/stateful-y/yohou-nixtla/pull/28)) by @gtauzin
- Update from template v0.27.0: commit-message CI workflow, refreshed pre-commit hooks and nox sessions  ([#32](https://github.com/stateful-y/yohou-nixtla/pull/32)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.4] - 2026-05-11

This **minor release** includes 2 commits.


### Features
- Update API from X to X_actual, add X_future and X_forecast support  ([#21](https://github.com/stateful-y/yohou-nixtla/pull/21)) by @gtauzin

### Miscellaneous Tasks
- Drop Python 3.14 from nightly matrix  ([#20](https://github.com/stateful-y/yohou-nixtla/pull/20)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.3] - 2026-04-22

This **minor release** includes 2 commits.


### Refactoring
- Simplify custom estimator API based on latest yohou interface  ([#18](https://github.com/stateful-y/yohou-nixtla/pull/18)) by @gtauzin

### Miscellaneous Tasks
- Update from template v0.17.0-v0.18.0 and rework docs with Diataxis  ([#14](https://github.com/stateful-y/yohou-nixtla/pull/14)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.2] - 2026-03-01

This **minor release** includes 3 commits.


### Bug Fixes
- Isolate Lightning log dirs to fix parallel test race  ([#5](https://github.com/stateful-y/yohou-nixtla/pull/5)) by @gtauzin

### Miscellaneous Tasks
- Update from copier template v0.14.0  ([#7](https://github.com/stateful-y/yohou-nixtla/pull/7)) by @gtauzin
- Update from copier template v0.15.0  ([#8](https://github.com/stateful-y/yohou-nixtla/pull/8)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.1] - 2026-02-20

This **minor release** includes 1 commit.

- Initial commit

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [Unreleased]

### Added
- Initial project setup
