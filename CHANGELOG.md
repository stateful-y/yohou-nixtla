# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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
