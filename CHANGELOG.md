# ExoMiner Pipeline Changelog

User-visible changes, newest release on top.

**Convention:** as work lands, note user-visible changes under `## Unreleased`. At
release time, rename that section to the new version (e.g., `## 2.3.0 — 2026-10-12`).
`create_podman_image.sh` prompts before building if the top section does not match
the requested version. Record only things a user can observe (table columns, scores,
flags, image contents) — internal refactors belong in the git history.

## Unreleased

### BREAKING

- Prediction/ranking tables now carry separate TOI columns:
  - `toiId` — target star TOI ID (e.g., "1234"); or `not_toi` if it has not been deemed a TOI yet
  - `toiMatchId` — matched TOI ID from SPOC DV (e.g., "1234.01"); or `not_matched` if no match found
  - `toiMatchCorrelation` — correlation score between TCE and matched TOI (0–1); or -1 if no match found
- **Removed:** the previous `toiId` (previously the *matched* TOI ID) and `toiCorrelation`
  columns. Old `toiCorrelation` data is now in `toiMatchCorrelation`.

### Changed

- Pipeline console logging now includes a timestamp and severity level on each line

### Fixed

- Removed a duplicated "Documentation can be found at..." line at pipeline startup
- `create_podman_image.sh` header comment now matches the script's actual filename

_Releases prior to 2.3.0 were not tracked in this file; see the git history of the
ExoMiner repository (https://github.com/nasa/Exominer) for release-to-release details._
