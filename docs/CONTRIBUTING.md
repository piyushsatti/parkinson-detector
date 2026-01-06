# Contributing Guide

A lightweight guide for improving documentation, recipes, and data prep without changing project scope.

## Repo conventions
- Use Python 3.10+ with Poetry; install via `make install`.
- Keep changes reproducible: prefer edits to YAML/recipes and scripts over ad hoc notebooks.
- Commit style: short imperative subject, e.g., `Improve README setup steps`.
- Branch naming: `feature/<short-description>` or `chore/<short-description>`.

## Run locally
```bash
git clone <your-repo-url>
cd parkinson-detector
make install
# Optional dataset download (expects Italian Parkinson Voice & Speech)
make download
make data  # build manifests
```

## Adding a change safely
1. Update or create documentation alongside code (e.g., README, docs/Architecture.md).
2. Prefer mechanical refactors (renames/moves) that do not alter training logic; keep recipe behavior intact.
3. Validate locally with the Make targets you touched (`make data`, `make train MODEL=...`, or `make predict ...`).
4. Avoid adding new tests or CI; this project stays lean for portfolio review.
