# Contributing

## Branch & Commits
- Branch naming: `feat/...`, `fix/...`, `perf/...`, `docs/...`, `ci/...`
- Conventional commits: `feat:`, `fix:`, `ci:`, `docs:`, etc.

## Code Style & Hooks
```bash
pip install -r requirements.txt
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Tests
- `scripts/smoke_test.py` must pass locally and in CI
- No NaNs into `predict()` functions
- Timestamps must be UTC and sorted
- Monotonic lap numbers

## Artifacts
- Do not commit raw datasets (CSV files)
- CI bot is the only one committing to `models/` and `reports/`
- Manual commits to models/reports will be overwritten by nightly job

## Data Contracts
- Use Pydantic schemas for validation
- Assert no NaNs before model predictions
- Validate monotonicity where required (e.g., lap numbers)

## Model Training
- Train on Race 1 only
- Validate on Race 2 with walk-forward (causal, no leakage)
- Always write metadata.json with versions, git SHA, RNG seeds


