# AGENTS.md — guidance for coding agents and contributors

This file states the load-bearing invariants of the codebase. Violating one
of these is a bug even if all tests pass. Read [DESIGN.md](DESIGN.md) for the
reasoning behind them.

## Project overview

`predictor` is an ML-driven fail-fast engine for hardware validation suites:
tests stream measurements into a `ValidationEngine`, which maintains a
fixed-width state vector, consults deterministic safety rules and then a
calibrated ML strategy, and raises `OrchestrationAbortError` subclasses to
halt a run early.

## Commands

```bash
pip install -e .[sklearn,pytest,dev]   # dev setup
python -m pytest tests/ -q             # full test suite (must pass)
python examples/end_to_end.py          # runnable demo (-v for library logs)
```

## Layout and dependency direction

```
src/predictor/
├── core/           # stdlib ONLY — engine, state, strategy ABC, rules,
│                   # records, exceptions
├── models/         # sklearn adapters       (depends on core + [sklearn])
├── training/       # dataset/train/evaluate/drift/registry (depends on models)
└── integrations/   # pytest plugin          (depends on core + [pytest])
```

Dependencies point strictly downward to `core`. `core` must never import
numpy, sklearn, pytest, or anything outside the standard library. The
integrations/models/training layers must never import from each other's
internals — only from `core` and their own optional extra.

## Invariants

1. **Models return probabilities, never decisions.** A `FailurePredictor`
   yields a calibrated P(suite failure) in [0, 1]. The threshold comparison
   and the abort live in `ValidationEngine` only.
2. **NaN is the canonical "test not yet run" placeholder in core.** Model
   adapters own NaN handling (the RF adapter imputes + appends an observed
   mask; HistGB consumes NaN natively). The transform lives in the adapter's
   `transform()` classmethod and MUST be applied identically at training and
   inference — never duplicate it elsewhere.
3. **Schema hash gates everything.** Any change to the feature layout changes
   `FeatureSchema.schema_hash`; artifacts store the hash they were trained
   against and the engine refuses mismatches. Never weaken or bypass this
   check, and never reorder schema features "harmlessly".
4. **Rules fire before the state update and before the model.** Rule checks
   are hardware protection; they must stay deterministic, dependency-free,
   and must never be merged into or gated by the ML path.
5. **Predictors are deterministic, stateless functions of the input vector.**
   Replaying a recorded run must reproduce the identical prediction sequence
   and abort point. No randomness at inference, no mutable state across calls.
6. **Training splits are temporal and per-run.** Holdout = newest runs.
   Random row-level splits leak a run's own future snapshots (and future
   hardware behavior) into training — never introduce one.
7. **The promotion gate ranks on the business metric.** False-abort cap
   first, then machine time saved, Brier as tiebreaker (`should_promote`).
   Raw accuracy is not a promotion criterion.
8. **Single-threaded by design.** Do not add background threads, queues, or
   async paths to the ingest flow; clean exception-based teardown depends on
   synchronous execution.

## Conventions

- Python ≥ 3.10, `from __future__ import annotations`, type hints on public
  APIs. Plain JavaScript-style cleverness is out; explicit is in.
- The library never prints. Use module-level `logger =
  logging.getLogger(__name__)`; the package root installs a `NullHandler`.
  Hot-path logging (per-ingest) stays at DEBUG with %-style lazy args.
  `print` is acceptable only in `examples/` as the demo's report UI.
- Errors: raise `ValueError` with an actionable message for misconfiguration;
  reserve the `OrchestrationAbortError` hierarchy strictly for engine-
  initiated halts.
- Tests: core behavior is tested with scripted stub predictors (no sklearn);
  model/training tests use small synthetic datasets with planted signals;
  plugin behavior is tested through `pytester` sub-runs. New features need
  tests at the matching layer.
- Keep `pyproject.toml` extras honest: a module importing an optional
  dependency must live in the layer whose extra declares it.

## Things that look like bugs but are decisions

- `ingest()` records the snapshot *before* the threshold check — the row that
  triggers an abort is exactly what the next retraining needs.
- `JsonlRecorder` opens the file per write — crash-safety on bench machines
  outweighs throughput.
- The evaluation replays the abort policy run-by-run instead of computing
  row-level classification metrics — machine time is the unit of value.
- Unfinalized runs are dropped (with a warning) when building datasets — an
  abort still gets finalized by the runner, so a missing label means the
  outcome is genuinely unknown, not "failed".
