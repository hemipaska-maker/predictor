# ML-Driven Hardware Validation Engine — Design Document

**Status:** Draft v0.1 (2026-06-10)
**Working package name:** `predictor` (placeholder — rename before publishing)

## 1. Overview

An independent, open-source orchestration engine that executes hardware
validation test suites and aborts them early ("fail fast") when a machine
learning model predicts that the overall suite is likely to fail. The goal is
to reclaim expensive bench/chamber/machine time without sacrificing
determinism or debuggability.

### Goals

- Predict suite-level failure probability in real time as tests emit data.
- Abort cleanly via exceptions when a user-defined threshold is crossed.
- Stay framework-agnostic: pure-Python core, zero test-framework dependencies.
- Support deterministic rule-based overrides for catastrophic events.
- Provide an MLOps path: data capture → retraining → drift monitoring.

### Non-Goals

- Test scheduling/parallelization (the host runner owns ordering).
- Background threads, queues, or async execution — everything is synchronous.
- Making pass/fail decisions inside the model — models emit probabilities
  only; the engine owns the decision.

## 2. Architecture Overview

```
┌─────────────────────────────  Host test runner (pytest, custom, …)  ─┐
│                                                                       │
│  test_voltage_rail()                                                  │
│      │  engine.ingest("test_voltage_rail", "vout", 3.31)              │
│      ▼                                                                │
│  ┌──────────────────────── ValidationEngine (core) ────────────────┐  │
│  │ 1. Rule checks (deterministic)   ──► RuleAbort (bypasses ML)    │  │
│  │ 2. StateVector.update(idx, val)                                 │  │
│  │ 3. predictor.predict_failure_probability(state)  ── strategy ──►│  │
│  │ 4. p >= threshold?               ──► MLThresholdAbort           │  │
│  │ 5. RunRecorder.log(state, p)     (training-data capture)        │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│      │ returns p (or raises OrchestrationAbortError)                  │
│      ▼                                                                │
│  test resumes — or runner catches abort, tears down hardware safely   │
└───────────────────────────────────────────────────────────────────────┘
```

Layering (dependency direction is strictly downward):

| Layer | Package | Dependencies |
|---|---|---|
| Integrations | `predictor.integrations` (pytest plugin, …) | core only |
| Models | `predictor.models` (sklearn strategies) | core + optional `scikit-learn` |
| Training/MLOps | `predictor.training` | models |
| **Core** | `predictor.core` | **stdlib only** |

## 3. Core Components

### 3.1 `FeatureSchema` and `StateVector` (fixed-vector state management)

ML models need a fixed-width input. The suite's feature layout is declared
**up front** as an ordered list of `(test_id, metric)` pairs:

```python
schema = FeatureSchema([
    ("test_voltage_rail", "vout"),
    ("test_voltage_rail", "ripple_mv"),
    ("test_thermal", "temp_c"),
    ...
])
```

- The `StateVector` is initialized to the placeholder `NaN` meaning
  *"test not yet run"*. As tests emit data, the corresponding index is
  overwritten in place.
- `NaN` is the canonical placeholder in the **core**. Model adapters decide
  how to handle it: `HistGradientBoostingClassifier` consumes NaN natively;
  Random Forest adapters impute (e.g. to `-1`) inside the adapter. The core
  never bakes in a model-specific sentinel.
- `FeatureSchema.schema_hash` is a SHA-256 over the canonical layout. Every
  trained model artifact stores the hash it was trained against; the engine
  **refuses to load a model whose hash doesn't match** the live schema. This
  prevents the silent killer of this architecture: feature/index drift after
  someone adds or reorders a test.
- `StateVector.coverage()` reports the fraction of features observed so far —
  used for the prediction warm-up gate (§3.4).

### 3.2 `FailurePredictor` (Strategy pattern)

```python
class FailurePredictor(ABC):
    @abstractmethod
    def predict_failure_probability(self, state: Sequence[float]) -> float:
        """Return P(suite failure) in [0.0, 1.0]. Never a binary decision."""
```

- The core depends only on this interface. Swapping Random Forest for
  Gradient Boosting (or a heuristic, or a remote model) requires no engine
  changes.
- Implementations must be **pure functions of the state** (no internal
  mutable state across calls) so that replaying a run log reproduces the
  exact same decisions.
- **Calibration requirement:** raw `predict_proba` outputs from tree
  ensembles are poorly calibrated. Since users set thresholds in probability
  terms ("abort at 85%"), shipped models are wrapped in
  `CalibratedClassifierCV` (isotonic/sigmoid) during training so that 0.85
  actually means ~85%.

Provided strategies (optional `predictor[sklearn]` extra):

| Strategy | Trade-off |
|---|---|
| `RandomForestPredictor` | Stable, robust to hardware noise, low variance. Default choice. |
| `GradientBoostingPredictor` | Higher accuracy ceiling; requires strict regularization (max depth, learning rate, early stopping) to avoid overfitting small datasets. Uses `HistGradientBoostingClassifier` for native NaN support. |

### 3.3 Rule-based overrides (deterministic, pre-ML)

Rules run **before** the state update and the model, on every ingested value:

```python
class Rule(ABC):
    def check(self, test_id, metric, value) -> str | None:
        """Return a violation reason, or None."""
```

- `BoundsRule(metric="rail_voltage", max=5.5)` → a 12 V spike on a 3.3 V rail
  raises `RuleAbort` immediately. The ML model is never consulted.
- Rules exist to **protect hardware**; the ML model exists to **save time**.
  They must never be merged into one mechanism — a rule firing is a hard
  engineering fact, a prediction is a statistical bet.

### 3.4 `ValidationEngine` (orchestration core)

```python
engine = ValidationEngine(
    schema=schema,
    predictor=RandomForestPredictor.load("model.joblib", schema),
    threshold=0.85,
    rules=[BoundsRule("rail_voltage", max_value=5.5)],
    min_coverage=0.10,          # warm-up gate
    recorder=JsonlRecorder(...),  # optional training-data capture
)

p = engine.ingest("test_voltage_rail", "vout", 3.31)
```

`ingest()` flow (single-threaded, synchronous):

1. Evaluate rules → may raise `RuleAbort`.
2. Update the state vector index.
3. If `coverage < min_coverage`, skip prediction (early predictions on a
   nearly-all-NaN vector are noise; the warm-up gate prevents spurious aborts
   on the first measurement).
4. Call the strategy → probability `p`.
5. Append `(timestamp, test_id, metric, state snapshot, p)` to the recorder.
6. If `p >= threshold` → raise `MLThresholdAbort(probability=p, ...)`.
7. Return `p` to the caller.

Inference on a fitted tree ensemble over a few-hundred-element vector is
microseconds-to-low-milliseconds — negligible against hardware settling
times, and fully deterministic.

### 3.5 Exception hierarchy (fail-fast mechanism)

```
OrchestrationAbortError(Exception)     # catch-all for "engine halted the run"
├── MLThresholdAbort                   # carries .probability, .threshold
└── RuleAbort                          # carries .rule, .reason, the offending value
```

Because execution is single-threaded, raising propagates straight up through
the running test into the host runner, which performs its normal teardown
(disconnect instruments, power down DUT, release the bench). The engine never
touches hardware itself.

## 4. Training Data Capture & MLOps

### 4.1 Run records

The `RunRecorder` interface captures, per run, every `(state snapshot,
prediction)` pair plus the **final suite outcome** (pass/fail, labeled by the
host runner at session end via `engine.finalize(outcome)`). Default
implementation: append-only JSONL — trivially diffable, greppable, and
mergeable across benches.

Each record carries the `schema_hash`, so training pipelines can partition
historical data by schema version and never mix incompatible layouts.

### 4.2 Retraining pipeline (concept drift)

Hardware revisions, firmware updates, and test-suite changes all decay model
accuracy. The training package provides a scriptable pipeline meant to run on
a schedule (cron/CI):

1. **Build dataset** from JSONL run records (latest schema version only).
   Each intermediate state snapshot is a training row; the label is the final
   suite outcome — this teaches the model to predict from *partial* vectors.
2. **Train** candidate (RF and/or GB) with calibration on a temporal split
   (train on older runs, validate on newest — never a random split, which
   leaks future hardware behavior into the past).
3. **Evaluate** against the currently deployed model on the holdout:
   - Brier score / calibration curve (threshold semantics depend on it),
   - simulated cost metric: *machine-minutes saved vs. good-runs wrongly
     aborted* at the configured threshold. This, not accuracy, is the
     business metric.
4. **Promote** only if the candidate wins; artifacts are versioned
   (`model_v{N}_{schema_hash[:8]}.joblib`) with a metadata sidecar
   (training window, metrics, calibration method).
5. **Drift monitor:** a lightweight job compares recent prediction
   distributions and realized outcomes against training-time baselines
   (e.g. PSI on inputs, rolling Brier on outputs) and flags when retraining
   is overdue rather than waiting for the schedule.

## 5. Framework Integrations

### 5.1 Pytest plugin (`predictor.integrations.pytest_plugin`)

Thin, opt-in (`pip install predictor[pytest]`), and dumb by design:

- A session-scoped `validation_engine` fixture built from a user-named
  factory (`--hve-engine module:factory`).
- Tests feed measurements via the function-scoped `ingest` fixture —
  `ingest("vout", 3.31)` — which fills `test_id` with the test function's
  name (parametrization suffix stripped) and returns the value unchanged so
  it can be asserted on inline. `engine.ingest(...)` remains available for
  full control.
- A hook wrapper catches `OrchestrationAbortError`, marks the current test as
  failed with the abort reason, and calls `session.shouldstop` so pytest's own
  teardown/fixture finalization runs normally.
- `pytest_sessionfinish` calls `engine.finalize(outcome)` to label the run
  record.

All logic lives in the core; the plugin only translates between pytest's
lifecycle and the engine's API. Other runners (Robot Framework, in-house
sequencers) integrate the same way.

## 6. Project Layout

```
predictor/
├── pyproject.toml              # core has zero deps; extras: sklearn, pytest
├── README.md / DESIGN.md / AGENTS.md
├── examples/
│   └── end_to_end.py           # full-lifecycle runnable demo
├── src/predictor/
│   ├── core/                   # stdlib only — the agnostic engine
│   │   ├── engine.py           # ValidationEngine
│   │   ├── state.py            # FeatureSchema, StateVector
│   │   ├── strategy.py         # FailurePredictor ABC
│   │   ├── rules.py            # Rule ABC, BoundsRule
│   │   ├── records.py          # RunRecorder, JsonlRecorder
│   │   └── exceptions.py       # OrchestrationAbortError hierarchy
│   ├── models/                 # [sklearn] extra
│   │   └── sklearn_models.py   # RF / HistGB adapters + joblib persistence
│   ├── training/               # dataset build, train, evaluate, drift
│   └── integrations/
│       └── pytest_plugin.py    # [pytest] extra, entry-point registered
└── tests/
```

## 7. Testing Strategy

- **Core:** pure unit tests with a stub predictor (returns scripted
  probabilities) — verify warm-up gating, threshold semantics (`>=`), rule
  precedence over ML, state-vector indexing, schema-hash mismatch rejection.
- **Models:** train on synthetic data; assert probability bounds, NaN
  handling per adapter, save/load round-trip including schema hash.
- **Determinism test:** replay a recorded run log twice → identical
  prediction sequence and identical abort point.
- **Plugin:** `pytester`-based tests that a threshold crossing stops the
  session and teardown still runs.

## 8. Open Questions / Roadmap

1. **Threshold hysteresis** — should one spike above threshold abort, or
   require N consecutive predictions above it? (v1: single-shot, configurable
   later.)
2. **Explainability** — surface per-feature contributions (e.g. tree SHAP) in
   the abort message so engineers trust the abort. Post-v1.
3. **Multi-suite / DUT-family models** — one model per schema hash for now;
   shared embeddings are out of scope.
4. **Categorical/string metrics** — v1 is numeric-only; encoders later.
5. **Naming** — `predictor` is a placeholder; pick a real name before the
   first release.
