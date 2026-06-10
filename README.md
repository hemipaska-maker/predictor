# predictor — ML-Driven Hardware Validation Engine

An open-source orchestration engine that executes hardware validation test
suites and aborts them early ("fail fast") when a machine learning model
predicts the overall suite is likely to fail. Built to reclaim expensive
bench, chamber, and machine time — without sacrificing determinism,
debuggability, or hardware safety.

> Working name: `predictor`. Rename before the first public release.

## How it works

```
test reads hardware  ──►  engine.ingest(test_id, metric, value)
                              │ 1. deterministic safety rules   ──► RuleAbort
                              │ 2. update fixed state vector (NaN = not run)
                              │ 3. strategy predicts P(suite failure)
                              │ 4. P >= threshold?               ──► MLThresholdAbort
                              ▼
                          test resumes — or the host runner catches the
                          abort and tears down hardware safely
```

Everything is synchronous and single-threaded: no queues, no background
threads, microsecond-scale inference, fully replayable decisions.

## Key properties

- **Framework-agnostic core** — `predictor.core` is pure stdlib. Pytest
  support is a thin opt-in plugin; any runner integrates the same way.
- **Strategy pattern** — models implement one interface and return a
  *calibrated probability*, never a decision. Swap Random Forest for
  Gradient Boosting (or your own model) without touching orchestration.
- **Rules before ML** — catastrophic readings (voltage spike, short) trigger
  a hardcoded abort that bypasses the model entirely. Rules protect
  hardware; the model saves time.
- **Schema-hash safety** — every trained model records the exact feature
  layout it saw; the engine refuses to run with a mismatched model.
- **MLOps included** — JSONL run capture, temporal-split retraining,
  a promotion gate driven by the business metric (machine time saved vs.
  good runs wrongly aborted), drift monitoring, and a versioned registry.

## Installation

```bash
pip install -e .                  # core only (zero dependencies)
pip install -e .[sklearn]         # + Random Forest / Gradient Boosting models
pip install -e .[pytest]          # + pytest plugin
pip install -e .[sklearn,pytest,dev]   # everything, for development
```

## Quickstart

```python
from predictor import BoundsRule, FeatureSchema, JsonlRecorder, ValidationEngine
from predictor.models import load_predictor

schema = FeatureSchema([
    ("test_voltage_rail", "vout"),
    ("test_voltage_rail", "ripple_mv"),
    ("test_thermal", "temp_c"),
])

engine = ValidationEngine(
    schema,
    load_predictor("registry/model_v3_1a2b3c4d.joblib"),
    threshold=0.85,                                   # abort at P(fail) >= 85%
    rules=[BoundsRule("vout", max_value=5.5)],        # hardware protection
    min_coverage=0.10,                                # prediction warm-up gate
    recorder=JsonlRecorder("records/runs.jsonl"),     # training-data capture
)

p = engine.ingest("test_voltage_rail", "vout", 3.31)  # may raise an abort
...
engine.finalize(suite_passed=True)                    # label the run
```

Catch `predictor.OrchestrationAbortError` in your runner, tear down the
hardware, and you are done.

### Pytest

```bash
pytest --hve-engine myproject.validation:make_engine
```

where `make_engine()` returns a configured `ValidationEngine`. Tests feed
measurements through the `ingest` fixture, which fills in the test's name as
`test_id` and returns the value unchanged for inline assertions:

```python
def test_3v3_rail(ingest, dmm, scope):
    assert ingest("vout", dmm.measure_voltage("3V3")) > 3.2
    assert ingest("ripple_mv", scope.measure_ripple_mv("3V3")) < 30
```

(The session-scoped `validation_engine` fixture is also available when you
need the engine itself.) An abort fails the current test, stops the session,
and still runs every teardown — hardware is released safely.

### Training & retraining

```python
from predictor.models import RandomForestPredictor
from predictor.training import ModelRegistry, build_dataset, should_promote, train_candidate

dataset = build_dataset(glob("records/*.jsonl"), schema.schema_hash)
result = train_candidate(dataset, RandomForestPredictor, threshold=0.85)

registry = ModelRegistry("registry/")
if should_promote(result.metrics, registry.latest_metrics(schema.schema_hash)):
    registry.save(result.predictor, result.metrics)
```

Run this on a schedule; `predictor.training.drift_report` tells you when the
model has gone stale between retrains.

## Choosing a model

The engine is not bound to any algorithm — it only knows the
`FailurePredictor` interface (one method returning P(suite failure) in
[0, 1]). Two adapters ship with the `[sklearn]` extra:

| Adapter | Backed by | Trade-off |
|---|---|---|
| `RandomForestPredictor` | `RandomForestClassifier` | Stable, robust to hardware noise. The recommended default. |
| `GradientBoostingPredictor` | `HistGradientBoostingClassifier` | Higher accuracy ceiling; ships strictly regularized to resist overfitting small datasets. |

Both are trained through the same `train_candidate(...)` call (pass the
class), calibrated so thresholds carry real probability meaning, and saved
with a `kind` tag — `load_predictor(path)` reconstructs the right adapter
automatically, so deployments never hardcode a model type.

To bring your own model, either subclass
`predictor.models.SklearnPredictor` (gets training, calibration, and
persistence for free) or implement `predictor.FailurePredictor` directly —
even a hand-written heuristic works:

```python
from predictor import FailurePredictor

class HotVrmHeuristic(FailurePredictor):
    def predict_failure_probability(self, state):
        temp = state[6]                      # index from your FeatureSchema
        return 0.95 if temp == temp and temp > 70 else 0.05  # NaN-aware

engine = ValidationEngine(schema, HotVrmHeuristic(), threshold=0.85)
```

The orchestration code never changes when the model does.

## Demo

A complete, realistic walkthrough — shadow-mode data capture, training and
promotion, live bench sessions with ML and rule aborts, and a drift check on
a simulated hardware revision:

```bash
pip install -e .[sklearn]
python examples/end_to_end.py        # add -v to stream the library's logs
```

## Documentation

- [DESIGN.md](DESIGN.md) — full architecture and the reasoning behind it
- [AGENTS.md](AGENTS.md) — invariants and conventions for contributors
  (human or AI)

## Development

```bash
pip install -e .[sklearn,pytest,dev]
python -m pytest tests/ -q
```

The library logs through `logging` (logger names under `predictor.*`) and
emits nothing unless your application configures handlers.

## License

MIT — see [LICENSE](LICENSE).
