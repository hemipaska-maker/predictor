"""End-to-end demo: ML-driven fail-fast validation of a 12V power board.

Scenario
--------
A bench validates power boards with an 8-test suite (~55 minutes of machine
time per board, dominated by burn-in and an EMC chamber scan). The demo walks
the full lifecycle of this engine:

  PART 1  Shadow mode   - 120 historical runs are recorded to JSONL with no
                          fail-fast active (this is how training data is born).
  PART 2  Training      - Random Forest and Gradient Boosting candidates are
                          trained on a temporal split, evaluated on the
                          business metric, and the winner is promoted into a
                          versioned model registry.
  PART 3  Live bench    - the promoted model + hard safety rules drive three
                          sessions: a healthy board, a degrading VRM (ML abort
                          saves ~40 min), and a short circuit (rule abort).
  PART 4  Drift check   - a simulated hardware revision shifts thermals; the
                          drift monitor flags the affected features.

Run it:    python examples/end_to_end.py
Verbose:   python examples/end_to_end.py -v   (streams the library's logs)

Note on output: the formatted report below is printed deliberately — it is
this script's user interface. The library itself never prints; it logs
through `logging` and stays silent unless the application opts in (as the
-v flag does here).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predictor import (
    BoundsRule,
    FeatureSchema,
    JsonlRecorder,
    MLThresholdAbort,
    OrchestrationAbortError,
    RuleAbort,
    StateVector,
    ValidationEngine,
)
from predictor.models import (
    GradientBoostingPredictor,
    RandomForestPredictor,
    load_predictor,
)
from predictor.training import (
    ModelRegistry,
    build_dataset,
    drift_report,
    evaluate,
    should_promote,
    temporal_split,
    train_candidate,
)

OUT = Path(__file__).resolve().parent / "demo_output"
ABORT_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# The validation suite: (test_id, [metrics], duration in seconds).
# Cheap electrical checks run first; burn-in and the EMC chamber dominate cost.
# ---------------------------------------------------------------------------
SUITE = [
    ("test_power_on",       ["vbus_v", "inrush_a"],               8),
    ("test_3v3_rail",       ["vout_v", "ripple_mv"],             12),
    ("test_5v_rail",        ["vout_v", "ripple_mv"],             12),
    ("test_vrm_thermal",    ["temp_c", "temp_rise_c"],           45),
    ("test_load_step",      ["sag_mv", "recovery_us"],           30),
    ("test_clock_integrity",["jitter_ps", "freq_error_ppm"],     60),
    ("test_burn_in",        ["peak_temp_c", "current_drift_pct"],900),
    ("test_emc_scan",       ["worst_margin_db"],               1200),
]

SCHEMA = FeatureSchema(
    [(test_id, metric) for test_id, metrics, _ in SUITE for metric in metrics]
)
DURATION = {test_id: secs for test_id, _, secs in SUITE}
TOTAL_SECS = sum(DURATION.values())

# Healthy readings: (mean, sigma) per (test_id, metric).
NOMINAL = {
    ("test_power_on", "vbus_v"):            (12.00, 0.03),
    ("test_power_on", "inrush_a"):          (1.40, 0.10),
    ("test_3v3_rail", "vout_v"):            (3.300, 0.010),
    ("test_3v3_rail", "ripple_mv"):         (12.0, 2.0),
    ("test_5v_rail", "vout_v"):             (5.000, 0.015),
    ("test_5v_rail", "ripple_mv"):          (18.0, 3.0),
    ("test_vrm_thermal", "temp_c"):         (52.0, 2.0),
    ("test_vrm_thermal", "temp_rise_c"):    (17.0, 2.0),
    ("test_load_step", "sag_mv"):           (45.0, 6.0),
    ("test_load_step", "recovery_us"):      (80.0, 10.0),
    ("test_clock_integrity", "jitter_ps"):  (9.0, 1.5),
    ("test_clock_integrity", "freq_error_ppm"): (1.2, 0.4),
    ("test_burn_in", "peak_temp_c"):        (71.0, 3.0),
    ("test_burn_in", "current_drift_pct"):  (0.7, 0.3),
    ("test_emc_scan", "worst_margin_db"):   (7.0, 1.5),
}

# Failure modes: additive offsets to the nominal means. Each mode leaves an
# early fingerprint long before the test that finally fails.
FAILURE_MODES = {
    # Degrading voltage regulator: runs hot early, fails burn-in at the end.
    "vrm_degraded": {
        ("test_vrm_thermal", "temp_c"):        +18.0,
        ("test_vrm_thermal", "temp_rise_c"):   +12.0,
        ("test_load_step", "sag_mv"):          +35.0,
        ("test_burn_in", "peak_temp_c"):       +24.0,
        ("test_burn_in", "current_drift_pct"): +1.8,
    },
    # Noisy supply: elevated ripple and jitter early, fails the EMC scan.
    "supply_noise": {
        ("test_3v3_rail", "ripple_mv"):        +18.0,
        ("test_5v_rail", "ripple_mv"):         +24.0,
        ("test_clock_integrity", "jitter_ps"): +7.0,
        ("test_emc_scan", "worst_margin_db"):  -9.0,
    },
}


def simulate_run(rng: np.random.Generator, mode: str | None):
    """Yield (test_id, metric, value) readings for one board, in suite order."""
    offsets = FAILURE_MODES.get(mode, {})
    for test_id, metrics, _ in SUITE:
        for metric in metrics:
            mean, sigma = NOMINAL[(test_id, metric)]
            value = rng.normal(mean + offsets.get((test_id, metric), 0.0), sigma)
            yield test_id, metric, value


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def prob_bar(p: float | None, width: int = 24) -> str:
    if p is None:
        return "[ warming up" + " " * (width - 11) + "]   --"
    filled = int(round(p * width))
    return f"[{'#' * filled}{'.' * (width - filled)}]  {p:5.1%}"


def fmt_metrics(m: dict) -> str:
    return (
        f"catch={m['early_catch_rate']:.0%}  "
        f"saved={m['mean_fraction_saved']:.0%}  "
        f"false_aborts={m['false_abort_rate']:.0%}  "
        f"brier={m['brier']:.3f}"
    )


# ---------------------------------------------------------------------------
# PART 1 - shadow mode: record history, no fail-fast yet
# ---------------------------------------------------------------------------
def generate_history(rng: np.random.Generator, n_runs: int, path: Path) -> dict:
    recorder = JsonlRecorder(path)
    counts = {"healthy": 0, "vrm_degraded": 0, "supply_noise": 0}
    for _ in range(n_runs):
        mode = rng.choice(
            ["healthy", "vrm_degraded", "supply_noise"], p=[0.72, 0.14, 0.14]
        )
        counts[mode] += 1
        state = StateVector(SCHEMA)
        for test_id, metric, value in simulate_run(rng, mode):
            state.update(test_id, metric, value)
            recorder.record(test_id, metric, state.snapshot(), None, SCHEMA.schema_hash)
        recorder.finalize(suite_passed=mode == "healthy")
    return counts


# ---------------------------------------------------------------------------
# PART 3 - live bench sessions against the promoted model
# ---------------------------------------------------------------------------
SAFETY_RULES = [
    # Catastrophic events bypass the ML model entirely (DESIGN.md §3.3).
    BoundsRule("vbus_v", min_value=10.5, max_value=13.5),
    BoundsRule("peak_temp_c", max_value=110.0),
]


def run_bench_session(name: str, mode: str | None, model_path: Path,
                      rng: np.random.Generator) -> None:
    print(f"\n--- Session: {name} " + "-" * (50 - len(name)))
    engine = ValidationEngine(
        SCHEMA,
        load_predictor(str(model_path)),
        threshold=ABORT_THRESHOLD,
        rules=SAFETY_RULES,
        min_coverage=0.10,
        recorder=JsonlRecorder(OUT / "live_records.jsonl"),
    )
    elapsed = 0.0
    current_test = None
    try:
        for test_id, metric, value in simulate_run(rng, mode):
            if test_id != current_test:
                current_test = test_id
                elapsed += DURATION[test_id]
            line = f"  {test_id:<22} {metric:<18} {value:9.2f}"
            try:
                p = engine.ingest(test_id, metric, value)
            except OrchestrationAbortError:
                print(line + "   << abort triggered by this reading")
                raise
            print(line + f"   {prob_bar(p)}")
        engine.finalize(suite_passed=True)
        print(f"  >> SUITE PASSED in {TOTAL_SECS / 60:.0f} min of machine time.")
    except RuleAbort as exc:
        engine.finalize(suite_passed=False)
        print(f"  !! RULE ABORT (ML bypassed): {exc}")
        print(f"  >> Hardware protected after ~{elapsed / 60:.0f} min; "
              f"runner tears down the bench safely.")
    except MLThresholdAbort as exc:
        engine.finalize(suite_passed=False)
        saved = TOTAL_SECS - elapsed
        print(f"  !! ML ABORT: P(failure)={exc.probability:.1%} "
              f">= threshold {exc.threshold:.0%}")
        print(f"  >> Aborted after ~{elapsed / 60:.0f} min; "
              f"~{saved / 60:.0f} min of machine time reclaimed "
              f"(burn-in + EMC chamber skipped).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show the predictor library's internal logs alongside the report",
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)-7s %(name)s: %(message)s")
    # The report below already narrates aborts, so without -v the library is
    # kept quiet rather than echoing each abort warning a second time.
    logging.getLogger("predictor").setLevel(
        logging.DEBUG if args.verbose else logging.ERROR
    )

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(42)

    section("PART 1  Shadow mode: recording 120 historical validation runs")
    history = OUT / "history.jsonl"
    counts = generate_history(rng, 120, history)
    print(f"  Suite: {len(SUITE)} tests, {len(SCHEMA)} metrics, "
          f"{TOTAL_SECS / 60:.0f} min of machine time per board")
    print(f"  Recorded -> {history}")
    print(f"  Outcomes: {counts['healthy']} passed, "
          f"{counts['vrm_degraded']} failed (VRM degradation), "
          f"{counts['supply_noise']} failed (supply noise)")

    section("PART 2  Training: temporal split, two candidates, promotion gate")
    dataset = build_dataset([history], SCHEMA.schema_hash)
    print(f"  Dataset: {dataset.X.shape[0]} snapshots from {dataset.n_runs} runs "
          f"(every partial state is a training row)")

    candidates = {}
    for cls in (RandomForestPredictor, GradientBoostingPredictor):
        result = train_candidate(
            dataset, cls, threshold=ABORT_THRESHOLD, holdout_fraction=0.2
        )
        candidates[cls.kind] = result
        print(f"  {cls.__name__:<28} holdout: {fmt_metrics(result.metrics)}")

    rf, gb = candidates["random_forest"], candidates["hist_gradient_boosting"]
    winner = gb if should_promote(gb.metrics, rf.metrics) else rf
    print(f"  Promotion gate picks: {type(winner.predictor).__name__} "
          f"(false-abort cap, then time saved, Brier as tiebreaker)")

    registry = ModelRegistry(OUT / "registry")
    artifact = registry.save(winner.predictor, winner.metrics)
    print(f"  Registered -> {artifact.name} (+ metrics sidecar)")

    section("PART 3  Live bench: promoted model + hard safety rules")
    model_path = registry.latest(SCHEMA.schema_hash)
    print(f"  Engine: threshold={ABORT_THRESHOLD:.0%}, min_coverage=10%, "
          f"rules on vbus_v and peak_temp_c")
    run_bench_session("healthy board", "healthy", model_path, rng)
    run_bench_session("degrading VRM (ML fail-fast)", "vrm_degraded", model_path, rng)

    # Catastrophic short circuit: vbus collapses on the very first reading.
    print(f"\n--- Session: short circuit (rule override) " + "-" * 18)
    engine = ValidationEngine(
        SCHEMA, load_predictor(str(model_path)),
        threshold=ABORT_THRESHOLD, rules=SAFETY_RULES,
        recorder=JsonlRecorder(OUT / "live_records.jsonl"),
    )
    try:
        print(f"  {'test_power_on':<22} {'vbus_v':<18} {0.42:9.2f}   (!!)")
        engine.ingest("test_power_on", "vbus_v", 0.42)
    except OrchestrationAbortError as exc:
        engine.finalize(suite_passed=False)
        print(f"  !! RULE ABORT (ML bypassed): {exc}")
        print(f"  >> Aborted in milliseconds -- the model was never consulted.")

    section("PART 4  Drift check: simulated hardware revision (rev B)")
    # Rev B runs its VRM ~6 degC hotter while staying within spec. Compare
    # per-board readings (one fully-observed snapshot per run) with the same
    # population mix as history, so only the genuine shift stands out.
    baseline_rows = dataset.X[~np.isnan(dataset.X).any(axis=1)]
    rev_b = np.random.default_rng(7)
    rows = []
    for _ in range(240):
        mode = rev_b.choice(
            ["healthy", "vrm_degraded", "supply_noise"], p=[0.72, 0.14, 0.14]
        )
        state = StateVector(SCHEMA)
        for test_id, metric, value in simulate_run(rev_b, mode):
            if metric in ("temp_c", "temp_rise_c", "peak_temp_c"):
                value += 6.0
            state.update(test_id, metric, value)
        rows.append(state.snapshot())
    report = drift_report(baseline_rows, np.asarray(rows))
    flagged = [SCHEMA.features[i] for i in report["flagged_features"]]
    print(f"  max PSI = {report['max_psi']:.2f}  "
          f"max missing-rate shift = {report['max_missing_shift']:.2f}")
    print(f"  Flagged features: {', '.join(f'{t}.{m}' for t, m in flagged)}")
    print(f"  Retrain recommended: {report['retrain_recommended']}")
    if report["retrain_recommended"]:
        print("  -> the scheduled pipeline (PART 2) reruns on fresh records; "
              "the promotion gate decides if rev-B data produces a better model.")

    print(f"\nAll artifacts under: {OUT}")


if __name__ == "__main__":
    main()
