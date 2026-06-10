"""End-to-end demo: ML-driven fail-fast validation of a 12V power board.

Scenario
--------
A bench validates power boards with an 8-test suite (~38 minutes of machine
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
                          saves ~36 min), and a short circuit (rule abort).
  PART 4  Drift check   - a simulated hardware revision shifts thermals; the
                          drift monitor flags the affected features.

Run it:    python examples/end_to_end.py
Verbose:   python examples/end_to_end.py -v   (adds the library's own logs)

Logging layout: the report you see on stdout is emitted through a dedicated
message-only logger (no prints); the predictor library logs independently to
stderr and stays quiet unless -v raises its level. This mirrors how a real
application should consume the library.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

# Make the demo runnable from a source checkout without installing.
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
    should_promote,
    train_candidate,
)

#: The demo's report channel. Configured in _configure_logging() with a
#: bare "%(message)s" format so the output reads as a report, not a log dump.
LOG = logging.getLogger("end_to_end")

OUT = Path(__file__).resolve().parent / "demo_output"
ABORT_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# The validation suite: (test_id, [metrics], duration in seconds).
# Cheap electrical checks run first; burn-in and the EMC chamber dominate
# cost — which is exactly why aborting before them is worth real money.
# ---------------------------------------------------------------------------
SUITE = [
    ("test_power_on",        ["vbus_v", "inrush_a"],                 8),
    ("test_3v3_rail",        ["vout_v", "ripple_mv"],               12),
    ("test_5v_rail",         ["vout_v", "ripple_mv"],               12),
    ("test_vrm_thermal",     ["temp_c", "temp_rise_c"],             45),
    ("test_load_step",       ["sag_mv", "recovery_us"],             30),
    ("test_clock_integrity", ["jitter_ps", "freq_error_ppm"],       60),
    ("test_burn_in",         ["peak_temp_c", "current_drift_pct"], 900),
    ("test_emc_scan",        ["worst_margin_db"],                 1200),
]

# The fixed feature layout: one slot per (test_id, metric), in suite order.
# Models are trained against this exact layout; its hash gates deployment.
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
# early fingerprint long before the test that finally fails — that gap
# between "first symptom" and "actual failure" is what the model exploits.
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

MODE_NAMES = ["healthy", "vrm_degraded", "supply_noise"]
MODE_PROBS = [0.72, 0.14, 0.14]


def simulate_run(rng: np.random.Generator, mode: str | None):
    """Yield (test_id, metric, value) readings for one board, in suite order."""
    offsets = FAILURE_MODES.get(mode, {})
    for test_id, metrics, _ in SUITE:
        for metric in metrics:
            mean, sigma = NOMINAL[(test_id, metric)]
            value = rng.normal(mean + offsets.get((test_id, metric), 0.0), sigma)
            yield test_id, metric, value


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------
def section(title: str) -> None:
    LOG.info("")
    LOG.info("=" * 72)
    LOG.info("  %s", title)
    LOG.info("=" * 72)


def prob_bar(p: float | None, width: int = 24) -> str:
    """Render a probability as an ASCII bar; None means the warm-up gate."""
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
    """Replay n_runs boards through a recorder only — no model, no aborts.

    This is the bootstrap phase of a real deployment: the engine ships
    recording-only until enough labeled history exists to train on.
    """
    recorder = JsonlRecorder(path)
    counts = dict.fromkeys(MODE_NAMES, 0)
    for _ in range(n_runs):
        mode = rng.choice(MODE_NAMES, p=MODE_PROBS)
        counts[mode] += 1
        # Mirror exactly what ValidationEngine would record: the progressive
        # state vector after every single measurement.
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
    # Catastrophic events bypass the ML model entirely (DESIGN.md §3.3):
    # these bounds protect the hardware, the model only protects the schedule.
    BoundsRule("vbus_v", min_value=10.5, max_value=13.5),
    BoundsRule("peak_temp_c", max_value=110.0),
]


def run_bench_session(name: str, mode: str | None, model_path: Path,
                      rng: np.random.Generator) -> None:
    """Execute one board's suite through a fresh engine, narrating each step."""
    LOG.info("")
    LOG.info("--- Session: %s %s", name, "-" * max(0, 50 - len(name)))
    # One engine per board: the state vector represents a single run.
    engine = ValidationEngine(
        SCHEMA,
        load_predictor(str(model_path)),
        threshold=ABORT_THRESHOLD,
        rules=SAFETY_RULES,
        min_coverage=0.10,  # no predictions until 10% of the vector is real
        recorder=JsonlRecorder(OUT / "live_records.jsonl"),
    )
    elapsed = 0.0
    current_test = None
    try:
        for test_id, metric, value in simulate_run(rng, mode):
            if test_id != current_test:
                # Account machine time once per test, when it starts.
                current_test = test_id
                elapsed += DURATION[test_id]
            line = f"  {test_id:<22} {metric:<18} {value:9.2f}"
            try:
                p = engine.ingest(test_id, metric, value)
            except OrchestrationAbortError:
                # Show the offending reading before re-raising: ingest()
                # raises before this line would otherwise be reported.
                LOG.info("%s   << abort triggered by this reading", line)
                raise
            LOG.info("%s   %s", line, prob_bar(p))
        engine.finalize(suite_passed=True)
        LOG.info("  >> SUITE PASSED in %.0f min of machine time.", TOTAL_SECS / 60)
    except RuleAbort as exc:
        # Deterministic safety abort: the model was never consulted.
        engine.finalize(suite_passed=False)
        LOG.info("  !! RULE ABORT (ML bypassed): %s", exc)
        LOG.info("  >> Hardware protected after ~%.0f min; "
                 "runner tears down the bench safely.", elapsed / 60)
    except MLThresholdAbort as exc:
        # Probabilistic abort: the suite is statistically doomed, stop paying
        # for burn-in and the EMC chamber.
        engine.finalize(suite_passed=False)
        LOG.info("  !! ML ABORT: P(failure)=%.1f%% >= threshold %.0f%%",
                 exc.probability * 100, exc.threshold * 100)
        LOG.info("  >> Aborted after ~%.0f min; ~%.0f min of machine time "
                 "reclaimed (burn-in + EMC chamber skipped).",
                 elapsed / 60, (TOTAL_SECS - elapsed) / 60)


def _configure_logging(verbose: bool) -> None:
    """Set up the two output streams.

    - The demo report: bare messages on stdout (this script's UI).
    - The predictor library: level-prefixed records on stderr, visible only
      with -v. The library itself never prints — it just logs and the
      application decides what to surface.
    """
    report = logging.StreamHandler(sys.stdout)
    report.setFormatter(logging.Formatter("%(message)s"))
    LOG.addHandler(report)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False  # keep report lines away from other handlers

    library = logging.StreamHandler(sys.stderr)
    library.setFormatter(
        logging.Formatter("%(levelname)-7s %(name)s: %(message)s")
    )
    lib_logger = logging.getLogger("predictor")
    lib_logger.addHandler(library)
    # The report already narrates aborts; without -v the library stays quiet
    # rather than echoing each abort warning a second time.
    lib_logger.setLevel(logging.DEBUG if verbose else logging.ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show the predictor library's internal logs alongside the report",
    )
    _configure_logging(parser.parse_args().verbose)

    # Fresh artifacts every run: the demo is fully reproducible (seeded RNGs).
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(42)

    # -- PART 1 -------------------------------------------------------------
    section("PART 1  Shadow mode: recording 120 historical validation runs")
    history = OUT / "history.jsonl"
    counts = generate_history(rng, 120, history)
    LOG.info("  Suite: %d tests, %d metrics, %.0f min of machine time per board",
             len(SUITE), len(SCHEMA), TOTAL_SECS / 60)
    LOG.info("  Recorded -> %s", history)
    LOG.info("  Outcomes: %d passed, %d failed (VRM degradation), "
             "%d failed (supply noise)",
             counts["healthy"], counts["vrm_degraded"], counts["supply_noise"])

    # -- PART 2 -------------------------------------------------------------
    section("PART 2  Training: temporal split, two candidates, promotion gate")
    dataset = build_dataset([history], SCHEMA.schema_hash)
    LOG.info("  Dataset: %d snapshots from %d runs "
             "(every partial state is a training row)",
             dataset.X.shape[0], dataset.n_runs)

    # Train both shipped model families on identical data and let the
    # promotion gate decide — nothing in the pipeline favors either.
    candidates = {}
    for cls in (RandomForestPredictor, GradientBoostingPredictor):
        result = train_candidate(
            dataset, cls, threshold=ABORT_THRESHOLD, holdout_fraction=0.2
        )
        candidates[cls.kind] = result
        LOG.info("  %-28s holdout: %s", cls.__name__, fmt_metrics(result.metrics))

    rf, gb = candidates["random_forest"], candidates["hist_gradient_boosting"]
    winner = gb if should_promote(gb.metrics, rf.metrics) else rf
    LOG.info("  Promotion gate picks: %s "
             "(false-abort cap, then time saved, Brier as tiebreaker)",
             type(winner.predictor).__name__)

    registry = ModelRegistry(OUT / "registry")
    artifact = registry.save(winner.predictor, winner.metrics)
    LOG.info("  Registered -> %s (+ metrics sidecar)", artifact.name)

    # -- PART 3 -------------------------------------------------------------
    section("PART 3  Live bench: promoted model + hard safety rules")
    model_path = registry.latest(SCHEMA.schema_hash)
    LOG.info("  Engine: threshold=%.0f%%, min_coverage=10%%, "
             "rules on vbus_v and peak_temp_c", ABORT_THRESHOLD * 100)
    run_bench_session("healthy board", "healthy", model_path, rng)
    run_bench_session("degrading VRM (ML fail-fast)", "vrm_degraded", model_path, rng)

    # Catastrophic short circuit: vbus collapses on the very first reading.
    # Fed manually (not via simulate_run) because no distribution produces it.
    LOG.info("")
    LOG.info("--- Session: short circuit (rule override) %s", "-" * 18)
    engine = ValidationEngine(
        SCHEMA, load_predictor(str(model_path)),
        threshold=ABORT_THRESHOLD, rules=SAFETY_RULES,
        recorder=JsonlRecorder(OUT / "live_records.jsonl"),
    )
    try:
        LOG.info("  %-22s %-18s %9.2f   (!!)", "test_power_on", "vbus_v", 0.42)
        engine.ingest("test_power_on", "vbus_v", 0.42)
    except OrchestrationAbortError as exc:
        engine.finalize(suite_passed=False)
        LOG.info("  !! RULE ABORT (ML bypassed): %s", exc)
        LOG.info("  >> Aborted in milliseconds -- the model was never consulted.")

    # -- PART 4 -------------------------------------------------------------
    section("PART 4  Drift check: simulated hardware revision (rev B)")
    # Rev B runs its VRM ~6 degC hotter while staying within spec. Compare
    # per-board readings (one fully-observed snapshot per run) with the same
    # population mix as history, so only the genuine shift stands out.
    baseline_rows = dataset.X[~np.isnan(dataset.X).any(axis=1)]
    rev_b = np.random.default_rng(7)
    rows = []
    for _ in range(240):
        mode = rev_b.choice(MODE_NAMES, p=MODE_PROBS)
        state = StateVector(SCHEMA)
        for test_id, metric, value in simulate_run(rev_b, mode):
            if metric in ("temp_c", "temp_rise_c", "peak_temp_c"):
                value += 6.0  # the revision's thermal signature
            state.update(test_id, metric, value)
        rows.append(state.snapshot())  # final snapshot = the board's readings
    report = drift_report(baseline_rows, np.asarray(rows))
    flagged = [SCHEMA.features[i] for i in report["flagged_features"]]
    LOG.info("  max PSI = %.2f  max missing-rate shift = %.2f",
             report["max_psi"], report["max_missing_shift"])
    LOG.info("  Flagged features: %s",
             ", ".join(f"{t}.{m}" for t, m in flagged))
    LOG.info("  Retrain recommended: %s", report["retrain_recommended"])
    if report["retrain_recommended"]:
        LOG.info("  -> the scheduled pipeline (PART 2) reruns on fresh records; "
                 "the promotion gate decides if rev-B data produces a better model.")

    LOG.info("")
    LOG.info("All artifacts under: %s", OUT)


if __name__ == "__main__":
    main()
