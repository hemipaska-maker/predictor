"""Thin pytest adapter for the validation engine (DESIGN.md §5.1).

All logic lives in the agnostic core; this plugin only translates between
pytest's lifecycle and the engine API:

- ``--hve-engine pkg.module:factory`` names a zero-arg callable returning a
  configured :class:`ValidationEngine`; the session-scoped
  ``validation_engine`` fixture builds it lazily.
- An ``OrchestrationAbortError`` escaping a test fails that test as usual,
  and the plugin sets ``session.shouldstop`` so no further tests start while
  pytest still runs every teardown/finalizer (hardware released safely).
- At session end the run is labeled via ``engine.finalize`` for training data.
"""

from __future__ import annotations

import importlib
import logging

import pytest

from predictor.core.engine import ValidationEngine
from predictor.core.exceptions import OrchestrationAbortError

logger = logging.getLogger(__name__)

_ENGINE_KEY: pytest.StashKey[ValidationEngine] = pytest.StashKey()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("validation-engine")
    group.addoption(
        "--hve-engine",
        default=None,
        metavar="MODULE:FACTORY",
        help=(
            "dotted path to a zero-arg callable returning a configured "
            "predictor.ValidationEngine, e.g. 'myproj.validation:make_engine'"
        ),
    )


def _load_factory(spec: str):
    module_name, sep, attr = spec.partition(":")
    if not sep or not attr:
        raise pytest.UsageError(
            f"--hve-engine expects 'module:factory', got {spec!r}"
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise pytest.UsageError(
            f"module {module_name!r} has no attribute {attr!r}"
        ) from None


@pytest.fixture(scope="session")
def validation_engine(request: pytest.FixtureRequest) -> ValidationEngine:
    """The configured engine; tests call ``.ingest(test_id, metric, value)``.

    Override this fixture in conftest.py to build the engine in code instead
    of via the command-line option.
    """
    spec = request.config.getoption("--hve-engine")
    if not spec:
        pytest.skip("validation engine not configured (--hve-engine)")
    engine = _load_factory(spec)()
    if not isinstance(engine, ValidationEngine):
        raise pytest.UsageError(
            f"--hve-engine factory returned {type(engine).__name__}, "
            "expected ValidationEngine"
        )
    request.config.stash[_ENGINE_KEY] = engine
    return engine


@pytest.fixture
def ingest(validation_engine: ValidationEngine, request: pytest.FixtureRequest):
    """Feed a measurement for the current test: ``ingest("vout", 3.31)``.

    A convenience over ``validation_engine.ingest`` that fills ``test_id``
    with the test function's name (parametrization suffix stripped), so
    schema entries line up with test names for free. Returns the measured
    value unchanged, allowing inline use in assertions::

        def test_3v3_rail(ingest, dmm):
            assert ingest("vout", dmm.measure_voltage("3V3")) > 3.2
    """
    test_id = getattr(request.node, "originalname", None) or request.node.name

    def _ingest(metric: str, value: float) -> float:
        validation_engine.ingest(test_id, metric, value)
        return value

    return _ingest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    yield
    if call.excinfo is not None and call.excinfo.errisinstance(
        OrchestrationAbortError
    ):
        # The test itself fails normally; stop scheduling further tests.
        # pytest still runs all teardowns, so hardware is released safely.
        item.session.shouldstop = (
            f"validation engine abort: {call.excinfo.value}"
        )
        logger.warning("stopping session: %s", item.session.shouldstop)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    engine = session.config.stash.get(_ENGINE_KEY, None)
    if engine is not None:
        engine.finalize(suite_passed=session.testsfailed == 0)
