"""
tests/unit/test_neo4j_retry_sync.py — Unit tests for
scinr.newton.utils.neo4j_retry.with_neo4j_retry_sync()
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from neo4j.exceptions import ClientError, Neo4jError, ServiceUnavailable, SessionExpired

from scinr.newton.utils.neo4j_retry import with_neo4j_retry_sync

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FlakyCallable:
    """A zero-arg callable that fails N times with a transient error, then succeeds."""

    def __init__(self, fail_times: int, exc_factory=None, result: str = "ok"):
        self.fail_times = fail_times
        self.calls = 0
        self.result = result
        self.exc_factory = exc_factory or (
            lambda: ServiceUnavailable("Failed to write to defunct connection")
        )

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc_factory()
        return self.result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retries_transient_error_then_succeeds():
    """A transient ServiceUnavailable error should be retried until success."""
    flaky = _FlakyCallable(fail_times=3)

    with patch("scinr.newton.utils.neo4j_retry.time.sleep") as mock_sleep:
        result = with_neo4j_retry_sync(flaky)

    assert result == "ok"
    assert flaky.calls == 4  # 3 failures + 1 success
    assert mock_sleep.call_count == 3


def test_exhausts_all_retries_then_raises():
    """If every attempt fails with a transient error, the last exception is re-raised."""
    flaky = _FlakyCallable(fail_times=999)  # always fails

    with patch("scinr.newton.utils.neo4j_retry.time.sleep"):
        with pytest.raises(ServiceUnavailable):
            with_neo4j_retry_sync(flaky, exp_retries=2, plateau_retries=1)

    # total_retries = 2 + 1 = 3 -> 4 total attempts (initial + 3 retries)
    assert flaky.calls == 4


def test_non_transient_error_raised_immediately_without_retry():
    """A non-transient exception (e.g. ValueError) must propagate immediately, no retries."""
    flaky = _FlakyCallable(fail_times=999, exc_factory=lambda: ValueError("boom"))

    with patch("scinr.newton.utils.neo4j_retry.time.sleep") as mock_sleep:
        with pytest.raises(ValueError, match="boom"):
            with_neo4j_retry_sync(flaky)

    assert flaky.calls == 1  # no retries at all
    mock_sleep.assert_not_called()


def test_returns_value_on_first_try_without_sleeping():
    """A callable that succeeds immediately should never sleep."""
    flaky = _FlakyCallable(fail_times=0)

    with patch("scinr.newton.utils.neo4j_retry.time.sleep") as mock_sleep:
        result = with_neo4j_retry_sync(flaky)

    assert result == "ok"
    assert flaky.calls == 1
    mock_sleep.assert_not_called()



def _server_error(code: str) -> Neo4jError:
    """The exception the driver raises for a server failure with *code*."""
    return Neo4jError._hydrate_neo4j(code=code, message="boom")


@pytest.mark.parametrize(
    "exc_factory",
    [
        # The session lost its server (or its leader) while it was open.
        lambda: SessionExpired("the server is no longer available"),
        # A write routed to a member that just stopped being the leader: the
        # code is a ClientError, the driver declares it retryable.
        lambda: _server_error("Neo.ClientError.Cluster.NotALeader"),
        lambda: _server_error("Neo.ClientError.General.ForbiddenOnReadOnlyDatabase"),
        # The transaction memory pool is shared by every transaction of the
        # DBMS: a batch that does not fit now can fit a moment later.
        lambda: _server_error("Neo.TransientError.General.MemoryPoolOutOfMemoryError"),
        lambda: _server_error("Neo.TransientError.Transaction.DeadlockDetected"),
    ],
    ids=["session-expired", "not-a-leader", "read-only-member", "memory-pool", "deadlock"],
)
def test_cluster_and_memory_errors_are_retried(exc_factory):
    flaky = _FlakyCallable(fail_times=2, exc_factory=exc_factory)

    with patch("scinr.newton.utils.neo4j_retry.time.sleep") as mock_sleep:
        result = with_neo4j_retry_sync(flaky)

    assert result == "ok"
    assert flaky.calls == 3
    assert mock_sleep.call_count == 2


@pytest.mark.parametrize(
    "code",
    ["Neo.ClientError.Statement.SyntaxError", "Neo.ClientError.Schema.ConstraintValidationFailed"],
)
def test_client_errors_are_not_retried(code):
    flaky = _FlakyCallable(fail_times=999, exc_factory=lambda: _server_error(code))

    with patch("scinr.newton.utils.neo4j_retry.time.sleep") as mock_sleep:
        with pytest.raises(ClientError):
            with_neo4j_retry_sync(flaky)

    assert flaky.calls == 1
    mock_sleep.assert_not_called()
