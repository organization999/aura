"""
Unit tests for :mod:`aura.breaker`.

The suite verifies configuration validation, closed-state accounting,
transition to open after the failure threshold, fail-fast admission while
open, timed transition to half-closed, single-probe enforcement, successful
recovery, failed recovery, and remaining-open-time reporting.

Run:
    From the repository root::

        python tests/unit/breaker.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)


class CircuitBreakerTests(unittest.TestCase):
    """Verify deterministic CircuitBreaker state-machine behavior."""

    def test_constructor_validates_configuration(self) -> None:
        """Reject invalid failure thresholds and recovery timeouts."""

        with self.assertRaises(ValueError):
            CircuitBreaker(failure_threshold=0)

        with self.assertRaises(ValueError):
            CircuitBreaker(recovery_timeout_seconds=-0.01)

    def test_breaker_starts_closed_with_zero_failures(self) -> None:
        """Expose initial configuration and closed state."""

        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout_seconds=7.5,
        )

        self.assertIs(
            breaker.state,
            CircuitBreakerState.CLOSED,
        )
        self.assertEqual(breaker.failure_count, 0)
        self.assertEqual(breaker.failure_threshold, 3)
        self.assertEqual(breaker.recovery_timeout, 7.5)
        self.assertEqual(breaker.remaining_open_seconds, 0.0)

    def test_failure_threshold_opens_and_rejects_new_calls(self) -> None:
        """Open after consecutive logical failures and fail fast afterward."""

        clock = [100.0]

        with patch(
            'aura.breaker.monotonic',
            side_effect=lambda: clock[0],
        ):
            breaker = CircuitBreaker(
                failure_threshold=2,
                recovery_timeout_seconds=10.0,
            )

            first = breaker._acquire()
            breaker._record_failure(first)
            breaker._release(first)

            self.assertIs(
                breaker.state,
                CircuitBreakerState.CLOSED,
            )
            self.assertEqual(breaker.failure_count, 1)

            second = breaker._acquire()
            breaker._record_failure(second)
            breaker._release(second)

            self.assertIs(
                breaker.state,
                CircuitBreakerState.OPEN,
            )
            self.assertEqual(breaker.failure_count, 2)

            with self.assertRaises(CircuitBreakerOpenError):
                breaker._acquire()

    def test_success_resets_closed_failure_count(self) -> None:
        """Reset consecutive failures when a later logical call succeeds."""

        breaker = CircuitBreaker(failure_threshold=3)

        failed = breaker._acquire()
        breaker._record_failure(failed)
        breaker._release(failed)

        self.assertEqual(breaker.failure_count, 1)

        succeeded = breaker._acquire()
        breaker._record_success(succeeded)
        breaker._release(succeeded)

        self.assertEqual(breaker.failure_count, 0)
        self.assertIs(
            breaker.state,
            CircuitBreakerState.CLOSED,
        )

    def test_half_closed_allows_one_probe_and_success_closes(self) -> None:
        """Permit one recovery probe after timeout and close on success."""

        clock = [20.0]

        with patch(
            'aura.breaker.monotonic',
            side_effect=lambda: clock[0],
        ):
            breaker = CircuitBreaker(
                failure_threshold=1,
                recovery_timeout_seconds=5.0,
            )

            failed = breaker._acquire()
            breaker._record_failure(failed)
            breaker._release(failed)

            self.assertIs(
                breaker.state,
                CircuitBreakerState.OPEN,
            )

            clock[0] = 25.0

            self.assertIs(
                breaker.state,
                CircuitBreakerState.HALF_CLOSED,
            )

            probe = breaker._acquire()

            with self.assertRaises(CircuitBreakerOpenError):
                breaker._acquire()

            breaker._record_success(probe)
            breaker._release(probe)

            self.assertIs(
                breaker.state,
                CircuitBreakerState.CLOSED,
            )
            self.assertEqual(breaker.failure_count, 0)

    def test_failed_half_closed_probe_reopens_circuit(self) -> None:
        """Restart the open timeout when the recovery probe fails."""

        clock = [50.0]

        with patch(
            'aura.breaker.monotonic',
            side_effect=lambda: clock[0],
        ):
            breaker = CircuitBreaker(
                failure_threshold=1,
                recovery_timeout_seconds=4.0,
            )

            failed = breaker._acquire()
            breaker._record_failure(failed)
            breaker._release(failed)

            clock[0] = 54.0
            probe = breaker._acquire()

            breaker._record_failure(probe)
            breaker._release(probe)

            self.assertIs(
                breaker.state,
                CircuitBreakerState.OPEN,
            )

            clock[0] = 56.0
            self.assertAlmostEqual(
                breaker.remaining_open_seconds,
                2.0,
            )


if __name__ == '__main__':
    unittest.main()
