"""
Unit tests for :mod:`aura.retry`.

The suite verifies successful first attempts, retry-until-success behavior,
exhaustion and final exception propagation, logical failure accounting in
CircuitBreaker, open-circuit fail-fast behavior, validation, and propagation
of process-control ``BaseException`` subclasses without retrying them.

Run:
    From the repository root::

        python tests/unit/retry.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.binding import Binder, OneWayBinder
from aura.breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)
from aura.retry import RetryPolicy


class FlakyOperation(OneWayBinder):
    """Fail a configured number of physical attempts before succeeding."""

    def __init__(
        self,
        bindee: Binder | None = None,
        failures_before_success: int = 0,
    ) -> None:
        """Initialize the failure schedule and physical call counter."""

        super().__init__(bindee)

        self.failures_before_success = failures_before_success
        self.calls = 0

    def fire(self) -> None:
        """Execute one physical attempt."""

        self.calls += 1

        if self.calls <= self.failures_before_success:
            raise RuntimeError(f'failure {self.calls}')


class AlwaysFailsOperation(OneWayBinder):
    """Raise a distinct final error on every physical attempt."""

    def __init__(
        self,
        bindee: Binder | None = None,
    ) -> None:
        """Initialize the physical call counter."""

        super().__init__(bindee)
        self.calls = 0

    def fire(self) -> None:
        """Raise a failure containing the current attempt number."""

        self.calls += 1
        raise ValueError(f'attempt {self.calls}')


class InterruptingOperation(OneWayBinder):
    """Raise ``KeyboardInterrupt`` to verify it is never retried."""

    def __init__(
        self,
        bindee: Binder | None = None,
    ) -> None:
        """Initialize the physical call counter."""

        super().__init__(bindee)
        self.calls = 0

    def fire(self) -> None:
        """Raise a process-control exception."""

        self.calls += 1
        raise KeyboardInterrupt()


class RetryPolicyTests(unittest.TestCase):
    """Verify transparent retry and breaker integration."""

    def test_constructor_validates_retry_count(self) -> None:
        """Reject a negative number of retries."""

        operation = FlakyOperation.create()

        with self.assertRaises(ValueError):
            RetryPolicy.create(operation, -1)

    def test_successful_first_attempt_is_not_retried(self) -> None:
        """Return after one physical execution when no failure occurs."""

        operation = FlakyOperation.create(None, 0)
        policy = RetryPolicy.create(operation, 3)

        operation()

        self.assertEqual(operation.calls, 1)
        self.assertEqual(policy.retry_attempts, 3)
        self.assertEqual(policy.max_attempts, 4)
        self.assertIs(
            policy.state,
            CircuitBreakerState.CLOSED,
        )

    def test_operation_retries_until_success(self) -> None:
        """Recover when a later permitted physical attempt succeeds."""

        operation = FlakyOperation.create(None, 2)
        policy = RetryPolicy.create(operation, 2)

        operation()

        self.assertEqual(operation.calls, 3)
        self.assertEqual(policy.breaker.failure_count, 0)
        self.assertIs(
            policy.state,
            CircuitBreakerState.CLOSED,
        )

    def test_exhaustion_re_raises_final_exception_once_logically(self) -> None:
        """Preserve the final failure and count one logical breaker failure."""

        operation = AlwaysFailsOperation.create()
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout_seconds=30.0,
        )
        RetryPolicy.create(
            operation,
            2,
            breaker,
        )

        with self.assertRaisesRegex(
            ValueError,
            'attempt 3',
        ):
            operation()

        self.assertEqual(operation.calls, 3)
        self.assertEqual(breaker.failure_count, 1)

    def test_open_breaker_rejects_without_another_attempt(self) -> None:
        """Fail fast after one exhausted logical call opens the breaker."""

        operation = AlwaysFailsOperation.create()
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_seconds=30.0,
        )
        policy = RetryPolicy.create(
            operation,
            1,
            breaker,
        )

        with self.assertRaises(ValueError):
            operation()

        self.assertEqual(operation.calls, 2)
        self.assertIs(
            policy.state,
            CircuitBreakerState.OPEN,
        )

        with self.assertRaises(CircuitBreakerOpenError):
            operation()

        self.assertEqual(operation.calls, 2)

    def test_keyboard_interrupt_is_not_retried_or_counted_as_failure(self) -> None:
        """Propagate ``KeyboardInterrupt`` immediately without breaker failure."""

        operation = InterruptingOperation.create()
        breaker = CircuitBreaker(failure_threshold=1)

        RetryPolicy.create(
            operation,
            5,
            breaker,
        )

        with self.assertRaises(KeyboardInterrupt):
            operation()

        self.assertEqual(operation.calls, 1)
        self.assertEqual(breaker.failure_count, 0)
        self.assertIs(
            breaker.state,
            CircuitBreakerState.CLOSED,
        )


if __name__ == '__main__':
    unittest.main()
