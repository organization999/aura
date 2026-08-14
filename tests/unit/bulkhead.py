"""
Unit tests for :mod:`aura.bulkhead`.

The suite verifies configuration validation, the public concurrency limit,
fail-fast rejection when capacity is exhausted, exception propagation, and
permit release after a managed operation fails.

Run:
    From the repository root::

        python tests/unit/bulkhead.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from threading import Event, Thread
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.binding import Binder, OneWayBinder
from aura.bulkhead import Bulkhead, BulkheadFullError


class BlockingOperation(OneWayBinder):
    """Hold a bulkhead permit until explicitly released."""

    def __init__(
        self,
        bindee: Binder | None = None,
        entered: Event | None = None,
        release: Event | None = None,
    ) -> None:
        """Initialize synchronization state used by concurrency tests."""

        super().__init__(bindee)

        self.entered = Event() if entered is None else entered
        self.release = Event() if release is None else release

    def fire(self) -> None:
        """Block until the test releases this invocation."""

        self.entered.set()

        if not self.release.wait(timeout=2.0):
            raise TimeoutError('test did not release blocking operation')


class FailOnceOperation(OneWayBinder):
    """Fail the first invocation and succeed on the second."""

    def __init__(
        self,
        bindee: Binder | None = None,
    ) -> None:
        """Initialize the physical invocation counter."""

        super().__init__(bindee)
        self.calls = 0

    def fire(self) -> None:
        """Raise once so the test can verify permit release in ``finally``."""

        self.calls += 1

        if self.calls == 1:
            raise RuntimeError('first call fails')


class BulkheadTests(unittest.TestCase):
    """Verify standalone concurrency isolation."""

    def test_constructor_validates_capacity(self) -> None:
        """Reject a non-positive maximum concurrency value."""

        operation = FailOnceOperation.create()

        with self.assertRaises(ValueError):
            Bulkhead.create(operation, 0)

    def test_max_concurrent_calls_property(self) -> None:
        """Expose the configured concurrency limit."""

        operation = FailOnceOperation.create()
        bulkhead = Bulkhead.create(operation, 3)

        self.assertEqual(
            bulkhead.max_concurrent_calls,
            3,
        )

    def test_second_concurrent_call_is_rejected(self) -> None:
        """Fail fast while the only configured permit is occupied."""

        entered = Event()
        release = Event()

        operation = BlockingOperation.create(
            None,
            entered,
            release,
        )
        Bulkhead.create(operation, 1)

        errors: list[BaseException] = []

        def worker() -> None:
            """Run the first protected invocation in another thread."""

            try:
                operation()
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=worker)
        thread.start()

        self.assertTrue(
            entered.wait(timeout=1.0),
            'first invocation never entered the managed operation',
        )

        try:
            with self.assertRaises(BulkheadFullError):
                operation()
        finally:
            release.set()
            thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_capacity_is_released_after_exception(self) -> None:
        """Allow a later invocation after the first admitted call raises."""

        operation = FailOnceOperation.create()
        Bulkhead.create(operation, 1)

        with self.assertRaisesRegex(
            RuntimeError,
            'first call fails',
        ):
            operation()

        operation()

        self.assertEqual(operation.calls, 2)


if __name__ == '__main__':
    unittest.main()
