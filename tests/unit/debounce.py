"""
Unit tests for :mod:`aura.debounce`.

The tests replace :class:`threading.Timer` with a deterministic manual timer.
This keeps the suite fast and avoids scheduler-dependent sleeps while still
validating the Debouncer generation, reset, cancel, flush, and expiration
behavior.

Run:
    From the repository root::

        python tests/unit/debounce.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, ClassVar
import unittest
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.binding import Binder, OneWayBinder
from aura.debounce import Debouncer


class ManualTimer:
    """Provide a controllable substitute for :class:`threading.Timer`."""

    instances: ClassVar[list['ManualTimer']] = []

    def __init__(
        self,
        interval: float,
        function: Callable[..., None],
        args: tuple[object, ...] | None = None,
        kwargs: dict[str, object] | None = None,
    ) -> None:
        """Capture one scheduled callback without creating a thread.

        Args:
            interval (float):
                Requested timer interval in seconds.

            function (Callable[..., None]):
                Callback that should execute when :meth:`fire` is invoked.

            args (tuple[object, ...] | None, optional):
                Positional callback arguments.

            kwargs (dict[str, object] | None, optional):
                Keyword callback arguments.

        Returns:
            None:
                The manual timer is registered in :attr:`instances`.
        """

        self.interval = interval
        self.function = function
        self.args = () if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.daemon = False
        self.started = False
        self.cancelled = False

        type(self).instances.append(self)

    def start(self) -> None:
        """Mark the manual timer as started."""

        self.started = True

    def cancel(self) -> None:
        """Mark the manual timer as cancelled."""

        self.cancelled = True

    def fire(self) -> None:
        """Execute the captured callback unless cancellation occurred."""

        if self.cancelled:
            return

        self.function(
            *self.args,
            **self.kwargs,
        )


class CountingOperation(OneWayBinder):
    """Count how many deferred managed operations actually execute."""

    def __init__(
        self,
        bindee: Binder | None = None,
    ) -> None:
        """Initialize the execution counter."""

        super().__init__(bindee)
        self.calls = 0

    def fire(self) -> None:
        """Increment the execution counter."""

        self.calls += 1


class DebouncerTests(unittest.TestCase):
    """Verify quiet-period interception without real asynchronous timers."""

    def setUp(self) -> None:
        """Clear timers created by a previous test."""

        ManualTimer.instances.clear()

    def test_constructor_validates_duration(self) -> None:
        """Reject a negative debounce duration."""

        operation = CountingOperation.create()

        with self.assertRaises(ValueError):
            Debouncer.create(operation, -0.01)

    def test_attachment_does_not_arm_timer(self) -> None:
        """Avoid scheduling work until the managed resource is accessed."""

        operation = CountingOperation.create()

        with patch('aura.debounce.Timer', ManualTimer):
            debouncer = Debouncer.create(operation, 5.0)

            self.assertFalse(debouncer.pending)
            self.assertIsNone(debouncer.since)
            self.assertEqual(ManualTimer.instances, [])

    def test_resource_access_is_intercepted_and_reset(self) -> None:
        """Cancel the previous generation and execute only the newest timer."""

        operation = CountingOperation.create()

        with (
            patch('aura.debounce.Timer', ManualTimer),
            patch('aura.debounce.monotonic', return_value=100.0),
        ):
            debouncer = Debouncer.create(operation, 5.0)

            operation()

            first = ManualTimer.instances[-1]

            self.assertTrue(first.started)
            self.assertTrue(first.daemon is False)
            self.assertTrue(debouncer.pending)
            self.assertEqual(operation.calls, 0)

            operation()

            second = ManualTimer.instances[-1]

            self.assertTrue(first.cancelled)
            self.assertIsNot(first, second)
            self.assertEqual(operation.calls, 0)

            first.fire()
            self.assertEqual(operation.calls, 0)

            second.fire()

            self.assertEqual(operation.calls, 1)
            self.assertFalse(debouncer.pending)

    def test_cancel_prevents_pending_execution(self) -> None:
        """Invalidate a pending generation without firing the managed binder."""

        operation = CountingOperation.create()

        with patch('aura.debounce.Timer', ManualTimer):
            debouncer = Debouncer.create(operation, 5.0)

            operation()
            timer = ManualTimer.instances[-1]

            debouncer.cancel()
            timer.fire()

            self.assertTrue(timer.cancelled)
            self.assertFalse(debouncer.pending)
            self.assertEqual(operation.calls, 0)

    def test_flush_executes_pending_operation_synchronously(self) -> None:
        """Cancel the timer and immediately resume the managed binder."""

        operation = CountingOperation.create()

        with patch('aura.debounce.Timer', ManualTimer):
            debouncer = Debouncer.create(operation, 5.0)

            operation()
            timer = ManualTimer.instances[-1]

            debouncer.flush()

            self.assertTrue(timer.cancelled)
            self.assertFalse(debouncer.pending)
            self.assertEqual(operation.calls, 1)

    def test_remaining_uses_monotonic_deadline(self) -> None:
        """Report the interval remaining relative to the monotonic clock."""

        operation = CountingOperation.create()

        clock = [100.0]

        with (
            patch('aura.debounce.Timer', ManualTimer),
            patch(
                'aura.debounce.monotonic',
                side_effect=lambda: clock[0],
            ),
        ):
            debouncer = Debouncer.create(operation, 5.0)
            operation()

            clock[0] = 102.0

            self.assertAlmostEqual(
                debouncer.remaining.total_seconds(),
                3.0,
            )

            debouncer.cancel()


if __name__ == '__main__':
    unittest.main()
