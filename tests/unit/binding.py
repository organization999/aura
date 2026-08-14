"""
Unit tests for :mod:`aura.binding`.

The suite verifies factory-only construction, one-way propagation,
reciprocal graph construction, cycle termination, duplicate-edge
idempotence, and self-binding rejection.

Run:
    From the repository root::

        python tests/unit/binding.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.binding import Binder, OneWayBinder, TwoWayBinder


class RecordingBinder(OneWayBinder):
    """Record every local firing in a shared list."""

    def __init__(
        self,
        bindee: Binder | None = None,
        name: str = '',
        events: list[str] | None = None,
    ) -> None:
        """Initialize a recording binder used by the unit tests."""

        super().__init__(bindee)

        self.name = name
        self.events = [] if events is None else events

    def fire(self) -> None:
        """Record one local firing."""

        self.events.append(self.name)


class ReciprocalRecordingBinder(TwoWayBinder):
    """Record firings for a reciprocal endpoint."""

    def __init__(
        self,
        bindee: Binder | None = None,
        name: str = '',
        events: list[str] | None = None,
    ) -> None:
        """Initialize a reciprocal recording endpoint."""

        super().__init__(bindee)

        self.name = name
        self.events = [] if events is None else events

    def fire(self) -> None:
        """Record one local firing."""

        self.events.append(self.name)


class BindingTests(unittest.TestCase):
    """Verify Binder graph construction and invocation semantics."""

    def test_direct_construction_is_rejected(self) -> None:
        """Require concrete one-way binders to use their factory."""

        with self.assertRaises(TypeError):
            RecordingBinder()

    def test_one_way_binding_propagates_only_forward(self) -> None:
        """Fire an upstream endpoint followed by its downstream endpoint."""

        events: list[str] = []

        downstream = RecordingBinder.create(
            None,
            'downstream',
            events,
        )
        upstream = RecordingBinder.create(
            downstream,
            'upstream',
            events,
        )

        upstream()
        self.assertEqual(
            events,
            ['upstream', 'downstream'],
        )

        events.clear()
        downstream()

        self.assertEqual(
            events,
            ['downstream'],
        )

    def test_two_way_factory_reuses_existing_endpoint(self) -> None:
        """Create one new reciprocal endpoint without cloning its bindee."""

        events: list[str] = []

        left = RecordingBinder.create(
            None,
            'left',
            events,
        )
        right = ReciprocalRecordingBinder.create(
            left,
            'right',
            events,
        )

        self.assertIn(right, left._bindees)
        self.assertIn(left, right._bindees)

        left()
        self.assertEqual(
            events,
            ['left', 'right'],
        )

        events.clear()
        right()
        self.assertEqual(
            events,
            ['right', 'left'],
        )

    def test_longer_cycle_fires_each_endpoint_once(self) -> None:
        """Terminate a three-node cycle with invocation-local visited state."""

        events: list[str] = []

        first = RecordingBinder.create(None, 'first', events)
        second = RecordingBinder.create(None, 'second', events)
        third = RecordingBinder.create(None, 'third', events)

        first._bind(second)
        second._bind(third)
        third._bind(first)

        first()

        self.assertEqual(
            events,
            ['first', 'second', 'third'],
        )

    def test_duplicate_binding_is_idempotent(self) -> None:
        """Avoid duplicate graph edges when the same bindee is added twice."""

        first = RecordingBinder.create()
        second = RecordingBinder.create()

        first._bind(second)
        first._bind(second)

        self.assertEqual(
            first._bindees,
            (second,),
        )

    def test_self_binding_is_rejected(self) -> None:
        """Reject a directed edge from one binder back to itself."""

        binder = RecordingBinder.create()

        with self.assertRaisesRegex(
            ValueError,
            'cannot bind to itself',
        ):
            binder._bind(binder)


if __name__ == '__main__':
    unittest.main()
