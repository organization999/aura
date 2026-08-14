"""
Demonstrate one-way and reciprocal Aura binder relationships.

This example exercises :class:`aura.binding.OneWayBinder` and
:class:`aura.binding.TwoWayBinder` using small recording binders.

The one-way example constructs::

    upstream -> downstream

Invoking ``upstream`` fires both objects in order. Invoking ``downstream``
fires only ``downstream`` because no reverse edge was installed.

The reciprocal example starts with one existing endpoint and creates one new
endpoint through :meth:`TwoWayBinder.create`. The resulting graph is::

    left <-> right

Invoking either endpoint reaches both endpoints exactly once because the
invocation-local visited set prevents reciprocal recursion.

Run:
    From the repository root::

        python examples/binding.py

Expected output:
    The script prints the firing order for one-way and reciprocal bindings.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.binding import Binder, OneWayBinder, TwoWayBinder


class RecordingOneWayBinder(OneWayBinder):
    """Record one-way binder executions in a shared event list."""

    def __init__(
        self,
        bindee: Binder | None = None,
        name: str = '',
        events: list[str] | None = None,
    ) -> None:
        """Initialize a named recording endpoint.

        Args:
            bindee (Binder | None, optional):
                Existing downstream binder. Defaults to ``None``.

            name (str, optional):
                Name appended to ``events`` when this binder fires.
                Defaults to an empty string.

            events (list[str] | None, optional):
                Shared list used to record firing order. A new list is
                created when ``None`` is supplied.

        Returns:
            None:
                The factory-allocated endpoint is initialized.
        """

        super().__init__(bindee)

        self.name = name
        self.events = [] if events is None else events

    def fire(self) -> None:
        """Append this endpoint's name to the shared event list.

        Returns:
            None:
                The firing event is recorded in place.
        """

        self.events.append(self.name)


class RecordingTwoWayBinder(TwoWayBinder):
    """Record executions for a reciprocal endpoint."""

    def __init__(
        self,
        bindee: Binder | None = None,
        name: str = '',
        events: list[str] | None = None,
    ) -> None:
        """Initialize a named reciprocal recording endpoint.

        Args:
            bindee (Binder | None, optional):
                Existing endpoint to which this new endpoint will be bound
                reciprocally by the factory.

            name (str, optional):
                Name appended when this endpoint fires.

            events (list[str] | None, optional):
                Shared firing-order list.

        Returns:
            None:
                The endpoint is initialized.
        """

        super().__init__(bindee)

        self.name = name
        self.events = [] if events is None else events

    def fire(self) -> None:
        """Append this endpoint's name to the shared event list.

        Returns:
            None:
                The firing event is recorded.
        """

        self.events.append(self.name)


def demonstrate_one_way() -> None:
    """Demonstrate directional propagation.

    Returns:
        None:
            The observed firing sequences are printed to standard output.
    """

    events: list[str] = []

    downstream = RecordingOneWayBinder.create(
        None,
        'downstream',
        events,
    )
    upstream = RecordingOneWayBinder.create(
        downstream,
        'upstream',
        events,
    )

    upstream()
    print(f'one-way upstream(): {events}')

    events.clear()
    downstream()
    print(f'one-way downstream(): {events}')


def demonstrate_two_way() -> None:
    """Demonstrate reciprocal propagation using one existing endpoint.

    Returns:
        None:
            The observed firing sequences are printed to standard output.
    """

    events: list[str] = []

    left = RecordingOneWayBinder.create(
        None,
        'left',
        events,
    )
    right = RecordingTwoWayBinder.create(
        left,
        'right',
        events,
    )

    left()
    print(f'two-way left(): {events}')

    events.clear()
    right()
    print(f'two-way right(): {events}')


def main() -> None:
    """Run all binding demonstrations.

    Returns:
        None:
            Demonstration results are written to standard output.
    """

    demonstrate_one_way()
    demonstrate_two_way()


if __name__ == '__main__':
    main()
