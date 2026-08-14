"""
Demonstrate backup debouncing through the original managed object.

The example intentionally calls ``example()`` rather than the attached
:class:`Debouncer`. Every call represents resource activity and postpones the
deferred ``Example.fire`` operation.

Running::

    python -m aura

prints exactly one ``Hello, World!`` after the final one-second quiet period.
"""

from __future__ import annotations

from time import sleep

from .binding import OneWayBinder
from .debounce import Debouncer


class Example(OneWayBinder):
    """Provide a minimal backup-like action for the package demonstration."""

    def fire(self) -> None:
        """Print one message when the deferred action is finally allowed.

        Returns:
            None:
                The message is written to standard output.
        """

        print('Hello, World!')


def main() -> None:
    """Run the debounce demonstration.

    Five invocations occur 250 milliseconds apart. Each invocation resets the
    one-second quiet-period timer. ``Example.fire`` therefore executes only
    once, one second after the fifth and final invocation.

    Returns:
        None:
            The demonstration writes its result to standard output.
    """

    example: Example = Example.create()

    # The returned Debouncer does not need a separate public call path.
    # Reciprocal binding keeps it attached to `example`, and callers continue
    # using the original managed object.
    Debouncer.create(example, 1.0)

    for _ in range(5):
        example()
        sleep(0.25)

    # Allow the final quiet period to expire before the demonstration exits.
    sleep(1.25)


if __name__ == '__main__':
    main()
