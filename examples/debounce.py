"""
Demonstrate quiet-period backup debouncing.

The application continues invoking the original resource. The attached
:class:`aura.debounce.Debouncer` intercepts each invocation before the
resource's :meth:`fire` method executes and restarts the quiet-period timer.

Three accesses are issued closer together than the configured debounce
interval. The backup therefore executes only once, after the final access has
remained quiet for the full interval.

Run:
    From the repository root::

        python examples/debounce.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from threading import Event
from time import sleep

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.binding import Binder, OneWayBinder
from aura.debounce import Debouncer


class BackupOperation(OneWayBinder):
    """Represent the deferred backup operation."""

    def __init__(
        self,
        bindee: Binder | None = None,
        completed: Event | None = None,
    ) -> None:
        """Initialize the backup operation.

        Args:
            bindee (Binder | None, optional):
                Optional downstream binder.

            completed (Event | None, optional):
                Event set after the backup executes.

        Returns:
            None:
                The operation is initialized.
        """

        super().__init__(bindee)

        self.completed = Event() if completed is None else completed
        self.backup_count = 0

    def fire(self) -> None:
        """Perform one simulated backup.

        Returns:
            None:
                The backup count is incremented and completion is signaled.
        """

        self.backup_count += 1
        print(f'backup #{self.backup_count} written')
        self.completed.set()


def main() -> None:
    """Run the debounce demonstration.

    Returns:
        None:
            Access activity and final backup count are printed.

    Raises:
        TimeoutError:
            Raised if the deferred backup does not complete.
    """

    completed = Event()
    operation = BackupOperation.create(
        None,
        completed,
    )
    debouncer = Debouncer.create(
        operation,
        0.20,
    )

    for access in range(1, 4):
        print(f'resource access #{access}')
        operation()
        sleep(0.08)

    if not completed.wait(timeout=1.0):
        debouncer.cancel()
        raise TimeoutError('debounced backup did not execute')

    print(f'final backup count={operation.backup_count}')


if __name__ == '__main__':
    main()
