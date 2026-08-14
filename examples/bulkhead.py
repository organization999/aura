"""
Demonstrate standalone concurrency isolation with Aura Bulkhead.

A :class:`aura.bulkhead.Bulkhead` limits how many invocations of a managed
binder may execute concurrently. It does not retry failures and does not own
circuit-breaker state.

This example configures one concurrent slot. A worker thread enters the
managed operation and deliberately remains there until the main thread
releases it. While that slot is occupied, a second invocation is rejected
immediately with :class:`aura.bulkhead.BulkheadFullError`.

Run:
    From the repository root::

        python examples/bulkhead.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from threading import Event, Thread

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.binding import Binder, OneWayBinder
from aura.bulkhead import Bulkhead, BulkheadFullError


class BlockingOperation(OneWayBinder):
    """Block inside ``fire`` until the demonstration releases the operation."""

    def __init__(
        self,
        bindee: Binder | None = None,
        entered: Event | None = None,
        release: Event | None = None,
    ) -> None:
        """Initialize the blocking operation.

        Args:
            bindee (Binder | None, optional):
                Optional downstream binder.

            entered (Event | None, optional):
                Event set when execution enters :meth:`fire`.

            release (Event | None, optional):
                Event that permits :meth:`fire` to return.

        Returns:
            None:
                The operation is initialized.
        """

        super().__init__(bindee)

        self.entered = Event() if entered is None else entered
        self.release = Event() if release is None else release

    def fire(self) -> None:
        """Occupy one bulkhead slot until released.

        Returns:
            None:
                Returns after the release event becomes set.

        Raises:
            TimeoutError:
                Raised if the demonstration fails to release the operation
                within two seconds.
        """

        self.entered.set()

        if not self.release.wait(timeout=2.0):
            raise TimeoutError('blocking example was not released')


def main() -> None:
    """Run the one-slot bulkhead demonstration.

    Returns:
        None:
            Admission and rejection results are printed.
    """

    entered = Event()
    release = Event()

    operation = BlockingOperation.create(
        None,
        entered,
        release,
    )
    bulkhead = Bulkhead.create(
        operation,
        1,
    )

    worker_error: list[BaseException] = []

    def worker() -> None:
        """Run the first admitted invocation on a worker thread."""

        try:
            operation()
        except BaseException as error:
            worker_error.append(error)

    thread = Thread(target=worker, name='bulkhead-example-worker')
    thread.start()

    if not entered.wait(timeout=1.0):
        release.set()
        thread.join(timeout=2.0)
        raise TimeoutError('worker never entered protected operation')

    try:
        operation()
    except BulkheadFullError as error:
        print(f'second call rejected: {error}')

    release.set()
    thread.join(timeout=2.0)

    if thread.is_alive():
        raise TimeoutError('worker failed to stop')

    if worker_error:
        raise worker_error[0]

    print(
        'first call completed; '
        f'max_concurrent_calls={bulkhead.max_concurrent_calls}'
    )


if __name__ == '__main__':
    main()
