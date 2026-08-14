"""
Demonstrate transparent retries around an existing Aura binder.

:class:`aura.retry.RetryPolicy` is attached reciprocally to the managed
operation. Application code continues invoking the original operation.

The example fails its first two physical attempts and succeeds on the third.
A policy configured with ``retry_attempts=2`` therefore completes the logical
call successfully without exposing either intermediate failure to the caller.

Run:
    From the repository root::

        python examples/retry.py
"""

from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.binding import Binder, OneWayBinder
from aura.retry import RetryPolicy


class FlakyOperation(OneWayBinder):
    """Fail a configurable number of attempts before succeeding."""

    def __init__(
        self,
        bindee: Binder | None = None,
        failures_before_success: int = 2,
    ) -> None:
        """Initialize the flaky operation.

        Args:
            bindee (Binder | None, optional):
                Optional downstream binder.

            failures_before_success (int, optional):
                Number of initial physical attempts that should fail.
                Defaults to ``2``.

        Returns:
            None:
                The operation is initialized.
        """

        super().__init__(bindee)

        self.failures_before_success = failures_before_success
        self.calls = 0

    def fire(self) -> None:
        """Execute one physical attempt.

        Returns:
            None:
                Returns normally after enough failures have occurred.

        Raises:
            RuntimeError:
                Raised for the configured number of initial failures.
        """

        self.calls += 1
        print(f'physical attempt #{self.calls}')

        if self.calls <= self.failures_before_success:
            raise RuntimeError('temporary failure')

        print('logical call succeeded')


def main() -> None:
    """Run one logical call through a two-retry policy.

    Returns:
        None:
            Physical attempt count and final breaker state are printed.
    """

    operation = FlakyOperation.create(
        None,
        2,
    )
    policy = RetryPolicy.create(
        operation,
        2,
    )

    operation()

    print(f'total physical attempts={operation.calls}')
    print(f'breaker state={policy.state.name}')


if __name__ == '__main__':
    main()
