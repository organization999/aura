"""
Demonstrate CircuitBreaker state transitions through RetryPolicy.

:class:`aura.breaker.CircuitBreaker` is designed to protect complete logical
calls. In normal Aura usage, :class:`aura.retry.RetryPolicy` asks the breaker
for admission, executes its retry sequence, and reports one logical success
or failure.

This example uses a service that initially fails every invocation.

With ``failure_threshold=2`` and no retries:

1. The first logical failure leaves the breaker ``CLOSED``.
2. The second logical failure transitions it to ``OPEN``.
3. A third call fails fast with :class:`CircuitBreakerOpenError`.
4. After the recovery timeout, the breaker becomes ``HALF_CLOSED``.
5. The service is marked healthy and the recovery probe succeeds.
6. The breaker returns to ``CLOSED``.

Run:
    From the repository root::

        python examples/breaker.py
"""

from __future__ import annotations

from pathlib import Path
import sys
from time import sleep

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from aura.binding import Binder, OneWayBinder
from aura.breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)
from aura.retry import RetryPolicy


class RemoteOperation(OneWayBinder):
    """Represent a dependency that can be switched between failed and healthy."""

    def __init__(
        self,
        bindee: Binder | None = None,
        healthy: bool = False,
    ) -> None:
        """Initialize the simulated remote operation.

        Args:
            bindee (Binder | None, optional):
                Optional downstream binder.

            healthy (bool, optional):
                Whether calls should currently succeed. Defaults to ``False``.

        Returns:
            None:
                The operation is initialized.
        """

        super().__init__(bindee)

        self.healthy = healthy
        self.calls = 0

    def fire(self) -> None:
        """Execute one simulated remote request.

        Returns:
            None:
                Returns normally when the dependency is healthy.

        Raises:
            RuntimeError:
                Raised while the dependency is unhealthy.
        """

        self.calls += 1

        if not self.healthy:
            raise RuntimeError('remote dependency unavailable')

        print(f'recovery call {self.calls}: success')


def main() -> None:
    """Drive the breaker through closed, open, half-closed, and closed states.

    Returns:
        None:
            State transitions and rejection behavior are printed.
    """

    operation = RemoteOperation.create()
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout_seconds=0.25,
    )

    RetryPolicy.create(
        operation,
        0,
        breaker,
    )

    for logical_call in range(1, 3):
        try:
            operation()
        except RuntimeError as error:
            print(
                f'logical call {logical_call}: {error}; '
                f'state={breaker.state.name}'
            )

    try:
        operation()
    except CircuitBreakerOpenError as error:
        print(f'open circuit: {error}')

    sleep(0.30)

    assert breaker.state is CircuitBreakerState.HALF_CLOSED

    operation.healthy = True
    operation()

    print(f'final state={breaker.state.name}')


if __name__ == '__main__':
    main()
