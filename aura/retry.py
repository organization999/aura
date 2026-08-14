"""
Provide transparent retries integrated with a circuit breaker.

This module defines :class:`RetryPolicy`, reciprocal binder middleware that
retries an existing :class:`aura.binding.Binder` when its local operation
raises an exception.

RetryPolicy uses :class:`aura.breaker.CircuitBreaker` to prevent repeated
logical calls from continuing to hit an unhealthy dependency.

The Bulkhead pattern is deliberately not part of RetryPolicy. Concurrency
isolation is provided independently by :class:`aura.bulkhead.Bulkhead`.

Relationship
------------

``RetryPolicy`` derives from :class:`aura.binding.TwoWayBinder`. Creating a
policy reuses the existing managed object and installs::

    managed      -> retry_policy
    retry_policy -> managed

The policy intercepts the public managed invocation before
``managed.fire()`` executes, consumes that original invocation, and executes
the managed binder through its retry loop.

Retry behavior
--------------

``retry_attempts`` counts retries after the initial attempt.

For example, ``retry_attempts=3`` permits at most four executions:

1. Initial attempt.
2. Retry 1.
3. Retry 2.
4. Retry 3.

Circuit-breaker behavior
------------------------

The breaker evaluates the complete retry sequence as one logical call.

A successful attempt records one logical success.

Exhausting the initial attempt and all configured retries records one logical
failure.

When the failure threshold is reached, the breaker becomes ``OPEN`` and new
calls are rejected without executing any attempts.

After the configured recovery timeout the breaker becomes ``HALF_CLOSED`` and
permits one probe logical call. A successful probe closes the breaker; a
failed probe opens it again.

Example:
    Configure a retry policy with a circuit breaker::

        operation = Operation.create()

        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout_seconds=5.0,
        )

        policy = RetryPolicy.create(
            operation,
            2,
            breaker,
        )

    Continue calling the original operation::

        operation()

    Inspect breaker state through either::

        breaker.state

    or::

        policy.state

Notes:
    Retryable failures are exceptions derived from :class:`Exception`.

    :class:`KeyboardInterrupt`, :class:`SystemExit`, and
    :class:`GeneratorExit` are not retried and are not converted into
    ordinary circuit failures.

    Retries are synchronous and currently begin immediately after failure.
"""

from __future__ import annotations

from typing import Final

from .binding import Binder, TwoWayBinder
from .breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    _CircuitPermit,
)


class RetryPolicy(TwoWayBinder):
    """Retry one managed binder under circuit-breaker protection.

    Application code continues invoking the original managed binder.

    Attributes:
        retry_attempts (int):
            Number of retries permitted after the initial attempt.

        max_attempts (int):
            Maximum total executions for one logical call.

        breaker (CircuitBreaker):
            Circuit breaker guarding logical calls.

        state (CircuitBreakerState):
            Convenience view of the breaker's current state.
    """

    def __init__(
        self,
        bindee: Binder | None = None,
        retry_attempts: int = 3,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        """Initialize one factory-allocated retry policy.

        Args:
            bindee (Binder | None, optional):
                Existing binder whose operation should be retried.

            retry_attempts (int, optional):
                Number of additional executions permitted after the initial
                attempt. Defaults to ``3``.

            breaker (CircuitBreaker | None, optional):
                Circuit breaker controlling logical-call admission. If
                ``None``, a default :class:`CircuitBreaker` is created.
                Defaults to ``None``.

        Returns:
            None:
                The already allocated retry policy is initialized.

        Raises:
            ValueError:
                Raised when ``bindee`` is ``None`` or ``retry_attempts`` is
                negative.
        """

        if bindee is None:
            raise ValueError('RetryPolicy requires an existing bindee')

        if retry_attempts < 0:
            raise ValueError(
                'retry_attempts must be greater than or equal to 0'
            )

        super().__init__(bindee)

        self.__bindee: Final[Binder] = bindee
        self.__retry_attempts: Final[int] = retry_attempts
        self.__breaker: Final[CircuitBreaker] = (
            CircuitBreaker()
            if breaker is None
            else breaker
        )

    def _intercept(
        self,
        source: Binder,
        visited: set[int],
    ) -> bool:
        """Intercept and retry invocation of the managed binder.

        Args:
            source (Binder):
                Binder whose local action is about to execute.

            visited (set[int]):
                Invocation-local identities already visited.

        Returns:
            bool:
                ``True`` when ``source`` is the managed bindee. ``False`` is
                returned for unrelated binders.

        Raises:
            CircuitBreakerOpenError:
                May propagate when the breaker rejects the logical call.

            Exception:
                Re-raises the final operation failure after retry exhaustion.
        """

        del visited

        if source is not self.__bindee:
            return False

        self.__execute()
        return True

    def _invoke(
        self,
        source: Binder | None = None,
        visited: set[int] | None = None,
    ) -> None:
        """Execute this retry policy directly.

        Normal callers should invoke the managed binder instead.

        Args:
            source (Binder | None, optional):
                Immediate forwarding source, if any.

            visited (set[int] | None, optional):
                Invocation-local visited state, if any.

        Returns:
            None:
                Returns when the logical call succeeds.

        Raises:
            Exception:
                May propagate breaker rejection or the final managed failure.
        """

        del source, visited
        self.fire()

    def fire(self) -> None:
        """Execute one protected retry sequence.

        Returns:
            None:
                Returns as soon as one attempt succeeds.

        Raises:
            Exception:
                May propagate breaker rejection or the final managed failure.
        """

        self.__execute()

    def __execute(self) -> None:
        """Execute one logical call through the breaker and retry loop.

        Returns:
            None:
                Returns when any attempt succeeds.

        Raises:
            CircuitBreakerOpenError:
                Raised before any attempt when the breaker rejects the call.

            Exception:
                Re-raises the final managed exception after retry exhaustion.

        Notes:
            The breaker permit is acquired once for the complete retry
            sequence.

            Only retry exhaustion records a logical breaker failure.

            Capacity/concurrency limiting is intentionally absent. That concern
            belongs to the independent :class:`aura.bulkhead.Bulkhead`
            pattern.
        """

        permit: _CircuitPermit = self.__breaker._acquire()

        try:
            for attempt in range(1, self.max_attempts + 1):
                try:
                    self.__bindee._invoke(
                        source=self,
                        visited={id(self)},
                    )
                except Exception:
                    if attempt < self.max_attempts:
                        continue

                    self.__breaker._record_failure(permit)
                    raise

                self.__breaker._record_success(permit)
                return
        finally:
            self.__breaker._release(permit)

    @property
    def retry_attempts(self) -> int:
        """Return the configured retry count.

        Returns:
            int:
                Additional executions allowed after the initial attempt.
        """

        return self.__retry_attempts

    @property
    def max_attempts(self) -> int:
        """Return the maximum total execution count.

        Returns:
            int:
                Initial attempt plus configured retries.
        """

        return self.__retry_attempts + 1

    @property
    def breaker(self) -> CircuitBreaker:
        """Return the circuit breaker used by this retry policy.

        Returns:
            CircuitBreaker:
                Logical-call circuit breaker.
        """

        return self.__breaker

    @property
    def state(self) -> CircuitBreakerState:
        """Return the retry policy's current circuit-breaker state.

        Returns:
            CircuitBreakerState:
                ``CLOSED``, ``OPEN``, or ``HALF_CLOSED``.
        """

        return self.__breaker.state
