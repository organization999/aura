"""
Provide standalone concurrency isolation through the Bulkhead pattern.

This module defines :class:`Bulkhead`, reciprocal binder middleware that
limits the number of concurrent invocations allowed to execute a managed
binder.

A bulkhead is independent of :class:`aura.retry.RetryPolicy` and independent
of :class:`aura.breaker.CircuitBreaker`.

The pattern is named after physical bulkheads in ships: one overloaded or
failing compartment should not consume all resources available to unrelated
work.

Relationship
------------

``Bulkhead`` derives from :class:`aura.binding.TwoWayBinder`.

Creating::

    Bulkhead.create(operation, 8)

reuses the existing operation and adds one new middleware endpoint::

    operation -> bulkhead
    bulkhead  -> operation

Callers continue invoking the original operation::

    operation()

The bulkhead intercepts that invocation before ``operation.fire()`` executes.
If capacity is available, the bulkhead acquires one permit, resumes the
operation while bypassing itself, and releases the permit afterward.

If all capacity is occupied, the invocation fails immediately with
:class:`BulkheadFullError`.

Example:
    Protect an operation with at most eight concurrent executions::

        operation = Operation.create()
        Bulkhead.create(operation, 8)

        operation()

Notes:
    ``Bulkhead`` does not implement retries.

    ``Bulkhead`` does not own ``OPEN``, ``CLOSED``, or ``HALF_CLOSED`` states.
    Those states belong to :class:`aura.breaker.CircuitBreaker`.

    Admission is non-blocking. Excess concurrent calls are rejected rather
    than queued indefinitely.
"""

from __future__ import annotations

from threading import BoundedSemaphore
from typing import Final

from .binding import Binder, TwoWayBinder


class BulkheadError(RuntimeError):
    """Base exception for bulkhead admission failures."""


class BulkheadFullError(BulkheadError):
    """Report that every configured concurrent execution slot is occupied."""


class Bulkhead(TwoWayBinder):
    """Limit concurrent execution of one managed binder.

    ``Bulkhead`` is a standalone binder interceptor. It contains no retry
    policy and no circuit-breaker state.

    Attributes:
        max_concurrent_calls (int):
            Maximum number of managed invocations that may execute
            simultaneously.
    """

    def __init__(
        self,
        bindee: Binder | None = None,
        max_concurrent_calls: int = 32,
    ) -> None:
        """Initialize one factory-allocated bulkhead.

        Args:
            bindee (Binder | None, optional):
                Existing binder whose concurrent execution should be limited.

            max_concurrent_calls (int, optional):
                Maximum number of simultaneous managed executions. Defaults
                to ``32``.

        Returns:
            None:
                The already allocated bulkhead is initialized.

        Raises:
            ValueError:
                Raised when ``bindee`` is ``None`` or when
                ``max_concurrent_calls`` is less than one.
        """

        if bindee is None:
            raise ValueError('Bulkhead requires an existing bindee')

        if max_concurrent_calls < 1:
            raise ValueError(
                'max_concurrent_calls must be greater than or equal to 1'
            )

        super().__init__(bindee)

        self.__bindee: Final[Binder] = bindee
        self.__max_concurrent_calls: Final[int] = max_concurrent_calls
        self.__capacity: Final[BoundedSemaphore] = BoundedSemaphore(
            max_concurrent_calls
        )

    def _intercept(
        self,
        source: Binder,
        visited: set[int],
    ) -> bool:
        """Intercept and capacity-limit the managed binder invocation.

        Args:
            source (Binder):
                Binder whose local :meth:`Binder.fire` operation is about to
                execute.

            visited (set[int]):
                Invocation-local identities already visited by the original
                public invocation.

        Returns:
            bool:
                ``True`` when ``source`` is the managed bindee. The original
                invocation is consumed because this bulkhead performs the
                protected execution itself.

                ``False`` is returned for unrelated sources.

        Raises:
            BulkheadFullError:
                Raised when every execution permit is currently occupied.

        Notes:
            The outer visited set is not reused for the resumed execution
            because it already contains the managed bindee's identity.
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
        """Execute the protected operation when the bulkhead is called directly.

        Args:
            source (Binder | None, optional):
                Binder that forwarded the invocation, if any.

            visited (set[int] | None, optional):
                Invocation-local visited state, if any.

        Returns:
            None:
                Returns after one admitted managed execution completes.

        Raises:
            BulkheadFullError:
                Raised when no capacity is available.
        """

        del source, visited
        self.fire()

    def fire(self) -> None:
        """Execute one managed invocation under concurrency isolation.

        Returns:
            None:
                Returns after the managed invocation completes.

        Raises:
            BulkheadFullError:
                Raised when no execution slot is available.
        """

        self.__execute()

    def __execute(self) -> None:
        """Acquire one slot, execute the managed binder, and release the slot.

        Returns:
            None:
                Managed work completed successfully.

        Raises:
            BulkheadFullError:
                Raised immediately when every configured slot is occupied.

            Exception:
                Any exception from the managed binder propagates unchanged.

        Notes:
            Capacity release occurs in ``finally`` and therefore also occurs
            when the managed operation raises.
        """

        if not self.__capacity.acquire(blocking=False):
            raise BulkheadFullError(
                'bulkhead concurrency capacity exhausted'
            )

        try:
            self.__bindee._invoke(
                source=self,
                visited={id(self)},
            )
        finally:
            self.__capacity.release()

    @property
    def max_concurrent_calls(self) -> int:
        """Return the configured concurrency limit.

        Returns:
            int:
                Maximum simultaneous managed executions.
        """

        return self.__max_concurrent_calls
