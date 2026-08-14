"""
Provide circuit-breaker state management for retry policies.

This module defines :class:`CircuitBreaker`, the state machine used by
:class:`aura.retry.RetryPolicy` to stop repeatedly invoking an unhealthy
operation.

A circuit breaker and a bulkhead solve different problems:

* :class:`CircuitBreaker` observes logical operation failures and temporarily
  prevents new calls when a dependency appears unhealthy.
* :class:`aura.bulkhead.Bulkhead` limits concurrent execution and is a
  standalone isolation pattern. It is not part of :class:`RetryPolicy`.

The circuit breaker owns three states:

``CLOSED``
    Normal operating state. Calls are allowed.

    A logical call that ultimately succeeds resets the consecutive failure
    count.

    A logical call that exhausts all retry attempts increments the failure
    count. Reaching ``failure_threshold`` opens the circuit.

``OPEN``
    Calls are rejected immediately without invoking the protected operation
    and without consuming retry attempts.

    The circuit remains open for ``recovery_timeout_seconds``.

``HALF_CLOSED``
    Recovery-probe state. Once the open timeout expires, exactly one logical
    call is admitted as a probe.

    If the probe succeeds, the circuit returns to ``CLOSED``.

    If the probe exhausts its retries, the circuit returns to ``OPEN`` and
    restarts the recovery timeout.

    While the probe is in flight, other calls are rejected.

``HALF_CLOSED`` is the terminology used by Aura. Many descriptions of the
circuit-breaker pattern call this same recovery state ``HALF_OPEN``.

Logical-call accounting
-----------------------

The breaker is intended to operate around a complete
:class:`aura.retry.RetryPolicy` invocation.

For example, with two retries::

    initial attempt -> failure
    retry 1         -> failure
    retry 2         -> failure

the breaker records **one** logical failure, not three failures. This prevents
the retry mechanism itself from artificially accelerating the breaker toward
``OPEN``.

Example:
    Create a breaker that opens after three exhausted logical calls and
    probes recovery after five seconds::

        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout_seconds=5.0,
        )

    Supply it to a retry policy::

        RetryPolicy.create(
            operation,
            2,
            breaker,
        )

    Callers continue invoking the original operation::

        operation()

Notes:
    State transitions use :func:`time.monotonic`; wall-clock adjustments do
    not affect recovery timing.

    Internal state is protected by :class:`threading.Lock`.

    A circuit breaker limits repeated application-level calls to a failing
    dependency. It is not by itself a complete DDoS defense. Network-edge
    filtering, authentication, rate limiting, and traffic shaping remain
    separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from threading import Lock
from time import monotonic
from typing import Final


class CircuitBreakerState(Enum):
    """Identify the current state of a :class:`CircuitBreaker`.

    Values:
        CLOSED:
            Normal state in which logical calls may execute.

        OPEN:
            Protective state in which logical calls are rejected immediately.

        HALF_CLOSED:
            Recovery state in which one logical probe call is permitted.
    """

    CLOSED = auto()
    OPEN = auto()
    HALF_CLOSED = auto()


class CircuitBreakerError(RuntimeError):
    """Base exception for circuit-breaker admission failures."""


class CircuitBreakerOpenError(CircuitBreakerError):
    """Report that the breaker rejected a call because the circuit is open."""


@dataclass(frozen=True, slots=True)
class _CircuitPermit:
    """Describe one logical call admitted by a circuit breaker.

    Attributes:
        half_closed_probe (bool):
            ``True`` when the admitted logical call is the single recovery
            probe allowed in ``HALF_CLOSED`` state.
    """

    half_closed_probe: bool


class CircuitBreaker:
    """Track logical failures and prevent calls to an unhealthy dependency.

    ``CircuitBreaker`` contains no retry loop and invokes no application code
    itself. :class:`aura.retry.RetryPolicy` asks the breaker for admission,
    executes its complete retry sequence, and then reports one logical success
    or one logical failure.

    Attributes:
        state (CircuitBreakerState):
            Current ``CLOSED``, ``OPEN``, or ``HALF_CLOSED`` state.

        failure_count (int):
            Number of consecutive exhausted logical calls recorded while the
            circuit is closed.

        failure_threshold (int):
            Number of consecutive logical failures required to open the
            circuit.

        recovery_timeout (float):
            Number of seconds the circuit remains open before recovery can be
            probed.

        remaining_open_seconds (float):
            Approximate number of seconds remaining before ``OPEN`` may
            transition to ``HALF_CLOSED``.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
    ) -> None:
        """Initialize a closed circuit breaker.

        Args:
            failure_threshold (int, optional):
                Consecutive exhausted logical calls required to transition
                from ``CLOSED`` to ``OPEN``. Defaults to ``5``.

            recovery_timeout_seconds (float, optional):
                Seconds the circuit remains ``OPEN`` before one recovery probe
                is permitted. Defaults to ``30.0``.

        Returns:
            None:
                The breaker begins in ``CLOSED`` state.

        Raises:
            ValueError:
                Raised when ``failure_threshold`` is less than one or
                ``recovery_timeout_seconds`` is negative.
        """

        if failure_threshold < 1:
            raise ValueError(
                'failure_threshold must be greater than or equal to 1'
            )

        if recovery_timeout_seconds < 0.0:
            raise ValueError(
                'recovery_timeout_seconds must be greater than or equal to 0'
            )

        self.__failure_threshold: Final[int] = failure_threshold
        self.__recovery_timeout_seconds: Final[float] = float(
            recovery_timeout_seconds
        )

        self.__lock: Final[Lock] = Lock()
        self.__state: CircuitBreakerState = CircuitBreakerState.CLOSED
        self.__failure_count: int = 0
        self.__opened_at: float | None = None
        self.__half_closed_probe_in_flight: bool = False

    def _acquire(self) -> _CircuitPermit:
        """Admit one logical call when the current state permits it.

        Returns:
            _CircuitPermit:
                Permit describing whether the admitted call is the
                ``HALF_CLOSED`` recovery probe.

        Raises:
            CircuitBreakerOpenError:
                Raised while the breaker is ``OPEN`` or when another
                half-closed recovery probe is already in flight.

        Notes:
            This operation does not execute the protected operation. It only
            performs the breaker admission decision.
        """

        with self.__lock:
            self.__refresh_locked()

            if self.__state is CircuitBreakerState.OPEN:
                raise CircuitBreakerOpenError(
                    'circuit breaker is open; call rejected'
                )

            if self.__state is CircuitBreakerState.HALF_CLOSED:
                if self.__half_closed_probe_in_flight:
                    raise CircuitBreakerOpenError(
                        'circuit breaker is half-closed; '
                        'recovery probe already in flight'
                    )

                self.__half_closed_probe_in_flight = True

                return _CircuitPermit(
                    half_closed_probe=True,
                )

            return _CircuitPermit(
                half_closed_probe=False,
            )

    def _record_success(self, permit: _CircuitPermit) -> None:
        """Record one successful logical call.

        Args:
            permit (_CircuitPermit):
                Permit associated with the completed logical call.

        Returns:
            None:
                State and counters are updated in place.

        Notes:
            A successful half-closed probe closes the circuit.

            A successful closed-state logical call resets the consecutive
            failure count.

            A late success from a call admitted before another concurrent call
            opened the circuit does not close that newer open state.
        """

        with self.__lock:
            if permit.half_closed_probe:
                if self.__state is CircuitBreakerState.HALF_CLOSED:
                    self.__close_locked()

                return

            if self.__state is CircuitBreakerState.CLOSED:
                self.__failure_count = 0

    def _record_failure(self, permit: _CircuitPermit) -> None:
        """Record one exhausted logical failure.

        Args:
            permit (_CircuitPermit):
                Permit associated with the failed logical call.

        Returns:
            None:
                Failure counters and state are updated.

        Notes:
            A failed half-closed recovery probe immediately reopens the
            circuit.

            While closed, each exhausted RetryPolicy invocation increments
            the counter exactly once, regardless of how many physical retry
            attempts were executed.
        """

        with self.__lock:
            if permit.half_closed_probe:
                if self.__state is CircuitBreakerState.HALF_CLOSED:
                    self.__open_locked()

                return

            if self.__state is not CircuitBreakerState.CLOSED:
                return

            self.__failure_count += 1

            if self.__failure_count >= self.__failure_threshold:
                self.__open_locked()

    def _release(self, permit: _CircuitPermit) -> None:
        """Release half-closed probe ownership when still applicable.

        Args:
            permit (_CircuitPermit):
                Permit associated with a completed or interrupted logical call.

        Returns:
            None:
                Probe ownership is cleared when necessary.

        Notes:
            Normally :meth:`_record_success` or :meth:`_record_failure`
            transitions a half-closed probe out of ``HALF_CLOSED`` first.

            This release path is still necessary for process-control
            :class:`BaseException` subclasses, which RetryPolicy intentionally
            does not treat as ordinary operation failures.
        """

        if not permit.half_closed_probe:
            return

        with self.__lock:
            if self.__state is CircuitBreakerState.HALF_CLOSED:
                self.__half_closed_probe_in_flight = False

    def __refresh_locked(self) -> None:
        """Advance an expired open circuit into half-closed recovery.

        Returns:
            None:
                State changes only when the open timeout has elapsed.

        Notes:
            The caller must hold ``self.__lock``.
        """

        if self.__state is not CircuitBreakerState.OPEN:
            return

        if self.__opened_at is None:
            return

        elapsed = monotonic() - self.__opened_at

        if elapsed < self.__recovery_timeout_seconds:
            return

        self.__state = CircuitBreakerState.HALF_CLOSED
        self.__half_closed_probe_in_flight = False

    def __open_locked(self) -> None:
        """Transition to ``OPEN`` and restart the recovery timeout.

        Returns:
            None:
                State is updated in place.

        Notes:
            The caller must hold ``self.__lock``.
        """

        self.__state = CircuitBreakerState.OPEN
        self.__opened_at = monotonic()
        self.__half_closed_probe_in_flight = False

    def __close_locked(self) -> None:
        """Transition to ``CLOSED`` and reset failure state.

        Returns:
            None:
                State is updated in place.

        Notes:
            The caller must hold ``self.__lock``.
        """

        self.__state = CircuitBreakerState.CLOSED
        self.__failure_count = 0
        self.__opened_at = None
        self.__half_closed_probe_in_flight = False

    @property
    def state(self) -> CircuitBreakerState:
        """Return the current breaker state.

        Returns:
            CircuitBreakerState:
                Current ``CLOSED``, ``OPEN``, or ``HALF_CLOSED`` state.

        Notes:
            Reading this property performs the time-based
            ``OPEN -> HALF_CLOSED`` transition when the recovery timeout has
            elapsed.
        """

        with self.__lock:
            self.__refresh_locked()
            return self.__state

    @property
    def failure_count(self) -> int:
        """Return the consecutive exhausted logical failure count.

        Returns:
            int:
                Current closed-state failure count.
        """

        with self.__lock:
            return self.__failure_count

    @property
    def failure_threshold(self) -> int:
        """Return the number of logical failures required to open the circuit.

        Returns:
            int:
                Configured failure threshold.
        """

        return self.__failure_threshold

    @property
    def recovery_timeout(self) -> float:
        """Return the configured open-state timeout.

        Returns:
            float:
                Recovery timeout in seconds.
        """

        return self.__recovery_timeout_seconds

    @property
    def remaining_open_seconds(self) -> float:
        """Return the approximate remaining open-state duration.

        Returns:
            float:
                Non-negative seconds before half-closed recovery may begin.
                ``0.0`` is returned when the circuit is not currently open.
        """

        with self.__lock:
            self.__refresh_locked()

            if (
                self.__state is not CircuitBreakerState.OPEN
                or self.__opened_at is None
            ):
                return 0.0

            elapsed = monotonic() - self.__opened_at

            return max(
                0.0,
                self.__recovery_timeout_seconds - elapsed,
            )
