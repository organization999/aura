"""
Provide quiet-period debouncing for deferred backup operations.

This module defines :class:`Debouncer`, middleware that attaches to an
existing :class:`aura.binding.Binder` whose :meth:`~aura.binding.Binder.fire`
method performs backup work.

Callers continue using the original managed object. They do not need to call
the debouncer directly.

Example::

    database = Database.create()
    Debouncer.create(database, 5.0)

    database()
    database()
    database()

Each call to ``database()`` represents activity on the managed resource. The
attached debouncer intercepts the invocation before ``Database.fire`` executes
and arms or resets a five-second quiet-period timer.

Only after five uninterrupted seconds does the debouncer resume the managed
binder and allow ``Database.fire`` to execute once.

Conceptually::

    database access
          |
          v
    reset backup timer
          |
          | another access
          v
    reset backup timer
          |
          | no access for full interval
          v
    Database.fire()
          |
          v
       backup

Construction itself does not arm the timer. Merely attaching backup middleware
must not create a backup for a resource that has never been accessed.

Relationship
------------

:class:`Debouncer` derives from :class:`aura.binding.TwoWayBinder`. Creation
adds one new debouncer endpoint and reuses the existing resource::

    managed   -> debouncer
    debouncer -> managed

The ``managed -> debouncer`` edge gives the debouncer a pre-fire interception
opportunity whenever the managed object is invoked.

The ``debouncer -> managed`` edge allows timer expiration to resume the
managed binder after the quiet interval.

Timer behavior
--------------

The first intercepted resource invocation arms the timer. Every subsequent
invocation before expiration cancels the previous timer generation and starts
a full new interval.

A monotonically increasing generation identifier is used in addition to
:meth:`threading.Timer.cancel`. A stale timer callback therefore cannot invoke
the managed binder after a newer generation has replaced it.

Timer threads are non-daemon. This is intentional for backups: normal
interpreter shutdown should not silently discard a pending backup.

Notes:
    This abstraction observes invocations routed through the managed binder.
    It does not automatically observe operating-system file reads or writes
    performed outside the binder abstraction.

    :func:`time.monotonic` is used for quiet-period calculations so wall-clock
    changes do not shorten or extend the debounce interval.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock, Timer
from time import monotonic
from typing import Final

from binding import Binder, TwoWayBinder

class Debouncer(TwoWayBinder):
    """Intercept managed-resource activity and defer its backup action.

    A ``Debouncer`` owns one managed bindee. When that bindee is publicly
    invoked, this object intercepts the invocation before the bindee's
    :meth:`~aura.binding.Binder.fire` method executes.

    Every intercepted invocation restarts the quiet-period timer. The newest
    timer generation eventually resumes the managed bindee after a full period
    of inactivity.

    Attributes:
        duration (timedelta):
            Configured quiet-period duration.

        since (datetime | None):
            UTC timestamp at which the current or most recent timer generation
            was armed. ``None`` before the first managed-resource invocation.

        pending (bool):
            Whether a backup is currently waiting for the quiet period.

        remaining (timedelta):
            Approximate monotonic time remaining before the pending backup.

    Example:
        Attach a one-second backup delay::

            database = Database.create()
            Debouncer.create(database, 1.0)

        Keep using the original resource::

            database()
            database()

        One second after the final call, ``database.fire()`` executes once.
    """

    def __init__(
        self,
        bindee: Binder | None = None,
        duration_seconds: float = 5.0,
    ) -> None:
        """Initialize one factory-allocated debouncer.

        Args:
            bindee (Binder | None, optional):
                Existing resource whose :meth:`Binder.fire` operation should
                be deferred. :meth:`TwoWayBinder.create` requires this value to
                be non-``None``.

            duration_seconds (float, optional):
                Required quiet period in seconds. Every intercepted invocation
                restarts the entire interval. Defaults to ``5.0``.

        Returns:
            None:
                The already allocated debouncer is initialized.

        Raises:
            ValueError:
                Raised when ``bindee`` is ``None`` or when
                ``duration_seconds`` is negative.

        Notes:
            No timer is started during construction. The first intercepted
            resource invocation starts the first pending backup generation.

            A zero duration is permitted. In that case expiration is scheduled
            as soon as the timer thread can run.
        """

        if bindee is None:
            raise ValueError('Debouncer requires an existing bindee')

        if duration_seconds < 0.0:
            raise ValueError(
                'duration_seconds must be greater than or equal to 0'
            )

        super().__init__(bindee)

        self.__bindee: Final[Binder] = bindee
        self.__duration_seconds: Final[float] = float(duration_seconds)
        self.__duration: Final[timedelta] = timedelta(
            seconds=self.__duration_seconds
        )

        self.__lock: Final[Lock] = Lock()
        self.__timer: Timer | None = None
        self.__generation: int = 0
        self.__since: datetime | None = None
        self.__deadline: float | None = None

    def _intercept(
        self,
        source: Binder,
        visited: set[int],
    ) -> bool:
        """Consume activity from the managed resource before it fires.

        Args:
            source (Binder):
                Binder whose local action is about to execute.

            visited (set[int]):
                Invocation-local identities that have already participated.

        Returns:
            bool:
                ``True`` when ``source`` is this debouncer's managed bindee.
                Returning ``True`` causes the source binder to stop before its
                :meth:`Binder.fire` implementation executes. ``False`` is
                returned for unrelated sources.

        Notes:
            The visited set is accepted as part of the interception protocol
            but does not need to be modified by this debouncer.

            This is the operation that converts an ordinary managed-resource
            invocation into a timer reset.
        """

        del visited

        if source is not self.__bindee:
            return False

        self.reset()
        return True

    def _invoke(
        self,
        source: Binder | None = None,
        visited: set[int] | None = None,
    ) -> None:
        """Treat direct debouncer invocation as a timer reset.

        Normal application code should invoke the managed resource rather than
        this middleware object. This override nevertheless makes a direct
        debouncer invocation safe: it resets the timer without immediately
        propagating to the managed resource.

        Args:
            source (Binder | None, optional):
                Binder that forwarded the invocation, if any.

            visited (set[int] | None, optional):
                Invocation-local visited state, if any.

        Returns:
            None:
                The quiet-period timer is reset.
        """

        del source, visited
        self.reset()

    def fire(self) -> None:
        """Restart the quiet-period timer.

        ``fire`` is the debouncer's local binder action. It never executes the
        managed resource directly.

        Returns:
            None:
                Pending backup state is updated.
        """

        self.reset()

    def reset(self) -> None:
        """Arm or restart the complete quiet-period interval.

        Any current timer is cancelled. A new generation identifier, UTC
        timestamp, monotonic deadline, and :class:`threading.Timer` are then
        installed.

        Returns:
            None:
                Pending backup state is replaced by a new generation.

        Notes:
            :meth:`Timer.cancel` alone cannot eliminate a race with a timer
            callback that is already starting. The generation identifier is
            therefore validated again by :meth:`__expire`.

            The timer is non-daemon so a normal interpreter shutdown waits for
            a pending backup rather than abandoning it.
        """

        with self.__lock:
            previous = self.__timer

            if previous is not None:
                previous.cancel()

            self.__generation += 1
            generation: int = self.__generation

            self.__since = datetime.now(timezone.utc)
            self.__deadline = monotonic() + self.__duration_seconds

            timer = Timer(
                self.__duration_seconds,
                self.__expire,
                args=(generation,),
            )
            timer.daemon = False

            self.__timer = timer
            timer.start()

    def cancel(self) -> None:
        """Cancel the currently pending backup.

        Cancellation invalidates the current timer generation and clears the
        pending timer state. The binding relationship remains installed, so a
        later managed-resource invocation can arm another backup.

        Returns:
            None:
                Pending timer state is cleared.
        """

        with self.__lock:
            self.__generation += 1

            timer = self.__timer
            self.__timer = None
            self.__deadline = None

            if timer is not None:
                timer.cancel()

    def flush(self) -> None:
        """Immediately execute a pending backup once.

        If no backup is pending, this operation does nothing. Otherwise the
        pending timer is cancelled and invalidated before the managed binder is
        resumed synchronously.

        Returns:
            None:
                A pending backup is executed synchronously when present.

        Notes:
            ``flush`` is useful during controlled shutdown when waiting for the
            remainder of the quiet interval is undesirable.
        """

        with self.__lock:
            if self.__timer is None:
                return

            self.__generation += 1

            timer = self.__timer
            self.__timer = None
            self.__deadline = None

            timer.cancel()

        self.__resume_bindee()

    def __expire(self, generation: int) -> None:
        """Resume the managed binder if this timer generation is still current.

        Args:
            generation (int):
                Generation captured when this timer was armed.

        Returns:
            None:
                Stale generations return without side effects. The current
                generation resumes the managed binder once.

        Notes:
            Managed work begins only after the timer-state lock is released.

            Resumption identifies this debouncer as the invocation source and
            marks it as already visited. The reciprocal edge is therefore
            skipped by the managed binder, allowing the managed
            :meth:`Binder.fire` implementation to execute without immediately
            resetting the timer again.
        """

        with self.__lock:
            if generation != self.__generation:
                return

            if self.__timer is None:
                return

            self.__timer = None
            self.__deadline = None

        self.__resume_bindee()

    def __resume_bindee(self) -> None:
        """Resume the managed invocation while bypassing this interceptor.

        Returns:
            None:
                The managed binder continues through its normal internal
                invocation path and may execute :meth:`Binder.fire`.
        """

        self.__bindee._invoke(
            source=self,
            visited={id(self)},
        )

    @property
    def duration(self) -> timedelta:
        """Return the configured quiet-period duration.

        Returns:
            timedelta:
                Debounce interval supplied during construction.
        """

        return self.__duration

    @property
    def since(self) -> datetime | None:
        """Return the most recent timer-arm timestamp.

        Returns:
            datetime | None:
                Timezone-aware UTC timestamp corresponding to the newest
                intercepted activity, or ``None`` before the first timer is
                armed.
        """

        with self.__lock:
            return self.__since

    @property
    def pending(self) -> bool:
        """Return whether a backup is waiting for the quiet period.

        Returns:
            bool:
                ``True`` while a timer generation is pending; otherwise
                ``False``.
        """

        with self.__lock:
            return self.__timer is not None

    @property
    def remaining(self) -> timedelta:
        """Return the approximate monotonic time remaining.

        Returns:
            timedelta:
                Non-negative remaining duration for the current generation.
                ``timedelta(0)`` is returned when no backup is pending.

        Notes:
            This value is observational. The timer may expire immediately
            after the property releases its lock.
        """

        with self.__lock:
            if self.__timer is None or self.__deadline is None:
                return timedelta(0)

            seconds = max(0.0, self.__deadline - monotonic())

        return timedelta(seconds=seconds)
