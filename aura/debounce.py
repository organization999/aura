"""
Provide quiet-period debouncing for deferred backups.

``Debouncer`` attaches to an existing :class:`binding.Binder` whose
:meth:`binding.Binder.fire` method performs backup work.

Callers continue using the original managed object. They do not call the
debouncer.

Example::

    database = Database.create()
    Debouncer.create(database, 5.0)

    database()
    database()
    database()

Each ``database()`` call represents resource activity. The attached debouncer
intercepts the call before ``Database.fire`` executes and resets a five-second
quiet-period timer. Only after five uninterrupted seconds does the debouncer
resume the database invocation, causing ``Database.fire`` to run once.

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
          | quiet period expires
          v
    Database.fire()
          |
          v
       backup

Construction does not arm a timer. Merely installing backup middleware must not
produce a backup when the resource has never been accessed.

The timer thread is non-daemon. This is deliberate for backup work: a normal
interpreter shutdown should not silently discard a pending backup.

Notes:
    This class observes invocations routed through the managed binder. It does
    not automatically detect operating-system file activity performed outside
    that abstraction.

    :func:`time.monotonic` is used for quiet-period calculations so wall-clock
    corrections do not affect the debounce interval.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock, Timer
from time import monotonic
from typing import Final

from binding import Binder, TwoWayBinder


class Debouncer(TwoWayBinder):
    """Intercept managed-resource activity and defer its local backup action.

    Creation establishes the reciprocal relationship::

        managed   -> debouncer
        debouncer -> managed

    The ``managed -> debouncer`` edge lets the debouncer intercept every public
    invocation before ``managed.fire()`` executes.

    The ``debouncer -> managed`` edge lets timer expiration resume that managed
    binder after the quiet period.

    Attributes:
        duration (timedelta):
            Configured quiet period.

        since (datetime | None):
            UTC timestamp of the newest timer arm, or ``None`` before the first
            managed-resource access.

        pending (bool):
            Whether a backup is currently waiting for the quiet period.

        remaining (timedelta):
            Approximate quiet time remaining.
    """

    def __init__(
        self,
        bindee: Binder | None = None,
        duration_seconds: float = 5.0,
    ) -> None:
        """Initialize one factory-allocated debouncer.

        Args:
            bindee (Binder | None, optional):
                Existing resource whose ``fire`` operation should be debounced.

            duration_seconds (float, optional):
                Required quiet period in seconds. Defaults to ``5.0``.

        Returns:
            None:
                The existing allocation is initialized.

        Raises:
            ValueError:
                Raised when ``bindee`` is ``None`` or the duration is negative.

        Notes:
            No timer is armed during construction. The first intercepted access
            starts the first backup timer.
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
                Invocation-local visited identities.

        Returns:
            bool:
                ``True`` when ``source`` is this debouncer's managed bindee;
                otherwise ``False``.

        Notes:
            Consuming the invocation causes :class:`binding.Binder` to skip
            ``source.fire()``. The backup is therefore postponed rather than
            executed and merely observed afterward.
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
        """Safely treat direct debouncer invocation as a timer reset.

        Normal application code should call the managed resource, not the
        debouncer. This override prevents an accidental direct debouncer call
        from immediately propagating to and firing the managed resource.

        Args:
            source (Binder | None, optional):
                Immediate forwarding source, if any.

            visited (set[int] | None, optional):
                Invocation-local visited set, if any.

        Returns:
            None:
                The quiet-period timer is reset.
        """

        del source, visited
        self.reset()

    def fire(self) -> None:
        """Restart the quiet-period timer.

        Returns:
            None:
                Pending backup state is updated.
        """

        self.reset()

    def reset(self) -> None:
        """Arm or restart the complete quiet-period timer.

        Any previous timer is cancelled. A new generation identifier is
        allocated so a stale timer racing with cancellation cannot perform the
        backup.

        Returns:
            None:
                A new pending generation is installed.

        Notes:
            The timer is explicitly non-daemon so normal interpreter shutdown
            waits for pending backup work rather than abandoning it.
        """

        with self.__lock:
            previous = self.__timer

            if previous is not None:
                previous.cancel()

            self.__generation += 1
            generation = self.__generation

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
        """Cancel a pending backup without removing the relationship.

        Returns:
            None:
                Pending timer state is cleared.

        Notes:
            A later managed-resource access can arm a new backup.
        """

        with self.__lock:
            self.__generation += 1

            timer = self.__timer
            self.__timer = None
            self.__deadline = None

            if timer is not None:
                timer.cancel()

    def flush(self) -> None:
        """Immediately perform a pending backup.

        Returns:
            None:
                When a backup is pending, it is resumed synchronously exactly
                once. With no pending backup, this method does nothing.

        Notes:
            ``flush`` is useful during controlled shutdown when waiting for the
            rest of the quiet interval is undesirable.
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
        """Resume the managed resource when this timer is still current.

        Args:
            generation (int):
                Generation captured when this timer was armed.

        Returns:
            None:
                Stale timers exit. The current timer resumes the managed binder.

        Notes:
            The managed binder is resumed with this debouncer marked as both the
            immediate source and an already-visited binder. The reciprocal
            debouncer edge is therefore bypassed and the managed binder can
            finally execute ``fire``.
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
        """Resume the managed binder while bypassing this interceptor.

        Returns:
            None:
                The managed binder continues through its internal invocation
                path and may execute its local ``fire`` operation.
        """

        self.__bindee._invoke(
            source=self,
            visited={id(self)},
        )

    @property
    def duration(self) -> timedelta:
        """Return the configured quiet period.

        Returns:
            timedelta:
                Debounce interval supplied during creation.
        """

        return self.__duration

    @property
    def since(self) -> datetime | None:
        """Return the newest timer-arm timestamp.

        Returns:
            datetime | None:
                Timezone-aware UTC timestamp, or ``None`` before first access.
        """

        with self.__lock:
            return self.__since

    @property
    def pending(self) -> bool:
        """Return whether a backup is pending.

        Returns:
            bool:
                ``True`` while a quiet-period timer is active.
        """

        with self.__lock:
            return self.__timer is not None

    @property
    def remaining(self) -> timedelta:
        """Return the approximate quiet time remaining.

        Returns:
            timedelta:
                Non-negative remaining time, or zero when no backup is pending.
        """

        with self.__lock:
            if self.__timer is None or self.__deadline is None:
                return timedelta(0)

            seconds = max(0.0, self.__deadline - monotonic())

        return timedelta(seconds=seconds)
