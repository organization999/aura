"""
Provide a resettable binder debouncer.

This module defines :class:`Debouncer`, a binder that delays propagation to an
existing bindee until a configured quiet interval has elapsed.

A debouncer is connected reciprocally to its bindee. The connection is
implemented using the graph semantics provided by :class:`TwoWayBinder`:

    debouncer -> bindee
    bindee    -> debouncer

The reverse edge allows an invocation that begins at the bindee to reach the
debouncer and reset its pending timer. The debouncer itself suppresses normal
immediate propagation. Instead, every invocation that reaches the debouncer
restarts a :class:`threading.Timer`.

When the newest timer expires, the debouncer invokes the bindee exactly once
for that debounce generation.

Example:
    Given an existing binder::

        target = Printer.create(None, 'target')

    create a half-second debouncer::

        debouncer = Debouncer.create(target, 0.5)

    Calling the debouncer repeatedly keeps postponing the target invocation::

        debouncer()
        debouncer()
        debouncer()

    The target is invoked once after 0.5 seconds pass without another
    debouncer reset.

    Because the relationship is reciprocal, an invocation beginning at
    ``target`` before expiration also reaches the debouncer and resets the
    pending timer.

Important:
    Calling the bindee directly still performs the bindee's own ``fire`` action
    immediately. The debouncer can observe that invocation through the reverse
    edge and reset its pending timer, but it cannot retroactively suppress work
    that a caller deliberately started at the bindee itself.

Notes:
    Timer calculations use :func:`time.monotonic`, so system wall-clock
    adjustments do not shorten or extend the debounce interval.

    :class:`threading.Timer` executes expiration on a background thread. Timer
    threads created here are daemon threads, so a pending debounce does not
    prevent interpreter shutdown.

    Reset uses a monotonically increasing generation counter in addition to
    :meth:`threading.Timer.cancel`. This prevents an older timer callback that
    races with a reset from invoking the bindee after it has become stale.

    ``Debouncer`` does not use ``__del__``. Python destructor timing is not
    deterministic enough to implement debounce semantics reliably, and
    blocking inside a destructor would delay whichever thread performs object
    finalization.
"""

from __future__ import annotations

from datetime  import datetime, timedelta, timezone
from threading import Lock, Timer
from time      import monotonic
from typing    import Final

from binding import Binder, TwoWayBinder

class Debouncer(TwoWayBinder):
    """Delay bindee invocation until a quiet interval has elapsed.

    ``Debouncer`` starts one pending debounce interval during construction.
    Every invocation that subsequently reaches the debouncer resets that
    interval to the full configured duration.

    The timer expires only when no newer reset supersedes it. Expiration then
    invokes the existing bindee once.

    Because :class:`Debouncer` derives from :class:`TwoWayBinder`, creation
    installs both relationships::

        debouncer -> bindee
        bindee    -> debouncer

    The reverse relationship is significant. If the bindee participates in an
    invocation before the debounce interval expires, propagation reaches the
    debouncer and restarts the timer.

    Attributes:
        duration (timedelta):
            Configured quiet interval.

        since (datetime):
            UTC wall-clock timestamp of the most recent timer arm or reset.

        pending (bool):
            Whether a current debounce generation is waiting to expire.

        remaining (timedelta):
            Approximate monotonic time remaining for the current generation.

    Example:
        Create a debouncer with a 250 ms interval::

            debouncer = Debouncer.create(target, 0.25)

        Reset it by invoking the debouncer::

            debouncer()

        If another call reaches it before 250 ms pass, the interval starts over.

    Warning:
        Directly invoking the bindee executes the bindee's own action before
        graph propagation reaches this debouncer. Such a direct call therefore
        resets the pending debounce interval but is not itself suppressed.
        Call the debouncer rather than the bindee when the operation itself
        must be delayed.
    """

    def __init__(
        self,
        bindee: Binder | None = None,
        duration_seconds: float = 5.0,
    ) -> None:
        """Initialize one factory-allocated debouncer.

        Args:
            bindee (Binder | None, optional):
                Existing binder whose invocation should be delayed. The
                :meth:`TwoWayBinder.create` factory requires this argument to be
                non-``None``. The optional annotation is retained so this
                initializer remains compatible with the binder hierarchy.

            duration_seconds (float, optional):
                Quiet interval in seconds. Each invocation that reaches the
                debouncer restarts the full interval. Defaults to ``5.0``.

        Returns:
            None:
                The already allocated debouncer is initialized.

        Raises:
            ValueError:
                Raised when ``bindee`` is ``None`` or when
                ``duration_seconds`` is negative.

        Notes:
            Construction starts the first pending debounce generation
            immediately. Consequently, the bindee is invoked once after the
            configured interval unless that timer is reset or cancelled.

            Zero is permitted. A zero-duration timer schedules expiration as
            soon as the timer thread can run.
        """

        if bindee is None:
            raise ValueError('Debouncer requires an existing bindee')

        if duration_seconds < 0.0:
            raise ValueError('duration_seconds must be greater than or equal to 0')

        # TwoWayBinder.__init__ is inherited from OneWayBinder and installs the
        # forward edge. TwoWayBinder.create installs the reverse edge after this
        # initializer completes.
        super().__init__(bindee)

        self.__bindee: Final[Binder] = bindee
        self.__duration_seconds: Final[float] = float(duration_seconds)
        self.__duration: Final[timedelta] = timedelta(
            seconds=self.__duration_seconds
        )

        self.__lock: Final[Lock] = Lock()
        self.__timer: Timer | None = None
        self.__generation: int = 0
        self.__since: datetime = datetime.now(timezone.utc)
        self.__deadline: float = monotonic()

        self.reset()

    def _invoke(
        self,
        source: Binder | None = None,
        visited: set[int] | None = None,
    ) -> None:
        """Consume an invocation and restart the debounce interval.

        Unlike a normal binder, a debouncer does not immediately propagate the
        current invocation to its bindee. Reaching the debouncer means only
        that the timer should be restarted.

        Args:
            source (Binder | None, optional):
                Binder that forwarded the invocation. It is intentionally not
                used for immediate propagation because propagation is delayed
                until expiration.

            visited (set[int] | None, optional):
                Invocation-local visited set supplied by the graph traversal.
                It is accepted to preserve the :class:`Binder` propagation
                contract. No downstream traversal occurs at this time.

        Returns:
            None:
                The pending timer is restarted.

        Notes:
            The parameters are deliberately consumed even though the debouncer
            does not forward immediately. This method is invoked both by a
            direct ``debouncer()`` call and when the reciprocal bindee forwards
            an invocation into the debouncer.
        """

        del source, visited
        self.fire()

    def fire(self) -> None:
        """Restart the debounce timer.

        ``fire`` is the local binder action for a debouncer. Every invocation
        that reaches this binder performs the same operation: the current timer
        is invalidated and a new full-duration timer is started.

        Returns:
            None:
                Timer state is updated for side effects only.
        """

        self.reset()

    def reset(self) -> None:
        """Restart the debounce interval from the current instant.

        Any currently pending timer is cancelled. A new generation number,
        UTC reset timestamp, monotonic deadline, and daemon
        :class:`threading.Timer` are then installed.

        Returns:
            None:
                The pending debounce generation is replaced.

        Notes:
            :meth:`Timer.cancel` alone is insufficient to rule out a race in
            which an old timer callback has already begun executing. The
            generation number is therefore checked again inside the expiration
            callback before the bindee can be invoked.
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
            timer.daemon = True

            self.__timer = timer
            timer.start()

    def cancel(self) -> None:
        """Cancel the currently pending debounce generation.

        Cancellation invalidates the current generation so a racing stale timer
        callback cannot invoke the bindee.

        Returns:
            None:
                Pending timer state is cleared.

        Notes:
            Cancellation does not remove graph bindings between the debouncer
            and its bindee. A later call to :meth:`reset`, :meth:`fire`, or the
            debouncer itself starts a new generation.
        """

        with self.__lock:
            self.__generation += 1

            timer = self.__timer
            self.__timer = None
            self.__deadline = monotonic()

            if timer is not None:
                timer.cancel()

    def __expire(self, generation: int) -> None:
        """Invoke the bindee if ``generation`` is still the newest timer.

        Args:
            generation (int):
                Generation captured when the timer was armed.

        Returns:
            None:
                A current generation invokes the bindee once; a stale generation
                exits without side effects.

        Notes:
            Bindee invocation occurs after releasing the timer-state lock so
            arbitrary binder work cannot deadlock reset or cancellation.

            The forwarded graph invocation marks this debouncer as already
            visited and uses it as the source. Consequently, the reciprocal
            edge from the bindee back to this debouncer does not immediately
            arm a new timer when expiration itself invokes the bindee.
        """

        with self.__lock:
            if generation != self.__generation:
                return

            self.__timer = None

        self.__bindee._invoke(
            source=self,
            visited={id(self)},
        )

    @property
    def duration(self) -> timedelta:
        """Return the configured quiet interval.

        Returns:
            timedelta:
                Debounce duration supplied during construction.
        """

        return self.__duration

    @property
    def since(self) -> datetime:
        """Return the UTC timestamp of the most recent timer reset.

        Returns:
            datetime:
                Timezone-aware UTC timestamp corresponding to the newest
                debounce generation.

        Notes:
            Unlike the previous lifetime-based implementation, ``since`` is
            updated on every reset rather than remaining fixed at construction.
        """

        with self.__lock:
            return self.__since

    @property
    def pending(self) -> bool:
        """Return whether a debounce generation is currently pending.

        Returns:
            bool:
                ``True`` when a timer is waiting to expire; otherwise ``False``.
        """

        with self.__lock:
            return self.__timer is not None

    @property
    def remaining(self) -> timedelta:
        """Return the approximate monotonic time remaining.

        Returns:
            timedelta:
                Non-negative remaining time for the current generation.
                ``timedelta(0)`` is returned when no timer is pending.

        Notes:
            The value is observational. The timer may expire immediately after
            this property releases its lock.
        """

        with self.__lock:
            if self.__timer is None:
                return timedelta(0)

            seconds = max(0.0, self.__deadline - monotonic())

        return timedelta(seconds=seconds)
