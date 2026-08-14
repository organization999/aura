'''
Provide a simple callback debouncer based on object lifetime.

This module defines :class:`Debouncer`, a small utility that delays execution
of a callback until at least a configured duration has elapsed since the
debouncer instance was created.

The callback is executed from :meth:`Debouncer.__del__`, meaning execution is
associated with destruction of the ``Debouncer`` instance.

Example:
    Create a debouncer whose callback prints ``'done'`` after at least five
    seconds have elapsed::

        context = Debouncer(lambda: print('done'))

Notes:
    ``Debouncer`` performs its delay synchronously by calling
    :func:`time.sleep`. Consequently, whichever thread executes ``__del__``
    may be blocked while waiting for the configured duration to expire.

    Python does not guarantee precisely when ``__del__`` will execute.
    Destruction generally occurs when an object's reference count reaches
    zero in CPython, but callers should not depend on deterministic destructor
    execution across Python implementations.

    Exceptions raised by the callback occur while ``__del__`` is executing.
    Python handles exceptions raised from destructors specially and typically
    reports them to ``sys.stderr`` rather than propagating them normally.
'''
from __future__ import annotations

from datetime  import datetime, timedelta, timezone
from threading import Lock
from time      import sleep
from typing    import Callable, Final

class Debouncer:
    '''
    Delay callback execution until a minimum lifetime has elapsed.

    A ``Debouncer`` records its creation time and a desired debounce duration.
    When the instance is destroyed, it determines how much of that duration
    remains. If necessary, destruction blocks for the remaining amount of
    time before invoking the configured callback.

    This implementation therefore guarantees, subject to normal system clock
    and scheduling behavior, that the callback is not invoked by ``__del__``
    earlier than ``duration_seconds`` after construction.

    Attributes:
        __callback (Callable):
            Function or callable object invoked when the debouncer is
            destroyed.

        __duration (timedelta):
            Minimum amount of time that should elapse between construction
            and callback invocation.

        __since (datetime):
            UTC timestamp recorded when the debouncer was constructed.

    Example:
        Construct a debouncer using the default five-second duration::

            debouncer = Debouncer(lambda: print('done'))

        Construct one using a custom duration::

            debouncer = Debouncer(
                lambda: print('done'),
                duration_seconds=2.5,
            )

    Warning:
        Callback execution depends on ``__del__``. Python does not provide a
        general guarantee that destructors execute at a particular instant,
        especially during interpreter shutdown or when reference cycles are
        involved.

        This class also blocks inside ``__del__`` by calling
        :func:`time.sleep`. That may be undesirable when deterministic,
        asynchronous, or non-blocking scheduling is required.
    '''

    def __init__(self, callback         : Callable,
                       duration_seconds : float) -> None:
        '''
        Initialize a debouncer.

        The current UTC timestamp is captured immediately during
        initialization and becomes the starting point from which the debounce
        duration is measured.

        Args:
            callback (Callable):
                Callable to execute when this object is destroyed after the
                configured debounce duration has elapsed.

                The callable is invoked without arguments.

            duration_seconds (float, optional):
                Minimum number of seconds that should elapse between creation
                of this object and execution of ``callback``.

                The value is converted to :class:`datetime.timedelta`.
                Defaults to ``5.0``.

        Returns:
            None:
                Constructors initialize instances and do not return a value.

        Notes:
            No validation is performed on ``callback`` beyond its type
            annotation. Supplying an object that cannot actually be called
            will cause an error when ``__del__`` attempts to invoke it.

            Likewise, this constructor does not reject zero or negative
            durations. A non-positive duration effectively causes the
            destructor to invoke the callback without sleeping.

        Example:
            Create a debouncer with a one-second duration::

                debouncer = Debouncer(
                    lambda: print('complete'),
                    duration_seconds=1.0,
                )
        '''

        # Store the callable that will be executed when the debouncer is
        # destroyed.
        self.__callback: Final[Callable] = callback

        # Convert the caller-provided duration from seconds into a timedelta
        # so it can be directly compared with elapsed datetime durations.
        self.__duration: Final[timedelta] = timedelta(seconds=duration_seconds)

        # A mechanism for protecting the __since member from race conditions,
        # since __since can be updated during the Debounder runtime.
        self.__lock: Lock = Lock()

        # Record the creation timestamp using an explicitly UTC-aware
        # datetime. This avoids mixing naive and timezone-aware timestamps.
        self.__since: datetime = datetime.now(timezone.utc)

    def __del__(self) -> None:
        '''
        Invoke the callback after the configured duration has elapsed.

        Destruction calculates how much time has passed since construction.
        If less than :attr:`duration` has elapsed, the current thread sleeps
        for the remaining interval. The callback is then invoked exactly once
        by this destructor invocation.

        The calculation is conceptually::

            elapsed = current_time - creation_time
            remaining = duration - elapsed

        If ``remaining`` is positive, :func:`time.sleep` is called with the
        remaining number of seconds.

        Returns:
            None:
                Destructors do not return a value.

        Notes:
            If the configured duration has already elapsed by the time
            destruction occurs, no sleep is performed.

            ``time.sleep`` blocks the thread responsible for executing this
            destructor.

            The exact time at which Python invokes ``__del__`` is not
            guaranteed by the language specification. Therefore this method
            should not be considered a general-purpose deterministic timer.

        Warning:
            Exceptions raised by ``self.__callback()`` cannot normally
            propagate to code that caused object destruction. Exceptions from
            ``__del__`` are handled specially by the Python runtime.
        '''

        while True:

            with self.__lock:
                since: datetime = self.__since

            # Determine how long the debouncer has been active since its most
            # recent reset.
            elapsed: timedelta = (
                datetime.now(timezone.utc) - since
            )

            # Compute the remaining duration.
            remaining_duration: timedelta = (self.__duration - elapsed)

            # Determine how much of the debounce interval remains.
            remaining_seconds: float = remaining_duration.total_seconds()

            # The debounce interval has expired.
            if remaining_seconds <= 0:
                break

            # Sleep for the remaining seconds and then check again.
            sleep(remaining_seconds)

        # Execute the callback after the minimum lifetime has been satisfied.
        self.__callback()

    def reset(self) -> None:
        with self.__lock:
            self.__since += self.__duration

    @property
    def duration(self) -> timedelta:
        '''
        Return the configured debounce duration.

        Returns:
            timedelta:
                Minimum amount of time between construction of the debouncer
                and callback execution by its destructor.

        Example:
            Inspect the configured duration::

                debouncer = Debouncer(lambda: None, 2.0)
                print(debouncer.duration.total_seconds())

            Output::

                2.0
        '''

        return self.__duration

    @property
    def since(self) -> datetime:
        '''
        Return the UTC timestamp at which the debouncer was created.

        Returns:
            datetime:
                Timezone-aware UTC :class:`datetime.datetime` captured during
                construction.

        Notes:
            The returned value uses :data:`datetime.timezone.utc` and is
            therefore timezone-aware.

        Example:
            Inspect the construction timestamp::

                debouncer = Debouncer(lambda: None)
                print(debouncer.since)
        '''

        with self.__lock:
            return self.__since

def main() -> None:

    # context: Debouncer = Debouncer(lambda: print('done'), duration_seconds=1.0)
    #
    # for _ in range(5):
    #     context.reset()
    #     sleep(1)

if __name__ == '__main__':
    main()
