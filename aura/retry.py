"""
Provide transparent retry middleware for bound binder operations.

This module defines :class:`RetryPolicy`, a reciprocal binder middleware
object that retries the operation of an existing :class:`aura.binding.Binder`
when that operation raises an exception.

The policy uses the same pre-fire interception mechanism as
:class:`aura.debounce.Debouncer`.

Callers continue invoking the original managed object::

    operation()

They do not replace the original object with the policy and do not need to
call the policy directly.

Relationship
------------

``RetryPolicy`` derives from :class:`aura.binding.TwoWayBinder`. Creating a
policy reuses the existing managed object and installs both graph edges::

    managed -> retry_policy
    retry_policy -> managed

The ``managed -> retry_policy`` edge lets the policy intercept the managed
invocation before ``managed.fire()`` executes.

The policy consumes that original invocation and resumes the managed binder
itself. Because the retry policy is supplied as the invocation ``source``,
the reciprocal edge back into the same policy is skipped while an attempt is
executing.

Retry behavior
--------------

``retry_attempts`` specifies the number of retries permitted *after* the
initial execution attempt.

Therefore::

    retry_attempts = 0

permits one total execution, while::

    retry_attempts = 3

permits at most four total executions:

1. Initial attempt.
2. Retry 1.
3. Retry 2.
4. Retry 3.

Each attempt resumes the managed binder through
:meth:`aura.binding.Binder._invoke`, so the managed object's normal
:meth:`aura.binding.Binder.fire` implementation executes.

If an attempt succeeds, the policy returns normally and the original public
invocation is considered complete.

If an attempt raises an :class:`Exception`, the policy catches that failure
and starts the next permitted attempt.

If the final permitted attempt also fails, the policy re-raises that final
exception. The caller therefore receives the actual failure from the managed
operation rather than an unrelated wrapper exception.

Example:
    Define an operation that fails twice before succeeding::

        class UnstableOperation(OneWayBinder):

            def __init__(
                self,
                bindee: Binder | None = None,
            ) -> None:
                super().__init__(bindee)
                self.attempt = 0

            def fire(self) -> None:
                self.attempt += 1

                if self.attempt < 3:
                    raise RuntimeError('temporary failure')

                print('success')

    Attach a policy allowing two retries::

        operation = UnstableOperation.create()
        RetryPolicy.create(operation, 2)

    Call the original operation normally::

        operation()

    The invocation performs three executions of ``fire``. The first two
    failures are caught by the policy and the third attempt succeeds.

Failure example:
    If the managed operation always fails::

        operation = AlwaysFails.create()
        RetryPolicy.create(operation, 2)

        operation()

    then ``fire`` is attempted three times total. The exception from the
    third attempt is re-raised to the caller.

Notes:
    A retryable failure is represented by an exception derived from
    :class:`Exception`.

    Process-control exceptions such as :class:`KeyboardInterrupt`,
    :class:`SystemExit`, and :class:`GeneratorExit` derive directly from
    :class:`BaseException` and are intentionally not retried.

    Retry attempts are synchronous. The caller remains inside the original
    invocation until one attempt succeeds or all configured retries are
    exhausted.

    This implementation does not impose a retry delay or backoff strategy.
    Each retry begins immediately after the preceding attempt fails.
"""

from __future__ import annotations

from typing import Final

from .binding import Binder, TwoWayBinder


class RetryPolicy(TwoWayBinder):
    """Retry a managed binder operation when its invocation raises.

    ``RetryPolicy`` transparently intercepts calls to one existing binder.
    The application continues invoking that original object.

    On each intercepted invocation, the policy executes the managed binder
    once and catches any :class:`Exception` raised by that attempt. Failed
    executions are retried until either:

    * one attempt succeeds; or
    * all configured retry attempts have been exhausted.

    When every permitted attempt fails, the exception raised by the final
    attempt is propagated back to the original caller.

    Attributes:
        retry_attempts (int):
            Number of retry executions permitted after the initial attempt.

        max_attempts (int):
            Maximum total number of executions for one public invocation.
            This value is always ``retry_attempts + 1``.

    Example:
        Attach three retries to an existing operation::

            operation = Operation.create()
            RetryPolicy.create(operation, 3)

        Continue using the original object::

            operation()

        ``Operation.fire`` may execute at most four times for that call.
    """

    def __init__(
        self,
        bindee: Binder | None = None,
        retry_attempts: int = 3,
    ) -> None:
        """Initialize one factory-allocated retry policy.

        Args:
            bindee (Binder | None, optional):
                Existing binder whose operation should be retried when it
                raises. :meth:`TwoWayBinder.create` requires this object to
                be non-``None``.

            retry_attempts (int, optional):
                Number of additional executions permitted after the initial
                attempt. ``0`` disables retries while retaining transparent
                interception. Defaults to ``3``.

        Returns:
            None:
                The already allocated policy is initialized.

        Raises:
            ValueError:
                Raised when ``bindee`` is ``None`` or when
                ``retry_attempts`` is negative.

        Notes:
            ``retry_attempts`` counts only retries. The initial execution is
            always attempted once, so the maximum total execution count is::

                retry_attempts + 1
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

    def _intercept(
        self,
        source: Binder,
        visited: set[int],
    ) -> bool:
        """Intercept and retry an invocation of the managed binder.

        Args:
            source (Binder):
                Binder whose local :meth:`Binder.fire` operation is about to
                execute.

            visited (set[int]):
                Invocation-local identities already visited by the original
                public invocation.

        Returns:
            bool:
                ``True`` when ``source`` is the managed bindee. Returning
                ``True`` consumes the original invocation because the retry
                policy has already executed the managed operation itself.

                ``False`` is returned for unrelated binders.

        Raises:
            Exception:
                Re-raises the exception from the final permitted execution
                when every attempt fails.

        Notes:
            The incoming ``visited`` set is intentionally not reused for
            retry executions. It already contains the managed bindee's
            identity because interception occurs after the source binder has
            been marked visited.

            Every retry therefore begins with a fresh invocation-local set
            containing only this policy's identity. The managed bindee can
            then participate again while the reciprocal edge back to this
            retry policy remains bypassed.
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
        """Execute the retry policy when invoked directly.

        Normal application code should invoke the managed binder rather than
        this policy object. Direct policy invocation is nevertheless defined
        consistently and performs the same retry operation.

        Args:
            source (Binder | None, optional):
                Binder that forwarded the invocation, if any.

            visited (set[int] | None, optional):
                Invocation-local visited state, if any.

        Returns:
            None:
                Returns normally when one managed execution succeeds.

        Raises:
            Exception:
                Re-raises the final managed exception after retry exhaustion.
        """

        del source, visited
        self.fire()

    def fire(self) -> None:
        """Execute the managed binder under the configured retry policy.

        Returns:
            None:
                Returns normally as soon as one attempt succeeds.

        Raises:
            Exception:
                Re-raises the final exception when the initial attempt and
                every configured retry attempt fail.
        """

        self.__execute()

    def __execute(self) -> None:
        """Run the managed invocation until success or retry exhaustion.

        Each execution resumes the managed binder with this policy supplied
        as the immediate ``source`` and as an already visited binder. This
        prevents the managed binder from immediately re-entering the same
        retry policy through the reciprocal relationship.

        Returns:
            None:
                Returns immediately when an attempt completes successfully.

        Raises:
            Exception:
                Re-raises the exception from the final permitted attempt.

        Notes:
            The exception is re-raised from the final ``except`` block with
            a bare ``raise``. This preserves the final failure's exception
            type, value, and traceback.

            Every attempt receives a fresh visited set. Reusing the outer
            invocation's set would incorrectly suppress the managed binder,
            because that set already contains its identity.
        """

        max_attempts: int = self.max_attempts

        for attempt in range(1, max_attempts + 1):
            try:
                self.__bindee._invoke(
                    source=self,
                    visited={id(self)},
                )
            except Exception:
                if attempt >= max_attempts:
                    raise

                continue

            return

    @property
    def retry_attempts(self) -> int:
        """Return the configured number of retries.

        Returns:
            int:
                Number of additional executions allowed after the initial
                attempt.
        """

        return self.__retry_attempts

    @property
    def max_attempts(self) -> int:
        """Return the maximum total executions for one invocation.

        Returns:
            int:
                Initial execution plus all configured retries.
        """

        return self.__retry_attempts + 1
