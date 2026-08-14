"""
Provide factory-constructed binder relationships with pre-fire interception.

This module defines the binding primitives used by :mod:`aura`.

A binder separates two concepts that are intentionally distinct:

* Invocation through ``binder()`` represents activity directed at the binder.
* :meth:`Binder.fire` performs the binder's local action.

Normally an invocation eventually executes ``fire``. Before that happens,
however, bound middleware is given an opportunity to intercept and consume the
invocation.

This distinction is required by middleware such as
:class:`aura.debounce.Debouncer`. A backup-capable resource may continue to be
used through its original public object::

    database()

while an attached debouncer observes that invocation before
``database.fire()`` executes. The debouncer can consume the invocation, reset
its quiet-period timer, and defer the backup until the resource has remained
inactive for the configured duration.

Binding model
-------------

Relationships are stored as directed graph edges.

:class:`OneWayBinder` creates exactly one new object. Given an existing binder
``B``::

    A = ConcreteOneWayBinder.create(B)

the factory installs::

    A -> B

:class:`TwoWayBinder` also creates exactly one new object. Given an existing
binder ``A``::

    B = ConcreteTwoWayBinder.create(A)

the factory installs the same one-way relationship in both directions::

    B -> A
    A -> B

This is equivalent to an undirected graph edge. The existing endpoint is
reused; reciprocal construction never creates a second artificial endpoint.

Invocation model
----------------

:meth:`Binder._invoke` processes an invocation in this order:

1. Mark the current binder as visited for the invocation.
2. Offer the invocation to eligible bound interceptors.
3. If any interceptor consumes the invocation, stop before ``fire``.
4. Otherwise execute :meth:`Binder.fire`.
5. Propagate the completed invocation to eligible bound neighbors.

An invocation-local set of object identities prevents reciprocal and longer
cyclic graphs from repeatedly visiting the same object.

Factory-only construction
-------------------------

Normal construction of :class:`OneWayBinder` and :class:`TwoWayBinder`
subclasses is rejected. Concrete objects must be created through ``create``::

    resource = Resource.create()
    middleware = Middleware.create(resource)

Python has no language-level private ``__init__``. Factory-only construction
is therefore enforced by guarding ``__new__``. The factory deliberately
bypasses that guard with :func:`object.__new__` and then invokes the concrete
initializer.

Example:
    Define a resource whose local action writes a backup::

        class Database(OneWayBinder):

            def fire(self) -> None:
                print('writing backup')

    Create and invoke it::

        database = Database.create()
        database()

    With no interceptor attached, ``Database.fire`` runs immediately.

Notes:
    Binding membership uses object identity rather than equality.

    A binder cannot bind to itself.

    Adding the same directed edge more than once is idempotent.

    Binding state is protected by :class:`threading.RLock`. Traversal uses an
    immutable snapshot so relationship changes cannot mutate the collection
    currently being traversed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import Any, Final, Self, final


class Binder(ABC):
    """Define shared binding, interception, firing, and propagation behavior.

    ``Binder`` is the abstract root of the relationship hierarchy.

    A binder owns zero or more outgoing relationships. Calling the binder
    begins a new invocation. Before the binder executes its local
    :meth:`fire` action, each eligible bound neighbor is offered an opportunity
    to intercept that invocation through :meth:`_intercept`.

    If at least one neighbor consumes the invocation, ``fire`` is skipped and
    ordinary post-fire propagation does not occur.

    Attributes:
        _bindee (Binder | None):
            Compatibility view of the first outgoing bindee. ``None`` means
            that no outgoing relationship currently exists.

        _bindees (tuple[Binder, ...]):
            Immutable snapshot of every outgoing bindee in insertion order.

    Notes:
        Interception and propagation are intentionally separate phases.
        Interception happens before local work. Propagation happens after local
        work.
    """

    def __init__(self) -> None:
        """Initialize an unbound binder.

        Returns:
            None:
                Constructors initialize an existing instance and do not return
                a separate object.
        """

        self.__binding_lock: Final[RLock] = RLock()
        self.__bindees: list[Binder] = []

    def __call__(self) -> None:
        """Begin a new public invocation at this binder.

        A fresh invocation-local visited set is created implicitly by
        :meth:`_invoke`.

        Returns:
            None:
                Invocation is processed entirely for side effects.
        """

        self._invoke()

    def _bind(self, bindee: Binder) -> None:
        """Install one directed relationship from this binder to ``bindee``.

        Args:
            bindee (Binder):
                Existing binder that should become an outgoing neighbor.

        Returns:
            None:
                The relationship is installed in place.

        Raises:
            ValueError:
                Raised when ``bindee`` is this binder itself.

        Notes:
            Adding the same object twice is idempotent. Identity comparison is
            used rather than ``__eq__``.
        """

        if bindee is self:
            raise ValueError('a binder cannot bind to itself')

        with self.__binding_lock:
            if any(candidate is bindee for candidate in self.__bindees):
                return

            self.__bindees.append(bindee)

    def _unbind(self, bindee: Binder) -> None:
        """Remove one directed relationship when present.

        Args:
            bindee (Binder):
                Existing outgoing neighbor to remove.

        Returns:
            None:
                Missing relationships are ignored.
        """

        with self.__binding_lock:
            self.__bindees = [
                candidate
                for candidate in self.__bindees
                if candidate is not bindee
            ]

    def _bind_two_way(self, bindee: Binder) -> None:
        """Connect two existing endpoints in both directions.

        Reciprocal binding is defined as two applications of the same
        directional primitive::

            self   -> bindee
            bindee -> self

        Args:
            bindee (Binder):
                Existing endpoint to connect reciprocally.

        Returns:
            None:
                Both directed relationships are installed.

        Raises:
            ValueError:
                Raised when ``bindee`` is ``self``.

        Notes:
            If installation of the reverse edge fails, the forward edge is
            removed before the exception is re-raised.
        """

        self._bind(bindee)

        try:
            bindee._bind(self)
        except BaseException:
            self._unbind(bindee)
            raise

    def _intercept(
        self,
        source: Binder,
        visited: set[int],
    ) -> bool:
        """Optionally consume ``source`` before ``source.fire()`` executes.

        The base implementation does not intercept anything.

        Middleware subclasses override this method when they need to observe
        and consume another binder's invocation before that binder performs its
        local action.

        Args:
            source (Binder):
                Binder whose invocation is about to execute local work.

            visited (set[int]):
                Invocation-local set containing identities that have already
                participated in traversal.

        Returns:
            bool:
                ``True`` when this binder consumed the invocation and
                ``source.fire()`` must not execute; otherwise ``False``.

        Notes:
            Interceptors should not recursively invoke ``source`` from this
            method. Deferred continuation should happen later through
            :meth:`_invoke`, identifying the middleware object as ``source`` so
            the reciprocal edge can be bypassed.
        """

        del source, visited
        return False

    def _invoke(
        self,
        source: Binder | None = None,
        visited: set[int] | None = None,
    ) -> None:
        """Process one invocation with interception before local firing.

        The processing sequence is:

        1. Create or reuse the invocation-local visited set.
        2. Stop if this binder has already participated.
        3. Mark this binder as visited.
        4. Snapshot the current outgoing relationships.
        5. Ask each eligible neighbor whether it intercepts this invocation.
        6. Stop before :meth:`fire` if any interceptor consumed the call.
        7. Execute :meth:`fire`.
        8. Propagate to each eligible downstream neighbor.

        Args:
            source (Binder | None, optional):
                Binder that directly forwarded this invocation. The source edge
                is skipped during interception and post-fire propagation.
                ``None`` means the current binder is the public origin.
                Defaults to ``None``.

            visited (set[int] | None, optional):
                Invocation-local set of binder identities that have already
                participated. ``None`` creates a new set. Defaults to ``None``.

        Returns:
            None:
                Processing is performed for side effects only.

        Example:
            For a backup resource reciprocally connected to a debouncer::

                resource <-> debouncer

            a public call behaves conceptually as::

                resource()
                    -> debouncer._intercept(resource, ...)
                    -> reset timer
                    -> return True
                    -> resource.fire() is skipped

            When the timer later expires, the debouncer resumes the resource
            with itself as ``source``. The reciprocal edge is skipped and
            ``resource.fire()`` is allowed to execute.

        Notes:
            All eligible interceptors are consulted even when an earlier
            interceptor consumes the invocation. This permits multiple
            middleware objects to observe one activity signal.

            If any interceptor consumes the invocation, ordinary post-fire
            propagation is skipped because the current binder never fired.
        """

        invocation: set[int] = set() if visited is None else visited
        identity: int = id(self)

        if identity in invocation:
            return

        invocation.add(identity)

        bindees: tuple[Binder, ...] = self._bindees
        consumed: bool = False

        for bindee in bindees:
            if bindee is source:
                continue

            if id(bindee) in invocation:
                continue

            if bindee._intercept(self, invocation):
                consumed = True

        if consumed:
            return

        self.fire()

        for bindee in bindees:
            if bindee is source:
                continue

            if id(bindee) in invocation:
                continue

            bindee._invoke(self, invocation)

    @property
    def _bindee(self) -> Binder | None:
        """Return the first outgoing bindee.

        Returns:
            Binder | None:
                First current outgoing neighbor, or ``None`` when this binder
                has no outgoing relationships.

        Notes:
            A binder may have multiple neighbors. Use :attr:`_bindees` when the
            complete relationship set is required.
        """

        bindees = self._bindees
        return bindees[0] if bindees else None

    @property
    def _bindees(self) -> tuple[Binder, ...]:
        """Return an immutable snapshot of all outgoing bindees.

        Returns:
            tuple[Binder, ...]:
                Current outgoing neighbors in insertion order.
        """

        with self.__binding_lock:
            return tuple(self.__bindees)

    @abstractmethod
    def fire(self) -> None:
        """Perform this binder's local action.

        Public invocation and local execution are deliberately separate.
        Interceptors may prevent this method from running immediately.

        Concrete resource classes place their actual work here. For a backup
        resource, this method should perform the backup.

        Returns:
            None:
                Local work is performed for side effects.
        """

        ...


class OneWayBinder(Binder):
    """Create one factory-only endpoint with an optional outgoing relationship.

    ``OneWayBinder.create`` creates exactly one new object.

    Given an existing binder ``B``::

        A = ConcreteOneWayBinder.create(B)

    the relationship becomes::

        A -> B

    ``B`` is reused and is not modified to point back to ``A``.

    Construction is factory-only. Normal concrete class construction raises
    :class:`TypeError`.
    """

    @final
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Reject normal direct construction.

        Args:
            *args (Any):
                Positional arguments supplied to an attempted direct class
                construction.

            **kwargs (Any):
                Keyword arguments supplied to an attempted direct class
                construction.

        Returns:
            Self:
                This method never returns.

        Raises:
            TypeError:
                Always raised so callers use :meth:`create`.
        """

        del args, kwargs

        raise TypeError(
            f'{cls.__name__} cannot be constructed directly; '
            f'use {cls.__name__}.create(...) instead.'
        )

    def __init__(self, bindee: Binder | None = None) -> None:
        """Initialize one factory-allocated directional endpoint.

        Args:
            bindee (Binder | None, optional):
                Existing downstream binder. ``None`` leaves the new endpoint
                without outgoing relationships. Defaults to ``None``.

        Returns:
            None:
                The already allocated endpoint is initialized.

        Notes:
            Concrete subclasses may append constructor parameters after
            ``bindee`` and should forward ``bindee`` to this initializer.
        """

        super().__init__()

        if bindee is not None:
            self._bind(bindee)

    @classmethod
    def create(
        cls,
        bindee: Binder | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Allocate and initialize exactly one concrete one-way binder.

        Args:
            bindee (Binder | None, optional):
                Existing downstream endpoint. Defaults to ``None``.

            *args (Any):
                Additional positional arguments forwarded to the concrete
                ``__init__`` after ``bindee``.

            **kwargs (Any):
                Additional keyword arguments forwarded unchanged to the
                concrete ``__init__``.

        Returns:
            Self:
                Fully initialized concrete instance.

        Raises:
            TypeError:
                May be raised when subclass-specific arguments do not match the
                concrete initializer.

        Notes:
            :func:`object.__new__` intentionally bypasses the guarded public
            allocation path. Exactly one new binder object is allocated.
        """

        instance: Self = object.__new__(cls)
        cls.__init__(instance, bindee, *args, **kwargs)
        return instance


class TwoWayBinder(OneWayBinder):
    """Create one endpoint reciprocally connected to an existing endpoint.

    ``TwoWayBinder`` does not allocate two new objects.

    Given an existing endpoint ``A``::

        B = ConcreteTwoWayBinder.create(A)

    only ``B`` is newly allocated. The graph becomes::

        B -> A
        A -> B

    This is equivalent to the undirected relationship::

        A <-> B

    The design is particularly useful for attached middleware because callers
    can continue invoking the original endpoint while the middleware remains
    reachable through the reverse edge.
    """

    @classmethod
    def create(
        cls,
        bindee: Binder | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Allocate one endpoint and connect it to ``bindee`` both ways.

        Args:
            bindee (Binder | None, optional):
                Existing opposite endpoint. Reciprocal construction requires a
                non-``None`` bindee.

            *args (Any):
                Additional positional arguments forwarded to the concrete
                initializer.

            **kwargs (Any):
                Additional keyword arguments forwarded unchanged to the
                concrete initializer.

        Returns:
            Self:
                The one newly allocated endpoint.

        Raises:
            ValueError:
                Raised when no existing bindee is supplied.

            TypeError:
                May be raised when subclass-specific constructor arguments are
                incompatible with the concrete initializer.

        Notes:
            The concrete initializer installs::

                new -> bindee

            The factory then installs the reverse relationship::

                bindee -> new

            No second new endpoint is created.
        """

        if bindee is None:
            raise ValueError(
                f'{cls.__name__}.create(...) requires an existing bindee'
            )

        instance: Self = object.__new__(cls)
        cls.__init__(instance, bindee, *args, **kwargs)

        try:
            bindee._bind(instance)
        except BaseException:
            instance._unbind(bindee)
            raise

        return instance
