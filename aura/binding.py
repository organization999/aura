"""
Provide factory-constructed one-way and reciprocal binder relationships.

This module defines a small graph-oriented invocation abstraction built around
:class:`Binder`. A binder performs a local action through :meth:`Binder.fire`
and may propagate the same invocation to one or more bound neighbors.

Bindings are represented internally as directed edges:

    A -> B

A :class:`OneWayBinder` creates exactly one new binder object and, when a
``bindee`` is supplied, installs one directed edge from that new object to the
existing bindee.

A :class:`TwoWayBinder` is a specialization of :class:`OneWayBinder`. It also
creates exactly one new binder object, but reciprocal binding is implemented
by applying the one-way operation in both directions:

    A -> B
    B -> A

Conceptually this is the same operation used to connect two vertices in an
undirected graph. The existing bindee is reused; a second artificial endpoint
is not allocated merely to establish reciprocity.

Both public binder families use factory-only construction. Normal construction
syntax is rejected:

    ConcreteBinder(...)

and callers must instead use the corresponding ``create`` factory:

    ConcreteBinder.create(bindee, ...)

The factory allocates the concrete instance with :func:`object.__new__`, then
explicitly invokes the concrete class's initializer. Subclass-specific
constructor state may follow ``bindee`` as positional or keyword arguments.

Invocation propagation carries a set of visited object identities. This makes
the propagation algorithm safe for reciprocal edges and for longer cyclic
graphs, rather than merely suppressing an immediate two-node bounce.

Example:
    Define a concrete one-way binder::

        class Printer(OneWayBinder):

            def __init__(
                self,
                bindee: Binder | None = None,
                name: str = '',
            ) -> None:
                super().__init__(bindee)
                self.name = name

            def fire(self) -> None:
                print(self.name)

    Create a terminal endpoint and an upstream endpoint::

        downstream = Printer.create(None, 'downstream')
        upstream = Printer.create(downstream, 'upstream')

        upstream()

    Output::

        upstream
        downstream

    A reciprocal endpoint reuses an existing binder instead of creating a
    second endpoint::

        class ReciprocalPrinter(TwoWayBinder):

            def __init__(
                self,
                bindee: Binder | None = None,
                name: str = '',
            ) -> None:
                super().__init__(bindee)
                self.name = name

            def fire(self) -> None:
                print(self.name)

        lhs = Printer.create(None, 'lhs')
        rhs = ReciprocalPrinter.create(lhs, 'rhs')

    The relationship is now equivalent to::

        lhs -> rhs
        rhs -> lhs

Notes:
    Python does not provide a language-level private constructor. Factory-only
    construction is therefore enforced by guarding ``__new__``. A caller that
    deliberately invokes ``object.__new__(ConcreteBinder)`` can bypass normal
    Python construction policy; this module prevents ordinary direct class
    construction.

    Bindings use object identity rather than equality. A binder cannot bind to
    itself, and adding the same directed edge more than once is idempotent.

    The binding collection is protected by a re-entrant lock. Invocation uses
    a snapshot of the current neighbors so binding changes do not mutate a
    collection while it is being traversed.
"""

from __future__ import annotations

from abc       import ABC, abstractmethod
from threading import RLock
from typing    import Any, Final, Self, final

class Binder(ABC):
    """Define common binder state, graph binding, and invocation propagation.

    ``Binder`` is the abstract root of the binding hierarchy. Each binder owns
    zero or more directed edges to other binders. When invoked, the binder:

    1. Executes its local :meth:`fire` implementation.
    2. Marks itself as visited for the current invocation.
    3. Propagates to each bound neighbor that has not already participated in
       the same invocation.

    The visited set makes invocation safe for reciprocal and cyclic graphs.

    Attributes:
        _bindee (Binder | None):
            Compatibility view of the first bound neighbor. ``None`` is
            returned when no neighbor exists. New code that needs every
            neighbor should use :attr:`_bindees`.

        _bindees (tuple[Binder, ...]):
            Snapshot of all current outgoing binding targets.

    Notes:
        Bindings are maintained as a collection rather than a single mutable
        pointer. This allows reciprocal relationships to behave like undirected
        graph edges and permits an existing binder to participate in more than
        one relationship without discarding prior edges.
    """

    def __init__(self) -> None:
        """Initialize an unbound binder.

        Returns:
            None:
                Constructors initialize an existing instance and do not return
                a separate value.

        Notes:
            Concrete relationship classes establish their edges after this
            common state exists. Callers do not normally invoke this initializer
            directly because :class:`OneWayBinder` and :class:`TwoWayBinder`
            enforce factory-only construction.
        """

        self.__binding_lock: Final[RLock] = RLock()
        self.__bindees: list[Binder] = []

    def __call__(self) -> None:
        """Begin a new invocation at this binder.

        A fresh visited set is created for every public call. The current
        binder then fires and the invocation is propagated through the bound
        graph.

        Returns:
            None:
                Invocation is performed for side effects only.
        """

        self._invoke()

    def _bind(self, bindee: Binder) -> None:
        """Install one directed binding edge from this binder to ``bindee``.

        The operation is the primitive used by both relationship types.
        :class:`OneWayBinder` applies it once, while :class:`TwoWayBinder`
        applies it once in each direction.

        Args:
            bindee (Binder):
                Existing binder that should receive propagated invocations from
                this binder.

        Returns:
            None:
                The graph relationship is modified in place.

        Raises:
            ValueError:
                Raised when ``bindee`` is the current binder itself.

        Notes:
            Adding the same object more than once is idempotent. Object identity
            is used for duplicate detection rather than ``__eq__``.
        """

        if bindee is self:
            raise ValueError('a binder cannot bind to itself')

        with self.__binding_lock:
            if any(candidate is bindee for candidate in self.__bindees):
                return

            self.__bindees.append(bindee)

    def _unbind(self, bindee: Binder) -> None:
        """Remove one directed binding edge when it exists.

        Args:
            bindee (Binder):
                Existing binder whose outgoing relationship from ``self`` should
                be removed.

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
        """Connect this binder and ``bindee`` with two directed edges.

        Reciprocal binding is intentionally defined in terms of the same
        one-way primitive used everywhere else:

            self   -> bindee
            bindee -> self

        No additional binder objects are allocated.

        Args:
            bindee (Binder):
                Existing endpoint to connect reciprocally to ``self``.

        Returns:
            None:
                Both directed relationships are installed in place.

        Raises:
            ValueError:
                Raised when ``bindee`` is ``self``.
        """

        self._bind(bindee)

        try:
            bindee._bind(self)
        except BaseException:
            # Preserve all-or-nothing behavior if a derived binder rejects the
            # reverse relationship for any reason.
            self._unbind(bindee)
            raise

    def _invoke(
        self,
        source: Binder | None = None,
        visited: set[int] | None = None,
    ) -> None:
        """Fire this binder and propagate the current invocation through its graph.

        Args:
            source (Binder | None, optional):
                Binder that immediately forwarded the invocation to this
                instance. The source is skipped when forwarding to neighbors.
                Defaults to ``None`` for a new public invocation.

            visited (set[int] | None, optional):
                Set containing ``id`` values for binders that have already
                participated in the current invocation. ``None`` starts a new
                invocation-local set.

        Returns:
            None:
                Invocation is performed entirely for side effects.

        Notes:
            The visited set provides graph-wide cycle termination. This is
            stronger than remembering only the immediately preceding binder,
            which would terminate ``A <-> B`` but not a longer cycle such as
            ``A -> B -> C -> A``.

            A snapshot of :attr:`_bindees` is captured after :meth:`fire`.
            Bindings added or removed during one binder's local action affect
            subsequent invocations rather than mutating the current iteration.
        """

        invocation: set[int] | None = set[int]() if visited is None else visited
        identity: int = id(self)

        if identity in invocation:
            return

        invocation.add(identity)

        self.fire()

        for bindee in self._bindees:
            if bindee is source:
                continue

            if id(bindee) in invocation:
                continue

            bindee._invoke(self, invocation)

    @property
    def _bindee(self) -> Binder | None:
        """Return the first outgoing bindee for compatibility-oriented code.

        Returns:
            Binder | None:
                First currently bound neighbor, or ``None`` when this binder has
                no outgoing relationships.

        Notes:
            A binder may now have multiple neighbors. Code that needs the full
            graph relationship should use :attr:`_bindees`.
        """

        bindees = self._bindees
        return bindees[0] if bindees else None

    @property
    def _bindees(self) -> tuple[Binder, ...]:
        """Return an immutable snapshot of all outgoing bindees.

        Returns:
            tuple[Binder, ...]:
                Current directed binding targets in insertion order.
        """

        with self.__binding_lock:
            return tuple(self.__bindees)

    @abstractmethod
    def fire(self) -> None:
        """Perform the action local to this binder.

        Concrete binders implement this method with the work that should occur
        when the binder participates in an invocation. Implementations should
        not manually traverse :attr:`_bindees`; graph propagation belongs to
        :meth:`_invoke`.

        Returns:
            None:
                The action is performed for side effects.
        """

        ...


class OneWayBinder(Binder):
    """Create one factory-only binder with an optional outgoing relationship.

    ``OneWayBinder`` creates exactly one new endpoint. If an existing ``bindee``
    is supplied, construction installs the directed edge::

        new -> bindee

    The bindee itself is never cloned, wrapped in a second artificial endpoint,
    or modified to point back to the new binder.

    Concrete subclasses provide their local action by implementing
    :meth:`Binder.fire`.

    Example:
        Define a concrete binder::

            class Printer(OneWayBinder):

                def __init__(
                    self,
                    bindee: Binder | None = None,
                    name: str = '',
                ) -> None:
                    super().__init__(bindee)
                    self.name = name

                def fire(self) -> None:
                    print(self.name)

        Create two existing endpoints and connect one directionally::

            rhs = Printer.create(None, 'rhs')
            lhs = Printer.create(rhs, 'lhs')

        Calling ``lhs`` produces::

            lhs
            rhs

        Calling ``rhs`` produces only::

            rhs

    Notes:
        Construction is factory-only. Normal ``ConcreteOneWayBinder(...)``
        syntax raises :class:`TypeError`; use :meth:`create`.
    """

    @final
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Reject normal direct construction.

        Args:
            *args (Any):
                Positional arguments supplied to an attempted direct
                construction.
            **kwargs (Any):
                Keyword arguments supplied to an attempted direct construction.

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
                Existing binder that should receive invocations from this
                endpoint. ``None`` leaves the new endpoint without outgoing
                edges. Defaults to ``None``.

        Returns:
            None:
                The already allocated endpoint is initialized.

        Notes:
            Concrete subclasses may append additional constructor parameters
            after ``bindee`` and should call ``super().__init__(bindee)``.
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
        """Allocate and initialize exactly one directional binder endpoint.

        Args:
            bindee (Binder | None, optional):
                Existing downstream endpoint. Defaults to ``None``.

            *args (Any):
                Additional positional arguments forwarded to the concrete
                ``__init__`` implementation after ``bindee``.

            **kwargs (Any):
                Additional keyword arguments forwarded unchanged to the
                concrete ``__init__`` implementation.

        Returns:
            Self:
                Fully initialized instance of the concrete ``cls``.

        Raises:
            TypeError:
                May be raised when the supplied subclass-specific arguments do
                not match the concrete initializer.

        Example:
            Given::

                class Printer(OneWayBinder):

                    def __init__(
                        self,
                        bindee: Binder | None = None,
                        name: str = '',
                    ) -> None:
                        super().__init__(bindee)
                        self.name = name

                    def fire(self) -> None:
                        print(self.name)

            construct with::

                endpoint = Printer.create(None, 'endpoint')
                upstream = Printer.create(endpoint, 'upstream')

        Notes:
            The factory intentionally calls :func:`object.__new__` so the
            guarded public ``__new__`` path is bypassed only here.

            The factory creates one new object. ``bindee`` is always an existing
            object supplied by the caller.
        """

        instance: Self = object.__new__(cls)
        cls.__init__(instance, bindee, *args, **kwargs)
        return instance


class TwoWayBinder(OneWayBinder):
    """Create one endpoint and connect it reciprocally to an existing bindee.

    ``TwoWayBinder`` is implemented as two one-way graph edges, not as two
    newly allocated binder objects.

    Given an existing binder ``A``, creating ``B`` with ``A`` as its bindee
    produces::

        B -> A
        A -> B

    which is equivalent to the undirected connection::

        A <-> B

    Only ``B`` is newly allocated. ``A`` is the exact object supplied by the
    caller.

    Example:
        Start with an existing endpoint::

            lhs = Printer.create(None, 'lhs')

        Define a reciprocal endpoint type::

            class ReciprocalPrinter(TwoWayBinder):

                def __init__(
                    self,
                    bindee: Binder | None = None,
                    name: str = '',
                ) -> None:
                    super().__init__(bindee)
                    self.name = name

                def fire(self) -> None:
                    print(self.name)

        Connect one new endpoint to ``lhs``::

            rhs = ReciprocalPrinter.create(lhs, 'rhs')

        The resulting graph contains both edges::

            lhs -> rhs
            rhs -> lhs

    Notes:
        :meth:`create` intentionally mirrors
        :meth:`OneWayBinder.create`. The difference is relationship
        establishment, not object allocation:

        * ``OneWayBinder.create`` installs ``new -> bindee``.
        * ``TwoWayBinder.create`` installs ``new -> bindee`` and
          ``bindee -> new``.

        No second artificial endpoint is created.
    """

    @classmethod
    def create(
        cls,
        bindee: Binder | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Allocate one endpoint and bind it to an existing endpoint both ways.

        Args:
            bindee (Binder | None, optional):
                Existing endpoint to connect reciprocally to the newly allocated
                object.

            *args (Any):
                Additional positional arguments forwarded to the concrete
                initializer after ``bindee``.

            **kwargs (Any):
                Additional keyword arguments forwarded unchanged to the
                concrete initializer.

        Returns:
            Self:
                Newly allocated concrete endpoint. The supplied ``bindee`` is
                reused and is not replaced or cloned.

        Raises:
            ValueError:
                Raised when ``bindee`` is ``None``. Reciprocal construction
                requires an existing opposite endpoint.

            TypeError:
                May be raised when subclass-specific constructor arguments do
                not match the concrete initializer.

        Example:
            Create an existing endpoint::

                lhs = Printer.create(None, 'lhs')

            Add one reciprocal endpoint::

                rhs = ReciprocalPrinter.create(lhs, 'rhs')

            ``rhs`` is the only new object and these relationships hold::

                rhs in lhs._bindees
                lhs in rhs._bindees
        """

        if bindee is None:
            raise ValueError(
                f'{cls.__name__}.create(...) requires an existing bindee'
            )

        instance: Self = object.__new__(cls)

        # The concrete initializer applies the normal one-way edge:
        #
        #     instance -> bindee
        cls.__init__(instance, bindee, *args, **kwargs)

        # Complete the undirected relationship by applying the same primitive
        # in the reverse direction:
        #
        #     bindee -> instance
        bindee._bind(instance)

        return instance
