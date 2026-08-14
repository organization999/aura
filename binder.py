'''
Provide callable callback-binding and propagation abstractions.

This module defines a hierarchy for associating callback-bearing objects and
propagating an invocation from one binder to another.

A binder behaves as a callable object. Invoking a binder with::

    binder()

causes the callback owned by that binder to execute. Depending on the
relationship represented by the binder, the invocation may then propagate to
another :class:`Binder`.

The central abstraction is :class:`Binder`. Every binder owns:

* A callback executed when the binder participates in an invocation.
* An optional reference to another binder, called the ``bindee``.
* An abstract :meth:`Binder.__call__` operation that concrete binder types must
  explicitly implement.
* A protected :meth:`Binder._invoke` operation containing the shared callback
  execution and propagation algorithm.

Relationships are established during object construction rather than through
a post-construction ``bind()`` operation. The stored callback and bindee are
therefore declared with :class:`typing.Final` to communicate that those
references are intended to remain unchanged after initialization.

Two concrete binding strategies are provided:

``OneWayBinder``
    Represents a directional relationship between two binders.

    Given::

        A -> B

    invoking ``A`` executes ``A`` and then propagates the invocation to ``B``.
    Invoking ``B`` does not propagate back to ``A`` unless ``B`` independently
    owns another relationship.

``TwoWayBinder``
    Represents a reciprocal relationship between two binders.

    A two-way pair is constructed as::

        A <-> B

    Because each object must reference the other during initialization,
    :meth:`TwoWayBinder.create` first allocates both instances without
    initializing them. After both objects exist, each object's constructor is
    invoked with the opposite object as its bindee.

Invocation propagation uses the immediately preceding binder as a ``source``.
This prevents a reciprocal relationship from indefinitely bouncing between
its two endpoints.

Example:
    Create a one-way relationship::

        b = OneWayBinder.create(lambda: print('b'))
        a = OneWayBinder.create(lambda: print('a'), b)

        a()

    This produces::

        a
        b

    Create a reciprocal relationship::

        a, b = TwoWayBinder.create(
            lambda: print('a'),
            lambda: print('b'),
        )

        a()

    This also produces::

        a
        b

    Invoking the opposite endpoint reverses the direction of propagation::

        b()

    producing::

        b
        a

Notes:
    Binder relationships are constructor-established. There is intentionally
    no public operation for replacing a bindee after construction.

    :class:`TwoWayBinder` requires a factory because two mutually referencing
    objects cannot ordinarily be passed to one another's constructors before
    both objects exist.

    The source-tracking algorithm implemented by :meth:`Binder._invoke`
    prevents immediate reciprocal recursion of the form::

        A -> B -> A

    This is sufficient for the two-object relationship created by
    :meth:`TwoWayBinder.create`.

Warning:
    The source-tracking algorithm remembers only the immediately preceding
    binder. It is not general graph-wide cycle detection.

    A larger cyclic structure such as::

        A -> B -> C -> A

    would require additional propagation state, such as a set containing all
    previously visited binders, in order to guarantee cycle termination.
'''

from __future__ import annotations

from abc    import ABC, abstractmethod
from typing import Callable, Final, Self

from typing_extensions import override

class Binder(ABC):
    '''
    Define the common state and invocation behavior of callback binders.

    ``Binder`` is the abstract base of all binder relationship types.

    A binder owns a callback and optionally references one other binder. The
    referenced object is called the ``bindee``. When the binder participates
    in an invocation, its callback executes first and the invocation may then
    propagate to its bindee.

    Concrete subclasses must explicitly implement :meth:`__call__`, thereby
    defining how invocation of the concrete binder begins. The common
    propagation behavior itself is implemented by :meth:`_invoke`.

    Binder relationships are established during construction. There is no
    post-construction ``bind()`` operation.

    Attributes:
        _bindee (Final[Binder | None]):
            Binder to which an invocation should propagate after this
            binder's callback has executed.

            ``None`` represents an endpoint with no downstream binder.

            The reference is declared :class:`typing.Final`, indicating that
            it is intended to be assigned once during initialization.

        _callback (Final[Callable]):
            Callable executed whenever this binder participates in an
            invocation.

            The callback is invoked without arguments.

            The reference is declared :class:`typing.Final`, indicating that
            the callback associated with a binder is intended to remain fixed
            throughout the binder's lifetime.

    Notes:
        ``Binder`` is an abstract class and cannot be instantiated directly
        because :meth:`__call__` is abstract.

        The actual callback execution and propagation algorithm is centralized
        in :meth:`_invoke`. Concrete subclasses generally begin propagation by
        calling::

            self._invoke()

        Keeping propagation in a protected helper avoids duplicating the
        callback and source-tracking logic across concrete binder types.

        ``Final`` communicates an intended static invariant to type checkers.
        It does not make the referenced Python object physically immutable at
        runtime.
    '''

    def __init__(self, callback : Callable,
                       bindee   : Binder | None = None) -> Self:
        '''
        Initialize the callback and relationship owned by this binder.

        The constructor permanently associates the binder with its callback
        and optional bindee.

        The bindee reference determines where invocation propagation continues
        after this binder's callback has executed.

        Args:
            callback (Callable):
                Callable associated with this binder.

                The callback is executed without arguments whenever the binder
                participates in an invocation.

                For example::

                    lambda: print('executed')

            bindee (Binder | None, optional):
                Binder to which an invocation should propagate after this
                binder's callback executes.

                If ``None``, the binder represents the end of a propagation
                chain.

                Defaults to ``None``.

        Returns:
            None:
                Constructors initialize existing instances and do not produce a
                separate return value.

        Example:
            Create an endpoint through a concrete implementation::

                endpoint = OneWayBinder(
                    lambda: print('endpoint')
                )

            Create another binder whose invocation propagates to that
            endpoint::

                upstream = OneWayBinder(
                    lambda: print('upstream'),
                    endpoint,
                )

            Invoking::

                upstream()

            produces::

                upstream
                endpoint

        Notes:
            Relationships are established directly through this constructor.
            The class intentionally does not provide a public mutator for
            replacing ``bindee`` after construction.

            :class:`TwoWayBinder.create` also uses this constructor, but first
            allocates both members of the reciprocal relationship so that each
            one can be supplied as the other's ``bindee``.
        '''

        # Store the binder that receives propagated invocations. The reference
        # is intended to remain unchanged after construction.
        self._bindee: Final[Binder | None] = bindee

        # Store the callable associated with this binder. The callback
        # reference is likewise intended to remain unchanged after
        # construction.
        self._callback: Final[Callable] = callback

    @abstractmethod
    def __call__(self) -> None:
        '''
        Begin invocation of this binder.

        ``__call__`` makes binder instances directly callable::

            binder()

        rather than requiring an explicit operation such as::

            binder.fire()

        Every concrete binder type must explicitly implement this method. This
        requirement makes invocation semantics part of the concrete binder's
        contract even when several binder types ultimately reuse the same
        protected propagation algorithm.

        Concrete implementations in this module begin invocation with::

            self._invoke()

        Returns:
            None:
                Invocation exists entirely for its callback and propagation
                side effects.

        Side Effects:
            A concrete implementation will normally execute this binder's
            callback through :meth:`_invoke`.

            If a bindee exists, execution may propagate to additional binder
            callbacks.

        Notes:
            This method intentionally contains no common implementation.
            Shared invocation mechanics belong to :meth:`_invoke`.

            Declaring ``__call__`` abstract forces each concrete relationship
            type to make an explicit decision about how invocation begins.
        '''

        ...

    def _invoke(self, source: Binder | None = None) -> None:
        '''
        Execute this binder and propagate the invocation to its bindee.

        This method implements the shared propagation algorithm used by the
        concrete binder classes.

        The current binder's callback is always executed first. Propagation
        then stops if either:

        * This binder has no bindee.
        * This binder's bindee is the binder from which the current invocation
          arrived.

        Otherwise, invocation is forwarded to the bindee and ``self`` becomes
        the source supplied to the next propagation step.

        Conceptually, the algorithm is::

            execute callback

            if no bindee:
                stop

            if bindee is source:
                stop

            invoke bindee with self as source

        Args:
            source (Binder | None, optional):
                Binder from which the current invocation was received.

                ``None`` indicates that this binder is the origin of the
                current invocation.

                Defaults to ``None``.

        Returns:
            None:
                Propagation operates entirely through callback side effects.

        Example:
            Consider the reciprocal relationship::

                A <-> B

            Beginning an invocation at ``A`` performs::

                A._invoke(None)
                    |
                    +-- execute A
                    |
                    +-- B._invoke(A)
                            |
                            +-- execute B
                            |
                            +-- B._bindee is A
                                A is source
                                STOP

            This causes each endpoint to execute exactly once for that
            invocation.

        Notes:
            The ``source`` parameter is invocation-local state. No mutable
            ``is_firing`` or similar state needs to be stored on the binder
            itself.

            This is particularly useful for reciprocal relationships because
            it prevents the two endpoints from repeatedly invoking each other.

            Source comparison uses object identity through ``is`` rather than
            equality. The question being answered is whether the downstream
            object is the exact binder that immediately invoked this object.

        Warning:
            ``source`` tracks only the immediately preceding binder.

            It therefore prevents the immediate reciprocal cycle::

                A -> B -> A

            but it does not detect arbitrary graph cycles such as::

                A -> B -> C -> A

            Supporting arbitrary cyclic binder graphs would require carrying
            broader invocation context, such as a collection of visited
            binders.
        '''

        # Execute the callback owned by the current binder before considering
        # downstream propagation.
        self._callback()

        # No bindee means that this binder is the terminal endpoint of the
        # current propagation chain.
        if self._bindee is None:
            return

        # Prevent an invocation from being immediately propagated back to the
        # binder from which it was received. This terminates the reciprocal
        # relationship produced by TwoWayBinder.create().
        if self._bindee is source:
            return

        # Continue propagation. The current binder becomes the source of the
        # invocation observed by the downstream binder.
        self._bindee._invoke(self)


class OneWayBinder(Binder):
    '''
    Implement a directional callback relationship.

    ``OneWayBinder`` represents a relationship in which invocation propagates
    from the current binder to an optional downstream binder.

    Given two instances ``A`` and ``B`` constructed as::

        A = OneWayBinder(callback_a, B)

    the relationship is::

        A -> B

    Invoking ``A`` executes ``A`` and then propagates to ``B``. The
    relationship itself does not create a reverse reference from ``B`` to
    ``A``.

    Relationships are fixed during construction. There is no separate
    ``bind()`` operation.

    Example:
        Construct a terminal binder::

            b = OneWayBinder(
                lambda: print('b')
            )

        Construct an upstream binder::

            a = OneWayBinder(
                lambda: print('a'),
                b,
            )

        Invoke the upstream endpoint::

            a()

        producing::

            a
            b

    Notes:
        A ``OneWayBinder`` may also be created through :meth:`create`.

        ``__call__`` begins the common propagation algorithm by invoking
        :meth:`Binder._invoke`.
    '''

    def __init__(self, callback : Callable,
                       bindee   : Binder | None = None) -> None:
        '''
        Initialize a directional binder.

        Args:
            callback (Callable):
                Callable executed whenever this binder participates in an
                invocation.

                The callable is expected to accept no arguments.

            bindee (Binder | None, optional):
                Optional downstream binder.

                If supplied, invocation of this binder propagates to the
                bindee after the current callback executes.

                If ``None``, invocation terminates after this binder's
                callback executes.

                Defaults to ``None``.

        Returns:
            None:
                Constructors initialize instances and do not return a value.

        Example:
            Create an endpoint::

                endpoint = OneWayBinder(
                    lambda: print('endpoint')
                )

            Create a directional relationship to that endpoint::

                root = OneWayBinder(
                    lambda: print('root'),
                    endpoint,
                )
        '''

        # Delegate initialization of the immutable callback and bindee
        # references to Binder.
        super().__init__(callback, bindee)

    @override
    def __call__(self) -> None:
        '''
        Invoke this directional binder.

        Invocation begins the shared propagation algorithm implemented by
        :meth:`Binder._invoke`.

        The current callback executes first. If a bindee exists and
        propagation is permitted, that bindee is then invoked.

        Returns:
            None:
                Invocation is performed only for its side effects.

        Example:
            Given::

                b = OneWayBinder(lambda: print('b'))
                a = OneWayBinder(lambda: print('a'), b)

            invoking::

                a()

            produces::

                a
                b
        '''

        # Begin propagation with no source because this binder is the origin of
        # the invocation.
        self._invoke()

    @staticmethod
    def create(callback : Callable,
               bindee   : Binder | None = None) -> Self:
        '''
        Create and initialize a one-way binder.

        This factory provides an explicit construction operation while
        preserving the same parameters accepted by :class:`OneWayBinder`.

        Args:
            callback (Callable):
                Callable executed whenever the newly created binder
                participates in an invocation.

            bindee (Binder | None, optional):
                Optional downstream binder associated with the newly created
                instance.

                Defaults to ``None``.

        Returns:
            Self:
                Newly initialized :class:`OneWayBinder`.

        Example:
            Create a terminal binder::

                b = OneWayBinder.create(
                    lambda: print('b')
                )

            Create a binder pointing to it::

                a = OneWayBinder.create(
                    lambda: print('a'),
                    b,
                )

            Invoke the relationship::

                a()

        Notes:
            Unlike :meth:`TwoWayBinder.create`, this factory does not need to
            separate allocation from initialization because a one-directional
            relationship has no circular construction dependency.
        '''

        return OneWayBinder(callback, bindee)


class TwoWayBinder(Binder):
    '''
    Implement a reciprocal callback relationship between two binders.

    ``TwoWayBinder`` represents a pair in which each endpoint stores the other
    endpoint as its bindee.

    Conceptually, the relationship is::

        A <-> B

    This creates a circular construction dependency: ``A`` requires ``B`` as
    a constructor argument while ``B`` simultaneously requires ``A``.

    :meth:`create` resolves that dependency by separating Python object
    allocation from initialization:

    1. Allocate ``A`` without calling ``__init__``.
    2. Allocate ``B`` without calling ``__init__``.
    3. Initialize ``A`` with ``B`` as its bindee.
    4. Initialize ``B`` with ``A`` as its bindee.

    Both resulting objects therefore receive their final bindee through their
    constructors rather than through post-construction mutation.

    Invocation uses the source-aware :meth:`Binder._invoke` algorithm so that
    an invocation crossing the reciprocal relationship does not immediately
    bounce back indefinitely.

    Example:
        Construct a reciprocal pair::

            a, b = TwoWayBinder.create(
                lambda: print('a'),
                lambda: print('b'),
            )

        Invoking the left endpoint::

            a()

        produces::

            a
            b

        Invoking the right endpoint::

            b()

        produces::

            b
            a

    Notes:
        The pair factory preserves the constructor-based relationship
        invariant. Neither binder's ``_bindee`` reference needs to be mutated
        after initialization.

        Calling the constructor directly is still possible and behaves like a
        normal binder constructor. The :meth:`create` factory is specifically
        required when constructing a mutually referencing pair.

    Warning:
        The source-aware invocation logic is designed to terminate the direct
        two-object reciprocal relationship produced by :meth:`create`.

        It is not general graph-wide cycle detection.
    '''

    def __init__(self, callback : Callable,
                       bindee   : Binder | None = None) -> None:
        '''
        Initialize one endpoint of a reciprocal-capable binder relationship.

        This constructor has the same parameter structure as
        :class:`OneWayBinder`.

        The distinction between one-way and reciprocal construction is not in
        the constructor signature. Instead, :meth:`create` determines whether
        two instances are allocated and initialized as a mutually referencing
        pair.

        Args:
            callback (Callable):
                Callable executed whenever this binder participates in an
                invocation.

            bindee (Binder | None, optional):
                Binder associated with this endpoint.

                In a reciprocal pair produced by :meth:`create`, this argument
                is the opposite endpoint.

                Defaults to ``None``.

        Returns:
            None:
                Constructors initialize instances and do not return a value.

        Example:
            Direct construction is valid::

                binder = TwoWayBinder(
                    lambda: print('binder')
                )

            Reciprocal construction should normally use::

                lhs, rhs = TwoWayBinder.create(
                    lambda: print('lhs'),
                    lambda: print('rhs'),
                )
        '''

        # Initialize the callback and optional bindee through the common Binder
        # constructor.
        super().__init__(callback, bindee)

    @override
    def __call__(self) -> None:
        '''
        Invoke this reciprocal binder endpoint.

        Invocation begins the source-aware propagation algorithm implemented
        by :meth:`Binder._invoke`.

        When this binder belongs to a reciprocal pair, the callback of the
        originating endpoint executes first, followed by the opposite
        endpoint. Propagation then stops when the opposite endpoint observes
        that its bindee is the source of the invocation.

        Returns:
            None:
                Invocation is performed only for its callback side effects.

        Example:
            Given::

                a, b = TwoWayBinder.create(
                    lambda: print('a'),
                    lambda: print('b'),
                )

            invoking::

                a()

            follows approximately::

                a._invoke(None)
                    -> callback a
                    -> b._invoke(a)
                        -> callback b
                        -> stop

        Notes:
            Calling either endpoint independently starts a new invocation with
            ``source=None``.
        '''

        # Begin propagation with no source because this endpoint is the origin
        # of the current invocation.
        self._invoke()

    @classmethod
    def create(cls, lhs_callback : Callable,
                    rhs_callback : Callable) -> tuple[Self, Self]:
        '''
        Create a mutually bound pair of binder instances.

        This factory resolves the circular constructor dependency inherent in a
        reciprocal relationship.

        Ordinarily, constructing::

            lhs = cls(lhs_callback, rhs)
            rhs = cls(rhs_callback, lhs)

        is impossible because ``rhs`` must exist before ``lhs`` can be
        initialized while ``lhs`` must simultaneously exist before ``rhs`` can
        be initialized.

        ``create`` resolves this by separating allocation from initialization.

        First, :meth:`object.__new__` is reached through ``cls.__new__`` to
        allocate both objects without invoking their constructors. Once both
        identities exist, ``cls.__init__`` initializes each endpoint with the
        other endpoint as its bindee.

        Args:
            lhs_callback (Callable):
                Callback associated with the left-hand endpoint.

                The callable is expected to accept no arguments.

            rhs_callback (Callable):
                Callback associated with the right-hand endpoint.

                The callable is expected to accept no arguments.

        Returns:
            tuple[Self, Self]:
                Two fully initialized binder instances.

                The first returned binder references the second::

                    lhs._bindee is rhs

                and the second references the first::

                    rhs._bindee is lhs

                producing the reciprocal relationship::

                    lhs <-> rhs

        Example:
            Create a reciprocal pair::

                lhs, rhs = TwoWayBinder.create(
                    lambda: print('lhs'),
                    lambda: print('rhs'),
                )

            Invoke from the left::

                lhs()

            producing::

                lhs
                rhs

            Invoke from the right::

                rhs()

            producing::

                rhs
                lhs

        Notes:
            Both objects are allocated before either object is initialized.

            This permits each constructor to receive the opposite endpoint
            while preserving the design in which ``_bindee`` is established
            during construction.

            The factory uses ``cls`` rather than hard-coding
            :class:`TwoWayBinder`, allowing subclasses to inherit the factory
            and receive instances of the subclass.

        Warning:
            Between ``cls.__new__`` and ``cls.__init__``, the allocated objects
            exist but have not yet had the state defined by :class:`Binder`
            initialized.

            The factory does not expose those partially initialized objects;
            both constructors complete before the pair is returned.
        '''

        # Allocate both objects without running __init__. This makes both
        # object identities available before either needs the other as a
        # constructor argument.
        lhs: Self = cls.__new__(cls)
        rhs: Self = cls.__new__(cls)

        # Initialize each endpoint with the opposite endpoint as its bindee.
        # The relationship is therefore established entirely through
        # construction rather than post-construction mutation.
        cls.__init__(lhs, lhs_callback, rhs)
        cls.__init__(rhs, rhs_callback, lhs)

        # Return the fully initialized reciprocal pair.
        return lhs, rhs
