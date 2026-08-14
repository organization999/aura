"""
Expose the public Aura binding and debounce abstractions.

The binding abstractions are imported eagerly because they form the core of
the package. :class:`Debouncer` is imported lazily so executing::

    python -m aura.debounce

does not cause :mod:`aura.debounce` to be imported once by package
initialization and then executed a second time by :mod:`runpy`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .binding import Binder, OneWayBinder, TwoWayBinder

if TYPE_CHECKING:
    from .debounce import Debouncer

__all__ = [
    'Binder',
    'Debouncer',
    'OneWayBinder',
    'TwoWayBinder',
]

def __getattr__(name: str):
    """Lazily resolve package-level attributes.

    Args:
        name (str):
            Name requested from the :mod:`aura` package.

    Returns:
        object:
            Lazily imported public object.

    Raises:
        AttributeError:
            Raised when ``name`` is not a lazily exported package attribute.

    Notes:
        ``Debouncer`` is deliberately imported only when requested. This
        preserves support for::

            from aura import Debouncer

        without eagerly importing :mod:`aura.debounce` during package
        initialization.
    """

    if name == 'Debouncer':
        from .debounce import Debouncer

        return Debouncer

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
