"""
Expose Aura binding and resilience abstractions.

Aura provides independent middleware patterns that may be selected according
to the protected operation:

* :class:`Debouncer` postpones work until a quiet period.
* :class:`RetryPolicy` retries failures and uses a :class:`CircuitBreaker`.
* :class:`Bulkhead` independently limits concurrent execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .binding import Binder, OneWayBinder, TwoWayBinder
from .breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerOpenError,
    CircuitBreakerState,
)
from .bulkhead import (
    Bulkhead,
    BulkheadError,
    BulkheadFullError,
)
from .retry import RetryPolicy

if TYPE_CHECKING:
    from .debounce import Debouncer

__all__ = [
    'Binder',
    'Bulkhead',
    'BulkheadError',
    'BulkheadFullError',
    'CircuitBreaker',
    'CircuitBreakerError',
    'CircuitBreakerOpenError',
    'CircuitBreakerState',
    'Debouncer',
    'OneWayBinder',
    'RetryPolicy',
    'TwoWayBinder',
]

def __getattr__(name: str):
    """Lazily resolve package members that should not load eagerly.

    Args:
        name (str):
            Attribute requested from :mod:`aura`.

    Returns:
        object:
            Lazily imported package member.

    Raises:
        AttributeError:
            Raised when ``name`` is not a lazily exported member.

    Notes:
        ``Debouncer`` is lazy so ``python -m aura.debounce`` does not import
        the module once during package initialization and then execute it a
        second time through :mod:`runpy`.
    """

    if name == 'Debouncer':
        from .debounce import Debouncer

        return Debouncer

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
