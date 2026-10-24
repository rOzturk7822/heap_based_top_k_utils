"""Core implementation for tracking the largest N numeric observations."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class TopKResult:
    """Immutable snapshot of the tracked top-k values, largest first.

    Values are sorted descending so callers can iterate without thinking
    about the min-heap ordering used internally.
    """

    capacity: int
    count: int
    values: List[float]


class TopKTracker:
    """Bounded min-heap that retains the largest *capacity* numeric values seen.

    A min-heap is the natural structure: the root is the smallest of the
    current winners, so a new value only needs to displace the root when
    it is larger. This gives O(log N) per observation and O(N log N) to
    extract everything in sorted (descending) order.

    Interpretation chosen (stated plainly so tests are not surprising):

    * Only real numbers (``int`` or ``float``) are accepted. Anything else
      raises ``TypeError`` *before* mutating the heap, so a rejected
      insertion leaves the tracker untouched.
    * ``float('nan')`` raises ``ValueError``. NaN comparisons are never
      useful in a top-k context and silently corrupt ordering, so the
      tracker refuses them rather than guessing a policy.
    * ``float('inf')`` and ``float('-inf')`` are accepted — they are
      real numbers with a total ordering and behave correctly here.
    * Inserting a value when the tracker is exactly full and the value
      ties the current minimum keeps the *existing* entry and discards
      the new one. With a min-heap this is the default ``<=`` behaviour
      and keeps the earliest observation among ties, which is the least
      surprising choice for streaming data.
    * Calling :meth:`extract` empties the tracker. Use :meth:`peek` for a
      non-destructive view.
    """

    __slots__ = ("_capacity", "_heap")

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool):
            # ``bool`` is a subclass of ``int`` in Python, so
            # ``TopKTracker(True)`` would otherwise be accepted as
            # capacity 1. Reject it to avoid a confusing silent success.
            raise TypeError("capacity must be an int, not bool")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity: int = capacity
        self._heap: List[float] = []

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_value(value: object) -> float:
        """Validate *value* and return it as a float.

        Kept separate from :meth:`push` so the validation order is
        explicit: type check first, NaN check second. This matters when
        reasoning about partial mutation — if validation fails, the heap
        is guaranteed to be untouched.
        """
        if isinstance(value, bool):
            # ``bool`` is also a subclass of ``int``; accepting ``True``
            # as 1 is almost always a caller bug, so reject early.
            raise TypeError("value must be a real number, not bool")
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"value must be an int or float, got {type(value).__name__}"
            )
        fval = float(value)
        if fval != fval:  # NaN check — the only self-unequal float.
            raise ValueError("NaN is not a valid observation for TopKTracker")
        return fval

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        """Number of values currently held (0 <= size <= capacity)."""
        return len(self._heap)

    def push(self, value: object) -> None:
        """Observe *value*.

        Raises ``TypeError`` if *value* is not a real number, or
        ``ValueError`` if it is NaN. On either error the heap is
        unchanged.
        """
        fval = self._check_value(value)
        heap = self._heap
        if len(heap) < self._capacity:
            heapq.heappush(heap, fval)
        elif fval > heap[0]:
            # Strict ``>`` so ties keep the existing entry. See class
            # docstring for the rationale.
            heapq.heapreplace(heap, fval)

    def push_many(self, values: Iterable[object]) -> None:
        """Observe every value in *values* in order.

        If an invalid value is encountered mid-iteration, all values
        processed before it remain in the heap (that is the natural
        consequence of streaming ``push`` calls — there is no snapshot
        to roll back to without a copy, and callers who need atomicity
        should snapshot first).
        """
        for v in values:
            self.push(v)

    def peek_min(self) -> Optional[float]:
        """Return the smallest currently-held value, or ``None`` if empty.

        This is the heap root — the "threshold to beat" — so it is O(1).
        """
        if not self._heap:
            return None
        return self._heap[0]

    def peek(self) -> TopKResult:
        """Non-destructive snapshot, sorted largest-first."""
        # ``sorted`` on a list of numbers is stable and descending via
        # ``reverse=True``. Copy first so the internal heap is untouched.
        sorted_vals = sorted(self._heap, reverse=True)
        return TopKResult(
            capacity=self._capacity,
            count=len(sorted_vals),
            values=sorted_vals,
        )

    def extract(self) -> TopKResult:
        """Destructive extraction: returns the result then empties the tracker.

        O(N log N) because we drain the heap via repeated pop to get
        ascending order, then reverse.
        """
        heap = self._heap
        if not heap:
            return TopKResult(
                capacity=self._capacity, count=0, values=[]
            )
        ascending: List[float] = [heapq.heappop(heap) for _ in range(len(heap))]
        ascending.reverse()
        return TopKResult(
            capacity=self._capacity,
            count=len(ascending),
            values=ascending,
        )

    def reset(self) -> None:
        """Discard all observations without changing the capacity."""
        self._heap.clear()

    def __iter__(self) -> Iterator[float]:
        # Iterate over a descending-sorted copy so callers see the
        # largest first. Mutating the heap during iteration would be
        # undefined, so we snapshot.
        return iter(sorted(self._heap, reverse=True))

    def __len__(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        return f"TopKTracker(capacity={self._capacity}, size={len(self._heap)})"
