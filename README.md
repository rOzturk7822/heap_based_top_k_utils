# heap_based_top_k_utils

Maintains a bounded min-heap of the largest N numeric observations seen so far, with O(log N) insertion and O(N log N) extraction.

```python
from heap_based_top_k_utils import TopKTracker, TopKResult

tracker = TopKTracker(3)
tracker.push(5)
tracker.push(1)
tracker.push(9)
tracker.push(3)
tracker.push(7)

snapshot: TopKResult = tracker.peek()
print(snapshot.values)   # [9, 7, 5]
print(snapshot.count)     # 3
print(snapshot.capacity)  # 3
print(tracker.peek_min()) # 5  — the threshold to beat

result = tracker.extract()  # empties the tracker
print(result.values)        # [9, 7, 5]
```

## Why this exists

When you need the largest *k* of a high-volume stream, keeping every
value and sorting at the end is O(M log M) in memory and time, where M
is the stream length. A min-heap of size *k* cuts that to O(M log k)
time and O(k) space: each observation is compared to the current
smallest winner in O(1), and only displaces it via an O(log k) heap
operation when it is larger.

The trade-off is that extraction in sorted (descending) order is
O(k log k) — you must drain the heap. If you need fully sorted output
on every observation, a sorted structure would be better; if you only
need the set of winners at the end, this is the right shape.

## Decisions worth knowing

- Only `int` and `float` are accepted. `bool` is rejected even though
  it subclasses `int`, because accepting `True` as 1 is almost always a
  caller bug. `NaN` is rejected with `ValueError` because it has no
  useful ordering and silently corrupts heap comparisons. `inf` and
  `-inf` are accepted — they are real numbers with total ordering.
- A value that ties the current minimum does **not** evict it. The heap
  uses strict `>` for replacement, so among equal values the earliest
  one observed is retained. This is the least surprising policy for
  streaming data.
- `push` validates before mutating. A rejected value (wrong type or
  NaN) leaves the heap untouched.
- `extract()` is destructive — it empties the tracker. Use `peek()` for
  a non-destructive snapshot, or `reset()` to clear without extracting.

## Run the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
