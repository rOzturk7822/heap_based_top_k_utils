"""Tests for heap_based_top_k_utils.core."""

import unittest

from heap_based_top_k_utils.core import TopKTracker, TopKResult


class TestConstruction(unittest.TestCase):

    def test_basic_capacity_one(self):
        t = TopKTracker(1)
        self.assertEqual(t.capacity, 1)
        self.assertEqual(t.size, 0)

    def test_zero_capacity_rejected(self):
        with self.assertRaises(ValueError):
            TopKTracker(0)

    def test_negative_capacity_rejected(self):
        with self.assertRaises(ValueError):
            TopKTracker(-5)

    def test_bool_capacity_rejected(self):
        # bool is a subclass of int; must be rejected explicitly.
        with self.assertRaises(TypeError):
            TopKTracker(True)
        with self.assertRaises(TypeError):
            TopKTracker(False)


class TestPush(unittest.TestCase):

    def test_retains_largest_n(self):
        t = TopKTracker(3)
        for v in (5, 1, 9, 3, 7, 2, 8):
            t.push(v)
        result = t.peek()
        self.assertEqual(result.values, [9, 8, 7])
        self.assertEqual(result.count, 3)
        self.assertEqual(result.capacity, 3)

    def test_fills_below_capacity(self):
        t = TopKTracker(5)
        t.push(10)
        t.push(20)
        self.assertEqual(t.size, 2)
        self.assertEqual(t.peek().values, [20, 10])

    def test_int_values_stored_as_float(self):
        t = TopKTracker(2)
        t.push(3)
        t.push(7)
        for v in t.peek().values:
            self.assertIsInstance(v, float)

    def test_tie_keeps_existing_entry(self):
        # Documented behaviour: a tie at the boundary does not evict.
        t = TopKTracker(3)
        t.push(5)
        t.push(5)
        t.push(5)
        self.assertEqual(t.size, 3)
        t.push(5)  # ties min (5); existing entry retained
        self.assertEqual(t.size, 3)
        self.assertEqual(t.peek().values, [5, 5, 5])

    def test_float_inf_accepted(self):
        t = TopKTracker(2)
        t.push(float("inf"))
        t.push(1e308)
        result = t.peek()
        self.assertEqual(result.values[0], float("inf"))

    def test_float_neg_inf_accepted(self):
        t = TopKTracker(2)
        t.push(float("-inf"))
        t.push(5)
        result = t.peek()
        self.assertEqual(result.values, [5, float("-inf")])

    def test_nan_rejected(self):
        t = TopKTracker(3)
        with self.assertRaises(ValueError):
            t.push(float("nan"))
        # Heap must be untouched after a rejected push.
        self.assertEqual(t.size, 0)

    def test_nan_rejected_after_valid_push(self):
        t = TopKTracker(3)
        t.push(10)
        with self.assertRaises(ValueError):
            t.push(float("nan"))
        self.assertEqual(t.size, 1)
        self.assertEqual(t.peek().values, [10])

    def test_bool_value_rejected(self):
        t = TopKTracker(3)
        with self.assertRaises(TypeError):
            t.push(True)
        self.assertEqual(t.size, 0)

    def test_string_value_rejected(self):
        t = TopKTracker(3)
        with self.assertRaises(TypeError):
            t.push("5")
        self.assertEqual(t.size, 0)

    def test_none_value_rejected(self):
        t = TopKTracker(3)
        with self.assertRaises(TypeError):
            t.push(None)
        self.assertEqual(t.size, 0)


class TestPushMany(unittest.TestCase):

    def test_push_many_basic(self):
        t = TopKTracker(3)
        t.push_many([1, 2, 3, 4, 5])
        self.assertEqual(t.peek().values, [5, 4, 3])

    def test_push_many_empty_iterable(self):
        t = TopKTracker(3)
        t.push_many([])
        self.assertEqual(t.size, 0)

    def test_push_many_generator(self):
        t = TopKTracker(3)
        t.push_many(x * 2 for x in range(10))
        self.assertEqual(t.peek().values, [18, 16, 14])

    def test_push_many_stops_at_invalid_value(self):
        t = TopKTracker(5)
        with self.assertRaises(TypeError):
            t.push_many([1, 2, "oops", 4])
        # Values before the error are retained (documented streaming
        # behaviour).
        self.assertEqual(t.size, 2)
        self.assertEqual(t.peek().values, [2, 1])


class TestPeekAndExtract(unittest.TestCase):

    def test_peek_does_not_mutate(self):
        t = TopKTracker(3)
        for v in (4, 8, 2):
            t.push(v)
        first = t.peek()
        second = t.peek()
        self.assertEqual(first.values, second.values)
        self.assertEqual(t.size, 3)

    def test_extract_empties_tracker(self):
        t = TopKTracker(3)
        for v in (4, 8, 2):
            t.push(v)
        result = t.extract()
        self.assertEqual(result.values, [8, 4, 2])
        self.assertEqual(t.size, 0)

    def test_extract_on_empty_returns_empty_result(self):
        t = TopKTracker(3)
        result = t.extract()
        self.assertEqual(result.count, 0)
        self.assertEqual(result.values, [])
        self.assertEqual(result.capacity, 3)

    def test_peek_min_returns_threshold(self):
        t = TopKTracker(3)
        self.assertIsNone(t.peek_min())
        t.push(5)
        t.push(9)
        t.push(3)
        self.assertEqual(t.peek_min(), 3)

    def test_peek_min_after_eviction(self):
        t = TopKTracker(2)
        t.push(10)
        t.push(20)
        # 15 evicts 10; new min is 15.
        t.push(15)
        self.assertEqual(t.peek_min(), 15)


class TestReset(unittest.TestCase):

    def test_reset_clears_values(self):
        t = TopKTracker(3)
        t.push(1)
        t.push(2)
        t.reset()
        self.assertEqual(t.size, 0)
        self.assertEqual(t.capacity, 3)
        self.assertIsNone(t.peek_min())

    def test_reusable_after_reset(self):
        t = TopKTracker(2)
        t.push(100)
        t.push(200)
        t.reset()
        t.push(1)
        t.push(2)
        t.push(3)
        self.assertEqual(t.peek().values, [3, 2])


class TestDunderMethods(unittest.TestCase):

    def test_len_tracks_size(self):
        t = TopKTracker(3)
        self.assertEqual(len(t), 0)
        t.push(1)
        self.assertEqual(len(t), 1)
        t.push(2)
        t.push(3)
        self.assertEqual(len(t), 3)
        t.push(4)  # evicts 1
        self.assertEqual(len(t), 3)

    def test_iter_descending(self):
        t = TopKTracker(3)
        for v in (3, 1, 2):
            t.push(v)
        self.assertEqual(list(iter(t)), [3, 2, 1])
        # Iteration must not mutate.
        self.assertEqual(t.size, 3)

    def test_repr(self):
        t = TopKTracker(5)
        t.push(1)
        text = repr(t)
        self.assertIn("TopKTracker", text)
        self.assertIn("capacity=5", text)
        self.assertIn("size=1", text)


class TestTopKResult(unittest.TestCase):

    def test_result_is_frozen(self):
        r = TopKResult(capacity=3, count=2, values=[5, 1])
        with self.assertRaises(Exception):
            r.capacity = 10  # type: ignore[misc]


class TestStress:
    """Property-style checks run deterministically."""

    def _sorted_desc(self, vals):
        return sorted(vals, reverse=True)

    def test_large_stream_matches_naive_sort(self):
        import random
        rng = random.Random(20240517)
        data = [rng.random() for _ in range(5000)]
        k = 100
        t = TopKTracker(k)
        t.push_many(data)
        result = t.peek()
        expected = self._sorted_desc(data)[:k]
        # Compare with a tolerance appropriate for float ordering.
        self.assertEqual(len(result.values), k)
        for got, exp in zip(result.values, expected):
            self.assertAlmostEqual(got, exp, places=12)

    def test_descending_input_keeps_last_k(self):
        # Descending input means every new value is a new max; the heap
        # should retain exactly the last k values of the stream.
        k = 5
        data = list(range(100, 0, -1))
        t = TopKTracker(k)
        t.push_many(data)
        result = t.extract()
        self.assertEqual(result.values, self._sorted_desc(data)[:k])

    def test_ascending_input_keeps_first_k(self):
        # Ascending input means no value ever evicts; the heap keeps
        # the first k values.
        k = 5
        data = list(range(1, 101))
        t = TopKTracker(k)
        t.push_many(data)
        result = t.extract()
        self.assertEqual(result.values, self._sorted_desc(data[:k]))


if __name__ == "__main__":
    unittest.main()
