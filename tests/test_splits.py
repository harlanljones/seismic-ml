"""Tests for the W3 `walk_forward_split` contract (HJ-490).

These augment `tests/test_features.py` and must stay green alongside it.
"""

from src.data.splits import walk_forward_split


class TestWalkForwardSplitDefault:
    def test_default_is_proportional_70_15_15(self):
        n = 100
        train, val, test = walk_forward_split(n)
        assert (train.start, train.stop) == (0, 70)
        assert (val.start, val.stop) == (70, 85)
        assert (test.start, test.stop) == (85, 100)

    def test_default_covers_all_indices_no_gap(self):
        n = 137
        train, val, test = walk_forward_split(n)
        # Contiguous: each slice starts where the previous ended.
        assert train.start == 0
        assert train.stop == val.start
        assert val.stop == test.start
        assert test.stop == n
        # Union equals the full index range with no overlap/gap.
        covered = range(n)
        train_idx = range(train.start, train.stop)
        val_idx = range(val.start, val.stop)
        test_idx = range(test.start, test.stop)
        assert list(train_idx) + list(val_idx) + list(test_idx) == list(covered)

    def test_default_slices_non_overlapping_full_union(self):
        n = 250
        train, val, test = walk_forward_split(n)
        # Non-overlapping: boundaries are strict.
        assert train.stop <= val.start
        assert val.stop <= test.start
        # Union is the full range.
        total = (train.stop - train.start) + (val.stop - val.start) + (test.stop - test.start)
        assert total == n


class TestWalkForwardSplitShifted:
    def test_shifted_contiguous_covers_all_indices(self):
        n = 200
        train, val, test = walk_forward_split(n, train_end=0.90, val_end=0.95)
        assert (train.start, train.stop) == (0, 180)
        assert (val.start, val.stop) == (180, 190)
        assert (test.start, test.stop) == (190, 200)
        # Contiguous with no gap.
        assert train.stop == val.start
        assert val.stop == test.start
        # Union covers all indices.
        covered = list(range(train.start, train.stop)) + list(range(val.start, val.stop)) + list(range(test.start, test.stop))
        assert covered == list(range(n))

    def test_shifted_non_overlapping_full_union(self):
        n = 311
        train, val, test = walk_forward_split(n, train_end=0.90, val_end=0.95)
        assert train.stop <= val.start
        assert val.stop <= test.start
        total = (train.stop - train.start) + (val.stop - val.start) + (test.stop - test.start)
        assert total == n
