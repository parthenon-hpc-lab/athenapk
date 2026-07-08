"""Tests for `ewah_bool_utils` package."""

import time

import numpy as np

from ewah_bool_utils._testing import (
    ewah_set_and_unset,
    find_ewah_collisions,
    make_and_select_from_ewah_index,
)

np.random.seed(0)


class Test_ewah_bool_array:
    """Test class for `ewah_set_and_unset` and `find_ewah_collisions`"""

    def test_ewah_set_and_unset_inputInt(self):
        """Test with integer inputs"""
        arr = np.array([1, 2, 3])

        assert np.all(ewah_set_and_unset(arr)) == np.all(arr)

    def test_find_ewah_collisions(self):
        """Test with integer inputs"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([3, 4, 5])

        assert find_ewah_collisions(arr1, arr2) == 1

    def test_make_and_select_from_ewah_index(self):
        """Test with float64 inputs"""

        arr = np.random.rand(2000000)
        np_idx = np.random.choice(range(2000000), size=1000000, replace=False)
        np_idx.sort()
        out_array_gt = arr[np_idx]

        start = time.process_time()
        out_array = make_and_select_from_ewah_index(arr, np_idx)
        end = time.process_time()

        print("Process completed in %f seconds" % (end - start))
        assert np.all(np.asarray(out_array)) == np.all(out_array_gt)
