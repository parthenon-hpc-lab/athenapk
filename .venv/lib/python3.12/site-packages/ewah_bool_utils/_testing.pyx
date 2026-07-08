# Cython interface for ewah_bool_utils tests

cimport cython
cimport numpy as np
from libcpp.vector cimport vector

from .ewah_bool_array cimport ewah_bool_array

import numpy as np


cdef np.uint64_t FLAG = ~(<np.uint64_t>0)

ctypedef fused dtype_t:
    np.float32_t
    np.float64_t

ctypedef fused int_t:
    np.int32_t
    np.int64_t


cdef class Index:
    cdef void* ewah_array

    def __cinit__(self):
        cdef ewah_bool_array *ewah_array = new ewah_bool_array()
        self.ewah_array = <void *> ewah_array

    cdef bint _get(self, np.uint64_t i1):
        cdef ewah_bool_array *ewah_array = <ewah_bool_array *> self.ewah_array
        return ewah_array[0].get(i1)

    def get(self, i1):
        return self._get(i1)

    cdef void _set(self, np.uint64_t i1):
        cdef ewah_bool_array *ewah_array = <ewah_bool_array *> self.ewah_array
        ewah_array[0].set(i1)

    def set(self, i1):
        return self._set(i1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def select(self, dtype_t[:] np_array):
        cdef ewah_bool_array *ewah_array = <ewah_bool_array *> self.ewah_array
        cdef int i, j
        cdef np.ndarray[dtype_t, ndim=1] out

        out = np.empty(ewah_array[0].numberOfOnes())
        j = 0
        for i in range(ewah_array[0].sizeInBits()):
            if ewah_array[0].get(i):
                out[j] = np_array[i]
                j += 1

        return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    def set_from(self, int_t[:] ids):
        cdef ewah_bool_array *ewah_array = <ewah_bool_array *> self.ewah_array
        cdef np.uint64_t i
        cdef np.uint64_t last = 0

        for i in range(ids.shape[0]):
            if ids[i] < last:
                raise RuntimeError
            self._set(ids[i])

        print('Set from %s array and ended up with %s bytes' % (
            ids.size, ewah_array[0].sizeInBytes()))


cpdef np.uint64_t[:] ewah_set_and_unset(int_t[:] arr):
    cdef ewah_bool_array *ewah_arr
    cdef vector[size_t] vec

    ewah_arr = new ewah_bool_array()

    for i2 in range(arr.shape[0]):
        i1 = arr[i2]
        ewah_arr[0].set(i1)

    vec = ewah_arr[0].toArray()
    np_arr = np.array(vec, 'uint64')

    return np_arr

cpdef int find_ewah_collisions(int_t[:] arr1, int_t[:] arr2):
    cdef ewah_bool_array *ewah_arr1
    cdef ewah_bool_array *ewah_arr2
    cdef ewah_bool_array *ewah_arr_keys
    cdef int ncoll

    ewah_arr1 = new ewah_bool_array()
    ewah_arr2 = new ewah_bool_array()
    ewah_arr_keys = new ewah_bool_array()

    for i2 in range(arr1.shape[0]):
        i1 = arr1[i2]
        ewah_arr1[0].set(i1)

    for j2 in range(arr2.shape[0]):
        j1 = arr2[j2]
        ewah_arr2[0].set(j1)

    ewah_arr2[0].logicaland(ewah_arr1[0], ewah_arr_keys[0])
    ncoll = ewah_arr_keys[0].numberOfOnes()

    return ncoll

cpdef dtype_t[:] make_and_select_from_ewah_index(dtype_t[:] arr,
      int_t[:] np_idx):
      cdef Index idx = Index()

      idx.set_from(np_idx)
      out_arr = idx.select(arr)
      return out_arr
