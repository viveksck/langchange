from scipy.sparse import coo_matrix

import numpy as np
cimport numpy as np
import cython
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def bootstrap_count_mat(count_mat, num_samples=10**9):
    cdef np.ndarray[np.float64_t, ndim=1] bootstrap_data = np.zeros((count_mat.data.shape[0],), dtype=np.float64) 
    cdef int i, n
    cdef float prob, r  = 1.0
    count_mat = count_mat / count_mat.sum()
    n = num_samples
    print "Making permutation order..."
    cdef np.ndarray[np.int64_t, ndim=1] perm = np.random.permutation(count_mat.data.shape[0])
    print "Drawing data..."
    for i in perm:
        draw = np.random.binomial(n, count_mat.data[i] / r)
        bootstrap_data[i] = draw
        n -= draw
        r -= count_mat.data[i] 
    print "Making matrix..."
    return coo_matrix((bootstrap_data, (count_mat.row, count_mat.col))) 
