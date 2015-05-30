from libc.stdio cimport FILE, fopen, fwrite, fclose, feof, fread
import collections
import util
import os
from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np

NGRAM_DIM = 739547

def export_cooccurrence(year_counts, output_dir):
    cdef FILE* fout
    cdef int word1
    cdef int word2
    cdef double val
    cdef char* fn
    for year, counts in year_counts.iteritems():
        filename = output_dir + str(year) + ".bin"
        fn = filename
        fout = fopen(fn, 'w')
        for (i, c), v in counts.iteritems():
            word1 = i
            word2  = c
            val = v
            fwrite(&word1, sizeof(int), 1, fout) 
            fwrite(&word2, sizeof(int), 1, fout) 
            fwrite(&val, sizeof(double), 1, fout) 
        fclose(fout)

def export_cooccurrence_eff(row_d, col_d, data_d, year, output_dir):
    cdef FILE* fout
    cdef int word1
    cdef int word2
    cdef double val
    cdef char* fn
    cdef int i
    filename = output_dir + str(year) + ".bin"
    fn = filename
    fout = fopen(fn, 'w')
    for i in xrange(len(row_d)):
        word1 = row_d[i]
        word2  = col_d[i]
        val = data_d[i]
        if val == 0:
            continue
        fwrite(&word1, sizeof(int), 1, fout) 
        fwrite(&word2, sizeof(int), 1, fout) 
        fwrite(&val, sizeof(double), 1, fout) 
    fclose(fout)

def retrieve_cooccurrence(filename):
    cdef FILE* fin
    cdef int word1
    cdef int word2
    cdef double val
    cdef char* fn
    year_count = collections.defaultdict(int)
    fn = filename
    fin = fopen(fn, 'r')
    while not feof(fin):
        fread(&word1, sizeof(int), 1, fin) 
        fread(&word2, sizeof(int), 1, fin) 
        fread(&val, sizeof(double), 1, fin) 
        year_count[(word1, word2)] = val
    fclose(fin)
    return year_count

def retrieve_cooccurrence_as_coo(matfn):
    cdef FILE* fin
    cdef int word1, word2, ret
    cdef double val
    cdef char* fn
    fn = matfn
    fin = fopen(fn, 'r')
    cdef int size = (os.path.getsize(matfn) / 16) + 1
    cdef np.ndarray[np.int32_t, ndim=1] row = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] col = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] data = np.empty(size, dtype=np.float64)
    cdef int i = 0
    while not feof(fin):
        fread(&word1, sizeof(int), 1, fin) 
        fread(&word2, sizeof(int), 1, fin) 
        ret = fread(&val, sizeof(double), 1, fin) 
        if ret != 1:
            break
        row[i] = word1
        col[i] = word2
        data[i] = val
        i += 1
    fclose(fin)
    data[-1] = 0
    row[-1] = NGRAM_DIM
    col[-1] = NGRAM_DIM
    return coo_matrix((data, (row, col)), dtype=np.float64)

def retrieve_cooccurrence_as_coo_thresh(matfn, thresh):
    cdef FILE* fin
    cdef int word1, word2, ret
    cdef double val
    cdef char* fn
    fn = matfn
    fin = fopen(fn, 'r')
    cdef int size = (os.path.getsize(matfn) / 16) + 1
    cdef np.ndarray[np.int32_t, ndim=1] row = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] col = np.empty(size, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] data = np.empty(size, dtype=np.float64)
    cdef int i = 0
    while not feof(fin):
        fread(&word1, sizeof(int), 1, fin) 
        fread(&word2, sizeof(int), 1, fin) 
        ret = fread(&val, sizeof(double), 1, fin) 
        if ret != 1:
            break
        if val < thresh:
            continue
        row[i] = word1
        col[i] = word2
        data[i] = val
        i += 1
    fclose(fin)
    data[-1] = 0
    row[-1] = NGRAM_DIM
    col[-1] = NGRAM_DIM
    return coo_matrix((data, (row, col)), dtype=np.float64)
