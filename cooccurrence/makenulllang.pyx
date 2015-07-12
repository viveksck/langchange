import collections
import random
import math

from cooccurrence.matstore import export_mats_from_dicts

import numpy as np
#cython.wraparound=False
#cython.boundscheck=False
cimport numpy as np

# Sentence length distribution fit to Brown corpus by:
# Sigurd, EEg-Olofsson, van de Weijer "Word Length, Sentence Length
# ...", 2005. Studia Linguistica
# http://onlinelibrary.wiley.com/doi/10.1111/j.0039-3193.2004.00109.x/
SENT_GAMMA_SHAPE = 2
SENT_GAMMA_SCALE = 9.5

cdef tuple make_int_hist(dict year_freqs, dict year_sample_sizes):
    int_hist = collections.defaultdict(float) 
    cdef float total_sum = 0
    cdef float count, year_sum
    for year, year_freq_hist in year_freqs.iteritems():
        year_sum = year_sample_sizes[year]
        for word, freq in year_freq_hist.iteritems():
            count = round(freq * year_sum) 
            int_hist[word] += count 
            total_sum += count
    print total_sum
    return dict(int_hist), total_sum
 
cdef tuple draw_approx_n(dict int_hist, float total_sum, int n):
    chunk = []
    word_is, counts = zip(*int_hist.items())
    probs = np.array(counts)
    print "Total sum:", total_sum, "Sum of counts:", probs.sum()
    probs = probs / (probs.sum() + 1)
    samples = np.random.multinomial(n, probs)
    cdef int i, j
    cdef float sample
    for i in xrange(len(samples)):
        sample = samples[i]
        word_i = word_is[i]
        if sample > counts[i]:
            sample = counts[i]
        int_hist[word_i] -= sample
        for j in xrange(int(sample)):
            chunk.append(word_i)
    print "Chunk length:", len(chunk)
    total_sum -= float(len(chunk))
    return np.random.permutation(np.array(chunk)), int_hist, total_sum

cdef void parse_year_cocrs(np.ndarray[np.long_t, ndim=1] chunk, int start_index, int end_index, int year, year_cocrs):
    end_index = min(len(chunk), end_index)
    cdef size_t i
    for i in xrange(start_index + 2, end_index):
            year_cocrs[year][(chunk[i], chunk[i - 2])] += 1.0
            year_cocrs[year][(chunk[i], chunk[i - 1])] += 1.0

cdef void parse_chunk(np.ndarray[np.long_t, ndim=1] chunk, dict year_sample_sizes, year_cocr, float total_sum):
    rand_years = year_sample_sizes.keys()
    random.shuffle(rand_years)
    start_index = 0
    for year in rand_years:
        size = int(round((year_sample_sizes[year] / total_sum) * len(chunk)))
        end_index = start_index + size
        parse_year_cocrs(chunk, start_index, end_index, year, year_cocr)
        start_index = end_index

cdef np.ndarray make_final_chunk(dict int_hist):
    chunk = []
    for word_i, count in int_hist.iteritems():
        for i in xrange(int(count)):
            chunk.append(word_i)
    return np.random.permutation(np.array(chunk))
        
def make_null_language(year_freqs, year_sample_sizes, output_dir, chunk_size = 10 ** 8):
    print "Making histogram..."
    int_hist, total_sum = make_int_hist(year_freqs, year_sample_sizes)
    year_cocr = collections.defaultdict(lambda : collections.defaultdict(float))
    count = 1
    print "Estimated number of chunks:", total_sum / chunk_size
    cdef float eff_sum = total_sum
    while eff_sum > 5 * chunk_size:
        print "Drawing samples for chunk", count
        rand_chunk, int_hist, new_total_sum = draw_approx_n(int_hist, eff_sum, chunk_size)
        print "Parsing chunk", count
        parse_chunk(rand_chunk, year_sample_sizes, year_cocr, total_sum)
        eff_sum = new_total_sum
        print "Finished chunk", count
        count += 1
    final_chunk = make_final_chunk(int_hist)
    parse_chunk(final_chunk, year_sample_sizes, year_cocr, total_sum)
    export_mats_from_dicts(year_cocr, output_dir)
