import random
import os
from multiprocessing import Process, Lock
from scipy.sparse import coo_matrix

from googlengram import matstore, util
from googlengram.statpullscripts.laplaceppmigen import make_ppmi_mat
from googlengram.statpullscripts.bootstrapping import bootstrap_count_mat

import numpy as np
cimport numpy as np

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_merged/'
OUTPUT_PREFIX = DATA_DIR + "/stats/bootstraptest-"
TMP_DIR = '/lfs/madmax4/0/will/google_ngrams/tmp/'
WORD_FILE = DATA_DIR + "info/interestingwords.pkl"

def get_word_indices(word_list):
    common_indices = [MERGED_INDEX[word] for word in word_list]
    common_indices = sorted(common_indices)
    return np.array(common_indices)

WORDS = util.load_pickle(WORD_FILE) 
MERGED_INDEX = util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl")
CONTEXT_INDICES = get_word_indices(WORDS)
YEARS = range(2000, 2001)

def compute_word_stats(mat, word, context_indices):
    word_i = MERGED_INDEX[word]
    if word_i >= mat.shape[0]:
        return -1, -1, -1, -1
    vec = mat[word_i, :]
    indices = vec.nonzero()[1]
    indices = np.intersect1d(indices, context_indices, assume_unique=True)
    if len(indices) < 1:
        return 0, 0, 0, 0
    vec = vec[:, indices]
    weights = vec/vec.sum()
    reduced = mat[indices, :]
    reduced = reduced[:, indices]
    weighted = (weights * reduced).sum() / (float(len(indices)) - 1)
    binary = float(reduced.nnz) / (len(indices) * (len(indices) - 1)) 
    deg = vec.nnz
    sum = vec.sum()
    return (weighted, binary, deg, sum)

def merge():
    binary_yearstats = {}
    weighted_yearstats = {}
    deg_yearstats = {}
    sum_yearstats = {}
    for word in WORDS:
        binary_yearstats[word] = {}
        weighted_yearstats[word] = {}
        deg_yearstats[word] = {}
        sum_yearstats[word] = {}
    for year in YEARS:
        binary_yearstat = util.load_pickle(TMP_DIR + str(year) + "-binary.pkl")
        weighted_yearstat = util.load_pickle(TMP_DIR + str(year) + "-weighted.pkl")
        deg_yearstat = util.load_pickle(TMP_DIR + str(year) + "-deg.pkl")
        sum_yearstat = util.load_pickle(TMP_DIR + str(year) + "-sum.pkl")
        for word in WORDS:
            binary_yearstats[word][year] = binary_yearstat[word]
            weighted_yearstats[word][year] = weighted_yearstat[word]
            deg_yearstats[word][year] = deg_yearstat[word]
            sum_yearstats[word][year] = sum_yearstat[word]
        os.remove(TMP_DIR + str(year) + "-binary.pkl")
        os.remove(TMP_DIR + str(year) + "-weighted.pkl")
        os.remove(TMP_DIR + str(year) + "-deg.pkl")
        os.remove(TMP_DIR + str(year) + "-sum.pkl")
    util.write_pickle(binary_yearstats, OUTPUT_PREFIX + "-binary.pkl")
    util.write_pickle(weighted_yearstats, OUTPUT_PREFIX + "-weighted.pkl")
    util.write_pickle(deg_yearstats, OUTPUT_PREFIX + "-deg.pkl")
    util.write_pickle(sum_yearstats, OUTPUT_PREFIX + "-sum.pkl")

def main(proc_num, lock):
    years = range(YEARS[0], YEARS[-1] + 1)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(TMP_DIR))
            if str(year) + "-binary.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = TMP_DIR + str(year) + "-binary.pkl"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break

        print proc_num, "Retrieving mat for year", year
        mat = matstore.retrieve_cooccurrence_as_coo(INPUT_DIR + str(year) + ".bin")
        print proc_num, "Bootstrapping year mat", year
        mat = bootstrap_count_mat(mat)
        row_d, col_d, data_d = make_ppmi_mat(mat)
        mat = coo_matrix(data_d, (row_d, col_d), dtype=np.float64)
        mat.setdiag(0)
        context_indices = CONTEXT_INDICES[CONTEXT_INDICES < min(mat.shape[1], mat.shape[0])]
        mat = mat.tocsr()
        mat.eliminate_zeros()
        weighted_word_stats = {}
        binary_word_stats = {}
        deg_word_stats = {}
        sum_word_stats = {}
        print proc_num, "Getting stats for year", year
        for word in WORDS:
            weighted, binary, deg, sum = compute_word_stats(mat, word, context_indices)
            weighted_word_stats[word] = weighted
            binary_word_stats[word] = binary
            deg_word_stats[word] = deg
            sum_word_stats[word] = sum

        print proc_num, "Writing stats for year", year
        util.write_pickle(weighted_word_stats, TMP_DIR + str(year) + "-weighted.pkl")
        util.write_pickle(binary_word_stats, TMP_DIR + str(year) + "-binary.pkl")
        util.write_pickle(deg_word_stats, TMP_DIR + str(year) + "-deg.pkl")
        util.write_pickle(sum_word_stats, TMP_DIR + str(year) + "-sum.pkl")

def run_parallel(num_procs, out_pref, in_dir, years):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("out_pref", help="output prefix")
    parser.add_argument("in_dir", help="path to unmerged data")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=END_YEAR)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    smooth = 10.0**(float(args.smooth))
    run_parallel(args.num_procs, args.out_pref, args.in_dir, years)       

