import random
import os
import argparse
from multiprocessing import Process, Lock

import ioutils
from cooccurrence import matstore
from cooccurrence.indexing import get_word_indices


START_YEAR = 1900
END_YEAR = 2000

def compute_word_stats(mat, word, index):
    word_i = index[word]
    if word_i >= mat.shape[0]:
        return 0
    return mat[word_i, :].sum()

def merge(word_list, years, in_dir, out_file):
    yearstats = {}
    for word in word_list:
        yearstats[word] = {}
    for year in years:
        yearstat = ioutils.load_pickle(in_dir + str(year) + "-freqstmp.pkl")
        for word in word_list:
            yearstats[word][year] = yearstat[word]
        os.remove(in_dir + str(year) + "-freqstmp.pkl")
    ioutils.write_pickle(yearstats, out_file)

def main(proc_num, lock, in_dir, years, word_list, index):
    years = range(years[0], years[-1] + 1)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(in_dir))
            if str(year) + "-freqstmp.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = in_dir + str(year) + "-freqstmp.pkl"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break

        print proc_num, "Retrieving mat for year", year
        mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin")
        print proc_num, "Making inverse freq mat", year
        mat = mat.tocsr()
        mat = mat / mat.sum()
        word_stats = {}
        print proc_num, "Getting stats for year", year
        for word in word_list:
            word_stats[word] = compute_word_stats(mat, word, index)

        print proc_num, "Writing stats for year", year
        ioutils.write_pickle(word_stats, in_dir + str(year) + "-freqstmp.pkl")


def run_parallel(num_procs, in_dir, years, word_list, index, out_file):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, in_dir, years, word_list, index]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge(word_list, years, in_dir, out_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("out_file", help="path to network data (also where output goes)")
    parser.add_argument("in_dir", help="path to network data (also where output goes)")
    parser.add_argument("word_file", help="path to sorted word file")
    parser.add_argument("index_file", help="path to sorted word file")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="end year (inclusive)", default=END_YEAR)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    index = ioutils.load_pickle(args.index_file)
    word_list = ioutils.load_pickle(args.word_file)
    word_list, _ = get_word_indices(word_list, index)
    run_parallel(args.num_procs, args.in_dir + "/", years, word_list, index, args.out_file)       
