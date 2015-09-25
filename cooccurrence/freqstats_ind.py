import random
import os
import argparse
from multiprocessing import Process, Lock

import ioutils

START_YEAR = 1900
END_YEAR = 2000

def merge(word_list, years, in_dir, out_file):
    yearstats = {}
    for word in word_list:
        yearstats[word] = {}
    for year in years:
        yearstat = ioutils.load_pickle(in_dir + str(year) + "-freqstmp.pkl")
        for word in yearstat.keys():
            yearstats[word][year] = yearstat[word]
        os.remove(in_dir + str(year) + "-freqstmp.pkl")
    ioutils.write_pickle(yearstats, out_file)

def main(proc_num, lock, in_dir, years, word_list):
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
        
        year_freqs = ioutils.load_pickle(in_dir + "/" + str(year) + "-freqs.pkl")
        word_stats = {}
        print proc_num, "Getting stats for year", year
        sum = 0
        for word in word_list:
            if word in year_freqs:
                word_count = year_freqs[word][1]
                sum += word_count
                word_stats[word] = word_count
        for word in word_stats:
            word_stats[word] /= float(sum)

        print proc_num, "Writing stats for year", year
        ioutils.write_pickle(word_stats, in_dir + str(year) + "-freqstmp.pkl")


def run_parallel(num_procs, in_dir, years, word_list, out_file):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, in_dir, years, word_list]) for i in range(num_procs)]
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
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="end year (inclusive)", default=END_YEAR)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    word_list = ioutils.load_pickle(args.word_file)
    run_parallel(args.num_procs, args.in_dir + "/", years, word_list, args.out_file)       
