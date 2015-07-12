import argparse
import os
import numpy as np

import ioutils
from cooccurrence.laplaceppmigen import run_parallel
from cooccurrence.indexing import get_word_indices

INDEX_FILE = "/dfs/scratch0/googlengrams/2012-eng-fic/5grams/merged_index.pkl"

SMOOTH = 9
THRESH = 8
START_YEAR = 1900
END_YEAR = 2000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("out_dir", help="directory where data will be stored")
    parser.add_argument("in_dir", help="path to unmerged data")
    parser.add_argument("sample_file", help="path to file with sample sizes")
    parser.add_argument("word_file", help="file of restricted word set", default=None)
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--conf-dir", help="optional file of restricted word set", default=None)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=END_YEAR)
    parser.add_argument("--num-words", type=int, help="size of vocabulary", default=30000)
    parser.add_argument("--smooth", type=int, help="smoothing factor", default=SMOOTH)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    if args.smooth == 0:
        smooth = 0
    else:
        smooth = 10.0**(-1*float(args.smooth))
    word_pickle = ioutils.load_pickle(args.word_file)
    index = ioutils.load_pickle(INDEX_FILE)
    word_info = {}
    if not args.start_year in word_pickle:
        word_pickle = word_pickle[:args.num_words]
        year_word_info = get_word_indices(word_pickle, index) 
        for year in years:
            word_info[year] = year_word_info
    else:
        for year in years:
            word_info[year] = get_word_indices(word_pickle[year][:args.num_words], index) 

    sample_sizes = ioutils.load_pickle(args.sample_file)
    eff_sample_size = (np.array(sample_sizes.values())).min()
    out_dir = args.out_dir + "/lsmooth" + str(args.smooth) 
    ioutils.mkdir(out_dir)
    run_parallel(args.num_procs,  out_dir + "/", args.in_dir + "/", years, smooth, word_info, args.conf_dir, eff_sample_size)       

