import argparse
import numpy as np

import ioutils
from cooccurrence.symconf import run_parallel
from cooccurrence.indexing import get_word_indices

ALPHA = 0.001
START_YEAR = 1900
END_YEAR = 2000
NUM_WORDS = 30000

INDEX_FILE = "/dfs/scratch0/googlengrams/2012-eng-fic/5grams/merged_index.pkl"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("out_dir", help="directory where data will be stored")
    parser.add_argument("in_dir", help="path to unmerged data")
    parser.add_argument("sample_file", help="path to file with sample sizes")
    parser.add_argument("word_file", help="optional file of restricted word set", default=None)
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--num-words", type=int, help="start year (inclusive)", default=NUM_WORDS)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=END_YEAR)
    parser.add_argument("--alpha", type=float, help="confidence threshold", default=ALPHA)
    parser.add_argument("--fwer-control", action='store_true', help="control for simultaneous testing")
    args = parser.parse_args()
    sample_sizes = ioutils.load_pickle(args.sample_file)
    eff_sample_size = np.percentile(np.array(sample_sizes.values()), 10)
    years = range(args.start_year, args.end_year + 1)
    word_pickle = ioutils.load_pickle(args.word_file)
    index = ioutils.load_pickle(INDEX_FILE)
    word_pickle = ioutils.load_pickle(args.word_file)
    word_info = {}
    if not args.start_year in word_pickle:
        word_pickle = word_pickle[:args.num_words]
        year_word_info = get_word_indices(word_pickle, index)[1] 
        for year in years:
            word_info[year] = year_word_info
    else:
        for year in years:
            word_info[year] = get_word_indices(word_pickle[year][:args.num_words], index)[1]
    out_dir = args.out_dir + "/alpha" + str(args.alpha) 
    ioutils.mkdir(out_dir)
    run_parallel(args.num_procs,  out_dir + "/", args.in_dir + "/", years, args.alpha, eff_sample_size, word_info, args.fwer_control)       

