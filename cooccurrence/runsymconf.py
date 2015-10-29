import argparse
import numpy as np

import ioutils
from cooccurrence.symconf import run_parallel

ALPHA = 0.001
START_YEAR = 1900
END_YEAR = 2000
NUM_WORDS = 20000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates binary matrix full of entries that are signicantly correlated.")
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
    index = ioutils.load_pickle(args.in_dir + "/index.pkl")
    year_index_infos = ioutils.load_year_index_infos_common(index, years, args.word_file, num_words=args.num_words) 
    out_dir = args.out_dir + "/alpha" + str(args.alpha) 
    ioutils.mkdir(out_dir)
    run_parallel(args.num_procs,  out_dir + "/", args.in_dir + "/", args.alpha, eff_sample_size, year_index_infos, args.fwer_control)       
