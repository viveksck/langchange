import argparse
import ioutils

import numpy as np

from cooccurrence.bootstrapstats import run_parallel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("dir", help="path to directory with count data and index")
    parser.add_argument("word_file", help="path to sorted word file(s).", default=None)
    parser.add_argument("sample_file", help="path to file with sample sizes.", default=None)
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--num-words", type=int, help="Number of words (of decreasing average frequency) to include. Must also specifiy word file and index.", default=-1)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=1900)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=2000)
    parser.add_argument("--num-boots", type=int, help="Number of bootstrap samples", default=10)
    parser.add_argument("--smooth", type=int, help="laplace smoothing factor", default=10)
    parser.add_argument("--alpha", type=float, help="confidence threshold for edges", default=0.05)
    parser.add_argument("--fwer-control", action='store_true', help="use Bonferroni")
    parser.add_argument("--id", type=int, help="run id", default=0)
    args = parser.parse_args()
    sample_sizes = ioutils.load_pickle(args.sample_file)
    eff_sample_size = np.percentile(np.array(sample_sizes.values()), 10)
    if args.smooth == 0:
        smooth = 0
    else:
        smooth = 10.0**(-1*float(args.smooth))
    years = range(args.start_year, args.end_year + 1)
    index = ioutils.load_pickle(args.dir + "/index.pkl")
    year_index_infos = ioutils.load_year_index_infos_common(index, years, args.word_file, num_words=args.num_words) 
    outpref = "/bootstats-" + str(args.alpha) + "-" + str(args.fwer_control) + "/" +  args.word_file.split("/")[-1].split(".")[0]
    if args.num_words != -1:
        outpref += "-top" + str(args.num_words)
    ioutils.mkdir(args.dir + "/" + outpref.split("/")[1])
    run_parallel(args.num_procs, args.dir + outpref, args.dir + "/", year_index_infos, args.num_boots, smooth, eff_sample_size, args.alpha, args.fwer_control, args.id)       
