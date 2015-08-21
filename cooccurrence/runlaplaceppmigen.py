import argparse

import ioutils
from cooccurrence.laplaceppmigen import run_parallel

SMOOTH = 10
START_YEAR = 1900
END_YEAR = 2000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computed laplace smoothed normalized PPMI values.")
    parser.add_argument("out_dir", help="directory where data will be stored")
    parser.add_argument("in_dir", help="path to unmerged data")
    parser.add_argument("word_file", help="file of restricted word set", default=None)
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--conf-dir", help="optional file of restricted word set", default=None)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=END_YEAR)
    parser.add_argument("--num-words", type=int, help="size of vocabulary", default=20000)
    parser.add_argument("--smooth", type=int, help="smoothing factor", default=SMOOTH)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    if args.smooth == 0:
        smooth = 0
    else:
        smooth = 10.0**(-1*float(args.smooth))
    index = ioutils.load_pickle(args.in_dir + "/index.pkl")
    year_index_infos = ioutils.load_year_index_infos_common(index, years, args.word_file, num_words=args.num_words) 
    out_dir = args.out_dir + "/lsmooth" + str(args.smooth) 
    ioutils.mkdir(out_dir)
    run_parallel(args.num_procs,  out_dir + "/", args.in_dir + "/", smooth, year_index_infos, args.conf_dir)       

