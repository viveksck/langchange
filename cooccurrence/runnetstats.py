import argparse
import ioutils

from cooccurrence.indexing import get_word_indices
from cooccurrence.netstats import run_parallel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("dir", help="path to directory with nppmi data and year indexes")
    parser.add_argument("word_file", help="path to sorted word file(s).", default=None)
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--num-words", type=int, help="Number of words (of decreasing average frequency) to include. Must also specifiy word file and index.", default=-1)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=1900)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=2000)
    parser.add_argument("--thresh", type=float, help="optional threshold", default=None)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    word_pickle = ioutils.load_pickle(args.word_file)
    if not args.start_year in word_pickle:
        word_lists = {}
        for year in years:
            word_lists[year] = word_pickle
    else:
        word_lists = word_pickle
    word_infos = {}
    year_indexes = {}
    for year, word_list in word_lists.iteritems():
        year_index = ioutils.load_pickle(args.dir + "/" + str(year) + "-index.pkl") 
        year_indexes[year] = year_index
        if args.num_words != -1:
            word_list = word_list[:args.num_words]
        word_list, word_indices = get_word_indices(word_list, year_index)
        word_infos[year] = (word_list, word_indices)
    outpref ="/netstats/" + args.word_file.split("/")[-1].split(".")[0]
    if args.num_words != -1:
        outpref += "-top" + str(args.num_words)
    if args.thresh != None:
        outpref += "-" + str(args.thresh)
    ioutils.mkdir(args.dir + "/netstats")
    run_parallel(args.num_procs, args.dir + outpref, args.dir + "/", years, year_indexes, word_infos, args.thresh)       
