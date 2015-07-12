import os
import argparse

from multiprocessing import Process, Lock

import ioutils
from cooccurrence import matstore
from cooccurrence.indexing import get_word_indices

START_YEAR = 1900
END_YEAR = 2000
THRESH = 0.0
REP_TYPE = "PPMI"
MIN_SIZE = 250000

def get_jaccard_deltas(base_embeds, delta_embeds, word_list, word_indices):
    deltas = {}
    for i in xrange(len(word_list)):
        word_i = word_indices[i]
        base_vec = base_embeds[word_i,:]
        base_vec = base_vec[:, word_indices]
        delta_vec = delta_embeds[word_i,:]
        delta_vec = delta_vec[:, word_indices]
        intersect = delta_vec.dot(base_vec.T)[0,0] 
        base_size = base_vec.dot(base_vec.T)[0,0]
        delta_size = delta_vec.dot(delta_vec.T)[0,0]
        if base_size == 0 and delta_size == 0:
            deltas[word_list[i]] = float('nan')
        else:
            deltas[word_list[i]] = intersect / (base_size + delta_size - intersect)
    return deltas

def merge(out_pref, tmp_out_pref, years, word_list):
    vol_yearstats = {}
    disp_yearstats = {}
    for word in word_list:
        vol_yearstats[word] = {}
        disp_yearstats[word] = {}
    for year in years:
        vol_yearstat = ioutils.load_pickle(tmp_out_pref + str(year) + "-jvols.pkl")
        disp_yearstat = ioutils.load_pickle(tmp_out_pref + str(year) + "-jdisps.pkl")
        for word in word_list:
            vol_yearstats[word][year] = vol_yearstat[word]
            disp_yearstats[word][year] = disp_yearstat[word]
        os.remove(tmp_out_pref + str(year) + "-jvols.pkl")
        os.remove(tmp_out_pref + str(year) + "-jdisps.pkl")
    ioutils.write_pickle(vol_yearstats, out_pref + "-jvols.pkl")
    ioutils.write_pickle(disp_yearstats, out_pref + "-jdisps.pkl")

def main(proc_num, lock, out_pref, tmp_out_pref, in_dir, years, word_list, word_indices, displacement_base, thresh):
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(in_dir + "/volstats/"))
            if tmp_out_pref.split("/")[-1] + str(year) + "-jvols.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = tmp_out_pref + str(year) + "-jvols.pkl"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break
        
        print proc_num, "Loading matrices..."
        base = matstore.retrieve_mat_as_binary_coo_thresh(in_dir + "/" + str(year - 1) + ".bin", args.thresh, min_size=MIN_SIZE)
        base = base.tocsr()
        delta = matstore.retrieve_mat_as_binary_coo_thresh(in_dir + "/" + str(year) + ".bin", args.thresh, min_size=MIN_SIZE)
        delta = delta.tocsr()
        print proc_num, "Getting deltas..."
        year_vols = get_jaccard_deltas(base, delta, word_list, word_indices)
        year_disp = get_jaccard_deltas(displacement_base, delta, word_list, word_indices)
        print proc_num, "Writing results..."
        ioutils.write_pickle(year_vols, tmp_out_pref + str(year) + "-jvols.pkl")
        ioutils.write_pickle(year_disp, tmp_out_pref + str(year) + "-jdisps.pkl")

def run_parallel(num_procs, out_pref, tmp_out_pref, in_dir, years, word_list, word_indices, displacement_base, thresh):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, out_pref, tmp_out_pref, in_dir, years, word_list, word_indices, displacement_base, thresh]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge(out_pref, tmp_out_pref, years, word_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("dir", help="path to network data (also where output goes)")
    parser.add_argument("word_file", help="path to sorted word file")
    parser.add_argument("index_file", help="path to word index file")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--num-words", type=int, help="Number of words (of decreasing average frequency) to include", default=-1)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="end year (inclusive)", default=END_YEAR)
    parser.add_argument("--thresh", type=float, help="relevance threshold", default=THRESH)
    args = parser.parse_args()
    years = range(args.start_year+1, args.end_year + 1)
    word_list = ioutils.load_pickle(args.word_file)
    index = ioutils.load_pickle(args.index_file)
    if args.num_words != -1:
        word_list = word_list[:args.num_words]
    ioutils.mkdir(args.dir + "/volstats")
    word_list, word_indices = get_word_indices(word_list, index)
    outpref ="/volstats/" + args.word_file.split("/")[-1].split(".")[0] + "-" + str(args.thresh)
    if args.num_words != -1:
        outpref += "-top" + str(args.num_words)
    displacement_base = matstore.retrieve_mat_as_binary_coo_thresh(args.dir + "/" + str(args.end_year) + ".bin", args.thresh, min_size=MIN_SIZE)
    displacement_base = displacement_base.tocsr()
    run_parallel(args.num_procs, args.dir + outpref, args.dir + outpref + "-tmp", args.dir + "/", years, word_list, word_indices, displacement_base, args.thresh)       
