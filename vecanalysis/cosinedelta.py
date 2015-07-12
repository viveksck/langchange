import os
import argparse

from multiprocessing import Process, Lock

import ioutils
import alignment
from vecanalysis.representations.representation_factory import simple_create_representation

START_YEAR = 1900
END_YEAR = 2000
THRESH = 0.0
REP_TYPE = "PPMI"

def get_cosine_deltas(base_embeds, delta_embeds, words):
    deltas = {}
    base_embeds, delta_embeds = alignment.explicit_intersection_align(base_embeds, delta_embeds)
    for word in words:
        if base_embeds.oov(word) or delta_embeds.oov(word):
            deltas[word] = float('nan')
        else:
            delta = base_embeds.represent(word).dot(delta_embeds.represent(word).T)
            if REP_TYPE == "PPMI":
                delta = delta[0,0]
            deltas[word] = delta
    return deltas

def merge(out_pref, tmp_out_pref, years, word_list):
    vol_yearstats = {}
    disp_yearstats = {}
    for word in word_list:
        vol_yearstats[word] = {}
        disp_yearstats[word] = {}
    for year in years:
        vol_yearstat = ioutils.load_pickle(tmp_out_pref + str(year) + "-vols.pkl")
        disp_yearstat = ioutils.load_pickle(tmp_out_pref + str(year) + "-disps.pkl")
        for word in word_list:
            if word not in vol_yearstat:
                vol = float('nan')
            else:
                vol = vol_yearstat[word]
            if word not in disp_yearstat:
                disp = float('nan')
            else:
                disp = disp_yearstat[word]
            vol_yearstats[word][year] = vol
            disp_yearstats[word][year] = disp
        os.remove(tmp_out_pref + str(year) + "-vols.pkl")
        os.remove(tmp_out_pref + str(year) + "-disps.pkl")
    ioutils.write_pickle(vol_yearstats, out_pref + "-vols.pkl")
    ioutils.write_pickle(disp_yearstats, out_pref + "-disps.pkl")

def main(proc_num, lock, out_pref, tmp_out_pref, in_dir, years, word_list, displacement_base, thresh):
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(in_dir + "/volstats/"))
            if tmp_out_pref.split("/")[-1] + str(year) + "-vols.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = tmp_out_pref + str(year) + "-vols.pkl"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break
        
        print proc_num, "Loading matrices..."
        base = simple_create_representation(REP_TYPE, in_dir + str(year-1) + ".bin", restricted_context=word_list[year-1], thresh=thresh)
        delta = simple_create_representation(REP_TYPE, in_dir + str(year) + ".bin", restricted_context=word_list[year], thresh=thresh)
        print proc_num, "Getting deltas..."
        year_vols = get_cosine_deltas(base, delta, word_list[year])
        year_disp = get_cosine_deltas(displacement_base, delta, word_list[year])
        print proc_num, "Writing results..."
        ioutils.write_pickle(year_vols, tmp_out_pref + str(year) + "-vols.pkl")
        ioutils.write_pickle(year_disp, tmp_out_pref + str(year) + "-disps.pkl")

def run_parallel(num_procs, out_pref, tmp_out_pref, in_dir, years, word_list, displacement_base, thresh):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, out_pref, tmp_out_pref, in_dir, years, word_list, displacement_base, thresh]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    full_word_set = set([])
    for year_words in word_list.itervalues():
        full_word_set = full_word_set.union(set(year_words))
    merge(out_pref, tmp_out_pref, years, list(full_word_set))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("dir", help="path to network data (also where output goes)")
    parser.add_argument("word_file", help="path to sorted word file")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--num-words", type=int, help="Number of words (of decreasing average frequency) to include", default=-1)
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="end year (inclusive)", default=END_YEAR)
    parser.add_argument("--thresh", type=float, help="relevance threshold", default=THRESH)
    args = parser.parse_args()
    years = range(args.start_year+1, args.end_year + 1)
    word_lists = ioutils.load_pickle(args.word_file)
    if args.num_words != -1:
        for year in years:
            word_lists[year] = word_lists[year][:args.num_words]
    ioutils.mkdir(args.dir + "/volstats")
    outpref ="/volstats/" + args.word_file.split("/")[-1].split(".")[0] + "-" + str(args.thresh)
    if args.num_words != -1:
        outpref += "-top" + str(args.num_words)
    displacement_base = simple_create_representation(REP_TYPE, args.dir + "/" +  str(args.end_year) + ".bin", restricted_context=word_lists[args.end_year], thresh=args.thresh)
    run_parallel(args.num_procs, args.dir + outpref, args.dir + outpref + "-tmp", args.dir + "/", years, word_lists, displacement_base, args.thresh)       
