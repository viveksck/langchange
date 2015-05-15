import os

from multiprocessing import Process, Lock

from googlengram import util
from representations.explicit import Explicit

DATA_DIR = "/dfs/scratch0/google_ngrams/"
PPMI_DIR = DATA_DIR + "5grams_ppmi_smooth/"
TMP_DIR = '/lfs/madmax5/0/will/google_ngrams/tmp/'
OUTPUT_PREFIX = DATA_DIR + "info/ppmi_cosine_f1900"

YEARS = range(1901, 2009)
CONTEXT_WORDS = context_words = util.load_pickle("/dfs/scratch0/google_ngrams/info/relevantwords.pkl")
WORDS = util.load_pickle("/dfs/scratch0/google_ngrams/info/interestingwords.pkl")
DISPLACEMENT_BASE = Explicit.load(PPMI_DIR + "1900.bin", restricted_context=CONTEXT_WORDS)

def get_cosine_deltas(base_embeds, delta_embeds, words):
    deltas = {}
    for word in words:
        if base_embeds.oov(word) or delta_embeds.oov(word):
            deltas[word] = float('nan')
        else:
            deltas[word] = base_embeds.represent(word).dot(delta_embeds.represent(word).T)[0, 0]
    return deltas

def merge():
    vol_yearstats = {}
    disp_yearstats = {}
    for word in WORDS:
        vol_yearstats[word] = {}
        disp_yearstats[word] = {}
    for year in YEARS:
        vol_yearstat = util.load_pickle(TMP_DIR + str(year) + "-vols.pkl")
        disp_yearstat = util.load_pickle(TMP_DIR + str(year) + "-disps.pkl")
        for word in WORDS:
            vol_yearstats[word][year] = vol_yearstat[word]
            disp_yearstats[word][year] = disp_yearstat[word]
        os.remove(TMP_DIR + str(year) + "-vols.pkl")
        os.remove(TMP_DIR + str(year) + "-disps.pkl")
    util.write_pickle(vol_yearstats, OUTPUT_PREFIX + "-vols.pkl")
    util.write_pickle(disp_yearstats, OUTPUT_PREFIX + "-disps.pkl")

def main(proc_num, lock):
    while True:
        lock.acquire()
        work_left = False
        for year in YEARS:
            dirs = set(os.listdir(TMP_DIR))
            if str(year) + "-vols.pkl" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = TMP_DIR + str(year) + "-vols.pkl"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break
        
        print proc_num, "Loading matrices..."
        base = Explicit.load(PPMI_DIR + str(year - 1) + ".bin", restricted_context=CONTEXT_WORDS)
        delta = Explicit.load(PPMI_DIR + str(year) + ".bin", restricted_context=CONTEXT_WORDS)
        print proc_num, "Getting deltas..."
        year_vols = get_cosine_deltas(base, delta, WORDS)
        year_disp = get_cosine_deltas(DISPLACEMENT_BASE, delta, WORDS)
        print proc_num, "Writing results..."
        util.write_pickle(year_vols, TMP_DIR + str(year) + "-vols.pkl")
        util.write_pickle(year_disp, TMP_DIR + str(year) + "-disps.pkl")

def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print "Merging"
    merge()
