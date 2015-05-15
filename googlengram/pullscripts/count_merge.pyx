import random
import urllib2
import re
import os
import subprocess
import collections
import matstore
import util
from multiprocessing import Process, Lock

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/5grams_fixed/'
INDEX_DIR = DATA_DIR + '/5grams_fixed/'
OUTPUT_DIR = DATA_DIR + '/5grams_merged/'

def main(proc_num, lock):
    merged_index = util.load_pickle(INDEX_DIR + "merged_index.pkl") 
    years = range(1700, 2009)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(OUTPUT_DIR))
            if str(year) + ".bin" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = OUTPUT_DIR + str(year) + ".bin"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break

        print proc_num, "Fixing counts for year", year
        fixed_counts = {}
        old_mat = matstore.retrieve_cooccurrence(INPUT_DIR + str(year) + ".bin")
        old_index = util.load_pickle(INDEX_DIR + str(year) + "-list.pkl") 
        for pair, count in old_mat.iteritems():
            i_word = old_index[pair[0]]
            c_word = old_index[pair[1]]
            new_pair = (util.word_to_static_id(i_word, merged_index), 
                    util.word_to_static_id(c_word, merged_index))
            fixed_counts[new_pair] = count
        
        print proc_num, "Writing counts for year", year
        matstore.export_cooccurrence({str(year) : fixed_counts}, OUTPUT_DIR)

def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()   
