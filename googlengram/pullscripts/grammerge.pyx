import collections
import matstore
import util
import os
import string
import random
from multiprocessing import Process, Lock

INPUT_DIR = '/dfs/scratch0/google_ngrams/byword5grams_raw/eng-all/20090715/5gram/'
OUTPUT_DIR = '/dfs/scratch0/google_ngrams/byword5grams/'
YEARS = range(1700, 2009)
NUM_CHUNKS = 800

def main(proc_num, lock):
    years = YEARS
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

        print proc_num, "Merging grams for year", year
        year_grams = {}
        for letter in string.ascii_lowercase:
            year_grams[letter] = collections.defaultdict(list)
        for chunk_num in range(NUM_CHUNKS): 
            chunk_name = INPUT_DIR + str(chunk_num) + "/" + str(year) + ".pkl"
            if not os.path.isfile(chunk_name):
                continue
            chunk_counts = util.load_pickle(chunk_name)
            for word, info_list in chunk_counts.iteritems():
                if word[0] not in year_grams:
                    continue
                for info in info_list:
                    gram_info = info[0]
                    gram_info = gram_info.split("\t")
                    if len(gram_info[0].split()) != 5:
                        continue
                    year_grams[word[0]][word].append((gram_info[0], info[1]))
            
        print proc_num, "Writing counts for year", year
        for letter, letter_grams in year_grams.iteritems():
             util.write_pickle(letter_grams, OUTPUT_DIR + str(year) + "-" + letter + ".pkl")

def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()   


