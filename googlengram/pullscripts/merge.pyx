import collections
import matstore
import util
import os
from multiprocessing import Process, Lock

INPUT_DIR = '/lfs/madmax5/0/will/google_ngrams/unmerged/eng-all/20090715/5gram/'
OUTPUT_DIR = '/dfs/scratch0/google_ngrams/year_counts/'
YEARS = range(1700, 2009)
NUM_CHUNKS = 800

def main(proc_num, lock):
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in YEARS:
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

        print proc_num, "Merging counts for year", year
        full_counts = collections.defaultdict(float)
        merged_index = collections.OrderedDict()
        for chunk_num in range(NUM_CHUNKS): 
            chunk_name = INPUT_DIR + str(chunk_num) + "/" + str(year) + ".bin"
            if not os.path.isfile(chunk_name):
                continue
            chunk_counts = matstore.retrieve_cooccurrence(chunk_name)
            chunk_index = util.load_pickle(INPUT_DIR + str(chunk_num) + "/index.pkl") 
            chunk_index = list(chunk_index)
            for pair, count in chunk_counts.iteritems():
                i_word = chunk_index[pair[0]]
                c_word = chunk_index[pair[1]]
                new_pair = (util.word_to_cached_id(i_word, merged_index), 
                        util.word_to_cached_id(c_word, merged_index))
                full_counts[new_pair] += count
        
        print proc_num, "Writing counts for year", year
        matstore.export_cooccurrence({str(year) : full_counts}, OUTPUT_DIR)
        util.write_pickle(merged_index, OUTPUT_DIR + str(year) + "-index.pkl")

def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()   


