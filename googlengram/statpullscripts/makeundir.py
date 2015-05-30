import random
import os

from multiprocessing import Lock, Process

from googlengram import util, matstore

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '5grams_ppmi_lsmooth/'
INPUT_PATH = INPUT_DIR + '{year}.bin'
OUTPUT_DIR = DATA_DIR + 'undir_interesting/'
OUTPUT_PATH = OUTPUT_DIR + 'graph{year}.txt'
MERGED_INDEX = util.load_pickle(INPUT_DIR + "merged_index.pkl")
TARGET_WORD_FILE = DATA_DIR + "info/interestingwords.pkl"
YEARS = range(1850, 2001)

def get_word_indices(word_file):
    common_words = util.load_pickle(word_file) 
    common_indices = set([])
    for i in xrange(len(common_words)):
        common_indices.add(MERGED_INDEX[common_words[i]])
    common_indices = sorted(common_indices)
    merged_list = list(MERGED_INDEX)
    common_words = [merged_list[word_ind] for word_ind in common_indices]
    return common_indices, common_words

def make_year_undir(year):
    print "Getting indices..."
    indices, words = get_word_indices(TARGET_WORD_FILE)
    print "Retrieving matrix..."
    year_mat = matstore.retrieve_cooccurrence_as_coo(INPUT_PATH.format(year=year))
    print "Manipulating matrix..."
    year_mat = year_mat.tocsr()
    year_mat = year_mat[indices, :]
    year_mat = year_mat[:, indices]
    year_mat = year_mat.tocoo()
    print "Writing data..."
    fp = open(OUTPUT_PATH.format(year=year), "w")
    for i in range(year_mat.data.shape[0]):
        if year_mat.row[i] < year_mat.col[i]:
            fp.write(str(year_mat.row[i]) + " " + str(year_mat.col[i]) + "\n")
    fp.close()

def main(proc_num, lock):
    years = range(YEARS[0], YEARS[-1] + 1)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(OUTPUT_DIR))
            if "graph"+str(year)+".txt" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = OUTPUT_PATH.format(year=year)
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break
        make_year_undir(year)


def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
