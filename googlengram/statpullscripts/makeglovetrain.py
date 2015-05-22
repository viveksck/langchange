import random
import os

from multiprocessing import Lock, Process
from itertools import izip

from googlengram import util, matstore

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '5grams_merged/'
INPUT_PATH = INPUT_DIR + '{year}.bin'
OUTPUT_DIR = DATA_DIR + 'modglove_train_smallrel/'
MERGED_INDEX = util.load_pickle(INPUT_DIR + "merged_index.pkl")
TARGET_WORD_FILE = DATA_DIR + "info/relevantwords-100000.pkl"
YEARS = range(1850, 2009)

def get_word_indices(word_file):
    common_words = util.load_pickle(word_file) 
    common_indices = set([])
    for i in xrange(len(common_words)):
        common_indices.add(MERGED_INDEX[common_words[i]])
    common_indices = sorted(common_indices)
    merged_list = list(MERGED_INDEX)
    common_words = [merged_list[word_ind] for word_ind in common_indices]
    return common_indices, common_words

def write_vocab(words, freq_counts, filename):
    fp = open(filename, "w")
    for word, count in izip(words, freq_counts.tolist()):
        fp.write(word.encode('utf-8') + " " + str(count[0]) + "\n")
    fp.close()

def make_year_train(year):
    print "Getting indices..."
    indices, words = get_word_indices(TARGET_WORD_FILE)
    print "Retrieving matrix..."
    year_mat = matstore.retrieve_cooccurrence_as_coo(INPUT_PATH.format(year=year))
    print "Manipulating matrix..."
    year_mat = year_mat.tocsr()
    year_mat = year_mat[indices, :]
    year_mat = year_mat[:, indices]
    freq_counts = year_mat.sum(1)
    year_mat = year_mat.tocoo()
    print "Writing data..."
    matstore.export_cooccurrence_eff(year_mat.row, year_mat.col, year_mat.data, year, OUTPUT_DIR)
    write_vocab(words, freq_counts, OUTPUT_DIR + str(year) + ".vocab")

def main(proc_num, lock):
    years = range(YEARS[0], YEARS[-1] + 1)
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
        make_year_train(year)


def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
