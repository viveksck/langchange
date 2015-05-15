import util
import os
import random
from multiprocessing import Process, Lock

DATA_DIR = '/dfs/scratch0/google_ngrams/'
SOURCE = 'eng-all'
VERSION = '20090715'#'20120701'
INDEX_DIR = DATA_DIR + '/5grams_fixed/'

def main(proc_num, lock):
    years = range(1780, 2009)
    random.shuffle(years)
    while True:
        found_year = False
        lock.acquire()
        for year in years:
            if (str(year) + "-list.pkl" in os.listdir(INDEX_DIR)):
                continue
            with open(INDEX_DIR + str(year) + "-index.pkl", "w") as fp:
                fp.write("")
            fp.close()
            found_year = True
        lock.release()
        if not found_year:
            break
        index = util.load_pickle(INDEX_DIR + str(year) + "-index.pkl")
        util.write_pickle(list(index), INDEX_DIR + str(year) + "-list.pkl")
        print "wrote year", year

if __name__ == '__main__':
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(0, 9)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


