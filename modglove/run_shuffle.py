import os 
import random
from multiprocessing import Lock, Process

DATA_DIR = "/dfs/scratch0/google_ngrams/"
OUTPUT_DIR = DATA_DIR + "modglove_train_smallrel/"
COOCCURRENCE_FILE = DATA_DIR + "modglove_train_smallrel/{year}.bin"
COOCCURRENCE_SHUF_FILE = DATA_DIR + "modglove_train_smallrel/{year}.shuf.bin"

YEARS = range(1850, 2009)

def shuffle_year(year):
    os.system('./shuffle -memory 10 -temp-file ' + str(year) + ' -verbose 2 < ' +COOCCURRENCE_FILE.format(year=year) + " > " + COOCCURRENCE_SHUF_FILE.format(year=year))
 
def main(proc_num, lock):
    years = range(YEARS[0], YEARS[-1] + 1)
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(OUTPUT_DIR))
            if str(year) + ".shuf.bin" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = OUTPUT_DIR + str(year) + ".shuf.bin"
            with open(fname, "w") as fp:
                fp.write("")
            fp.close()
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break
        shuffle_year(year)


def run_parallel(num_procs):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == '__main__':
    run_parallel(10)
