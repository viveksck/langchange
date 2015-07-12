import collections
import argparse
import ioutils
import os
import string
import random
from multiprocessing import Process, Lock

YEARS = range(1900, 2000)

def main(proc_num, lock, out_dir, in_dir):
    years = YEARS
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(out_dir))
            if str(year) + "-a.pkl" in dirs:
                continue
            
            work_left = True
            print proc_num, "year", year
            fname = out_dir + str(year) + "-a.pkl"
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

        for chunk_name in os.listdir(in_dir): 
            print "Processing chunk", chunk_name
            chunk_name = in_dir + str(chunk_name) + "/" + str(year) + ".pkl"
            if not os.path.isfile(chunk_name):
                continue
            chunk_counts = ioutils.load_pickle(chunk_name)
            for word, info_list in chunk_counts.iteritems():
                if word[0] not in year_grams:
                    continue
                for info in info_list:
                    gram = info[0].split("\t")[0]
                    count = info[1]
                    year_grams[word[0]][word].append((gram, count))
            
        print proc_num, "Writing counts for year", year
        for letter, letter_grams in year_grams.iteritems():
            for word in letter_grams:
                letter_grams[word] = sorted(letter_grams[word], key = lambda info : info[1], reverse=True)
            ioutils.write_pickle(letter_grams, out_dir + str(year) + "-" + letter + ".pkl")

def run_parallel(num_procs, out_dir, in_dir):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, out_dir, in_dir]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pulls and unzips raw 5gram data")
    parser.add_argument("out_dir", help="directory where data will be stored")
    parser.add_argument("in_dir", help="source dataset to pull from (must be available on the N-Grams website")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    args = parser.parse_args()
    run_parallel(args.num_procs, args.out_dir, args.in_dir) 
