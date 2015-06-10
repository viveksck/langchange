import random
import os
import collections
from multiprocessing import Process, Lock

from cooccurrence import matstore
import ioutils

YEARS = range(1800, 2001)

def main(proc_num, lock, out_dir, in_dir):
    merged_index = ioutils.load_pickle(in_dir + "merged_index.pkl") 
    random.shuffle(years)
    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for year in years:
            dirs = set(os.listdir(out_dir))
            if str(year) + ".bin" in dirs:
                continue
            work_left = True
            print proc_num, "year", year
            fname = out_dir + str(year) + ".bin"
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
        old_mat = matstore.retrieve_mat_as_dict(in_dir + str(year) + ".bin")
        old_index = ioutils.load_pickle(in_dir + str(year) + "-list.pkl") 
        for pair, count in old_mat.iteritems():
            i_word = old_index[pair[0]]
            c_word = old_index[pair[1]]
            new_pair = (indexing.word_to_static_id(i_word, merged_index), 
                    indexing.word_to_static_id(c_word, merged_index))
            fixed_counts[new_pair] = count
        
        print proc_num, "Writing counts for year", year
        matstore.export_mats_from_dict({str(year) : fixed_counts}, out_dir)

def run_parallel(num_procs, out_dir, in_dir):
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, out_dir, in_dir]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Consolidates index for cooccurrence data.")
    parser.add_argument("out_dir", help="directory where the consolidated data will be stored. Must also contain merged index.")
    parser.add_argument("in_dir", help="path to unmerged data")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    args = parser.parse_args()
    run_parallel(args.num_procs, args.out_dir, args.in_dir)       

