import argparse
import random
import collections
from Queue import Empty
from multiprocessing import Process, Queue
from sklearn.preprocessing import normalize

import ioutils
from cooccurrence import matstore

START_YEAR = 1900
END_YEAR = 2000
THRESHOLD = 0.1
MIN_COOCCURS = 10

def make_secondorder_mat(old_mat, thresh=THRESHOLD, min_cooccurs=MIN_COOCCURS, shrink_mat=True):
    old_mat.setdiag(0)
    old_mat = old_mat.tocsr()
    rows_to_del = set([])
    for row_i in xrange(old_mat.shape[0]):
        if len(old_mat[row_i, :].nonzero()[1]) < min_cooccurs:
            rows_to_del.add(row_i)
    print "To delete:", len(rows_to_del)
    old_mat = old_mat.tocoo()
    for i in xrange(len(old_mat.data)):
        if old_mat.row[i] in rows_to_del:
            old_mat.data[i] = 0
    old_mat = old_mat.tocsr()
    normalize(old_mat, copy=False)
    new_mat = old_mat.dot(old_mat.T)
    new_mat = new_mat.tocoo()
    new_mat.data[new_mat.data < thresh] = 0
    keep_rows = []
    if shrink_mat:
        new_mat = new_mat.tocsr()
        for row_i in xrange(new_mat.shape[0]):
            if new_mat[row_i, :].sum() > 0:
                keep_rows.append(row_i)
        print "Keep rows:", len(keep_rows)

        new_mat.eliminate_zeros()
        new_mat = new_mat[keep_rows, :]
        new_mat = new_mat[:, keep_rows]
        print "New mat dim:", new_mat.shape[0]
        new_mat = new_mat.tocoo()
    return new_mat.row, new_mat.col, new_mat.data, keep_rows

def worker(proc_num, queue, in_dir):
    print proc_num, "Start loop"
    while True:
        try: 
            year = queue.get(block=False)
        except Empty:
            print proc_num, "Finished"
            break

        print proc_num, "Making second orders for year", year
        old_mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin")
        row_d, col_d, data_d, keep_rows = make_secondorder_mat(old_mat)
        old_index = list(ioutils.load_pickle(in_dir + str(year) + "-index.pkl"))
        new_index = collections.OrderedDict()
        for i in xrange(len(keep_rows)):
            new_index[old_index[keep_rows[i]]] = i
        ioutils.write_pickle(new_index, in_dir + "/second/" + str(year) + "-index.pkl")
        print proc_num, "Writing counts for year", year
        matstore.export_mat_eff(row_d, col_d, data_d, year, in_dir + "/second/")

def run_parallel(num_procs, in_dir, years):
    queue = Queue()
    random.shuffle(years)
    for year in years:
        queue.put(year)
    procs = [Process(target=worker, args=[i, queue, in_dir, years]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
 
if __name__ == '__worker__':
    parser = argparse.ArgumentParser(description="Makes and stores second order matrices from first order PPMI data..")
    parser.add_argument("in_dir", help="path to first order data")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=START_YEAR)
    parser.add_argument("--end-year", type=int, help="start year (inclusive)", default=END_YEAR)
    args = parser.parse_args()
    years = range(args.start_year, args.end_year + 1)
    ioutils.mkdir(args.in_dir + "/second")
    run_parallel(args.num_procs, args.in_dir + "/", years) 
