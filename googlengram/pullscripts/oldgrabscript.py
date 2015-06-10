import requests
import urllib2
import re
import os
import subprocess
import collections
import argparse
from multiprocessing import Process, Lock

import ioutils
from cooccurrence import matstore, indexing

VERSION = '20090715'
TYPE = '5gram'

def main(proc_num, lock, download_dir, source):
    page = requests.get("http://storage.googleapis.com/books/ngrams/books/datasetsv2.html")
    pattern = re.compile('href=\'(.*%s-%s-%s-.*\.csv.zip)' % (source, TYPE, VERSION))
    urls = pattern.findall(page.text)
    del page

    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for url in urls:
            name = re.search('%s-(.*).csv.zip' % VERSION, url).group(1)
            dirs = set(os.listdir(download_dir))
            if name in dirs:
                continue

            work_left = True
            print proc_num, "Name", name
            loc_dir = download_dir + "/" + name + "/"
            ioutils.mkdir(loc_dir)
            break
        lock.release()
        if not work_left:
            print proc_num, "Finished"
            break

        print proc_num, "Downloading", name

        success = False
        while not success:
            with open(loc_dir + name + '.csv.zip', 'w') as f:
                try:
                    f.write(urllib2.urlopen(url, timeout=60).read())
                    success = True
                except:
                    continue

        print proc_num, "Unzipping", name
        subprocess.call(['unzip', '-o', loc_dir + name + '.csv.zip', '-d', loc_dir])
        subprocess.call(['mv', loc_dir + 'googlebooks-' + source + '-' +  TYPE + '-' + VERSION + '-' + name + '.csv', loc_dir + name])

        print proc_num, "Going through", name
        index = collections.OrderedDict()
        year_counters = collections.defaultdict(collections.Counter)
        n = 0
        with open(loc_dir + name) as f:
            for l in f:
                split = l.strip().split('\t')
                try:
                    ngram = split[0].split()
                    middle_index = len(ngram) // 2
                    item = ngram[middle_index]
                    context = ngram[:middle_index] + ngram[middle_index + 1:]
                    item_id = indexing.word_to_id(item, index)
                    year = split[1]
                    count = int(split[2])
                    for context_word in context:
                        pair = (item_id, indexing.word_to_id(context_word, index))
                        year_counters[year][pair] += count
                except:
                    pass

        print proc_num, "Writing", name, n
        matstore.export_mats_from_dicts(year_counters, loc_dir)
        ioutils.write_pickle(index, loc_dir + "index.pkl")

        print proc_num, "Deleting", name
        try:
            os.remove(loc_dir + name)
            os.remove(loc_dir + name + '.csv.zip')
        except:
            pass


def run_parallel(num_processes, out_dir, source):
    ioutils.mkdir(out_dir)
    ioutils.mkdir(out_dir + '/' + source)
    ioutils.mkdir(out_dir + '/' + source + '/' + VERSION)
    download_dir = out_dir + '/' + source + '/' + VERSION + '/' + TYPE + '/'
    ioutils.mkdir(download_dir)
    lock = Lock()
    procs = [Process(target=main, args=[i, lock, download_dir, source]) for i in range(num_processes)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pulls and unzips raw 5gram data")
    parser.add_argument("out_dir", help="directory where data will be stored")
    parser.add_argument("source", help="source dataset to pull from (must be available on the N-Grams website")
    parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    args = parser.parse_args()
    run_parallel(args.num_procs, args.out_dir, args.source) 
