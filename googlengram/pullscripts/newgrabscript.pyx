import requests
import urllib2
import re
import os
import subprocess
import collections
import matstore
import util
from multiprocessing import Process, Lock

DATA_DIR = '/lfs/madmax5/0/will/google_ngrams/unmerged'
SOURCE = 'eng-all'
VERSION = '20090715'#'20120701'
TYPE = '5gram'
DOWNLOAD_DIR = DATA_DIR + '/' + SOURCE + '/' + VERSION + '/' + TYPE + '/'

EXCLUDE_PATTERN = re.compile('.*_[A-Z]+_.*')

def main(proc_num, lock):
    page = requests.get("http://storage.googleapis.com/books/ngrams/books/datasetsv2.html")
    pattern = re.compile('href=\'(.*%s-%s-%s-.*\.csv.zip)' % (SOURCE, TYPE, VERSION))
    urls = pattern.findall(page.text)
    del page

    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for url in urls:
            name = re.search('%s-(.*).csv.zip' % VERSION, url).group(1)
            dirs = set(os.listdir(DOWNLOAD_DIR))
            if name in dirs:
                continue

            work_left = True
            print proc_num, "Name", name
            loc_dir = DOWNLOAD_DIR + "/" + name + "/"
            util.mkdir(loc_dir)
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
        subprocess.call(['mv', loc_dir + 'googlebooks-' + SOURCE + '-' +  TYPE + '-' + VERSION + '-' + name + '.csv', loc_dir + name])

        print proc_num, "Going through", name
        index = collections.OrderedDict()
        year_counters = collections.defaultdict(collections.Counter)
        n = 0
        with open(loc_dir + name) as f:
            for l in f:
                split = l.strip().split('\t')
                if EXCLUDE_PATTERN.match(split[0]):
                    continue
                try:
                    ngram = split[0].split()
                    middle_index = len(ngram) // 2
                    item = ngram[middle_index]
                    context = ngram[:middle_index] + ngram[middle_index + 1:]
                    item_id = util.word_to_id(item, index)
                    year = split[1]
                    count = int(split[2])
                    for context_word in context:
                        pair = (item_id, util.word_to_id(context_word, index))
                        year_counters[year][pair] += count
                except:
                    pass

        print proc_num, "Writing", name, n
        matstore.export_cooccurrence(year_counters, loc_dir)
        util.write_pickle(index, loc_dir + "index.pkl")

        print proc_num, "Deleting", name
        try:
            os.remove(loc_dir + name)
            os.remove(loc_dir + name + '.csv.zip')
        except:
            pass


def run_parallel(num_processes):
    util.mkdir(DATA_DIR)
    util.mkdir(DATA_DIR + '/' + SOURCE)
    util.mkdir(DATA_DIR + '/' + SOURCE + '/' + VERSION)
    util.mkdir(DOWNLOAD_DIR)
    lock = Lock()
    procs = [Process(target=main, args=[i, lock]) for i in range(num_processes)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
