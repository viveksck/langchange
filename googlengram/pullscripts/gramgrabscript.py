import requests
import urllib2
import re
import os
import subprocess
import collections
import argparse
import ioutils
from multiprocessing import Process, Lock
from nltk.corpus import stopwords

VERSION = '20120701'
TYPE = '5gram'
EXCLUDE_PATTERN = re.compile('.*_[A-Z]+[_,\s].*')
STOPWORDS = set(stopwords.words('english'))

def main(proc_num, lock, download_dir, source):
    page = requests.get("http://storage.googleapis.com/books/ngrams/books/datasetsv2.html")
    pattern = re.compile('href=\'(.*%s-%s-%s-.*\.gz)' % (source, TYPE, VERSION))
    urls = pattern.findall(page.text)
    del page

    print proc_num, "Start loop"
    while True:
        lock.acquire()
        work_left = False
        for url in urls:
            name = re.search('%s-(.*).gz' % VERSION, url).group(1)
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
            with open(loc_dir + name + '.gz', 'w') as f:
                try:
                    f.write(urllib2.urlopen(url, timeout=60).read())
                    success = True
                except:
                    continue

        print proc_num, "Unzipping", name
        subprocess.call(['gunzip', '-f', loc_dir + name + '.gz', '-d'])

        print proc_num, "Going through", name
        year_grams = collections.defaultdict(dict)
        n = 0
        with open(loc_dir + name) as f:
            for l in f:
                l = l.decode('utf-8').lower()
                split = l.strip().split('\t')
                if EXCLUDE_PATTERN.match(split[0]):
                    continue
                try:
                    ngram = split[0].split()
                    middle_index = len(ngram) // 2
                    item = ngram[middle_index]
                    if (not item.isalpha()) or item in STOPWORDS:
                        continue
                    year = split[1]
                    count = int(split[2])
                    if item not in year_grams[year]:
                        year_grams[year][item] = [(l, count)]
                    else:
                        year_grams[year][item].append((l, count))
                except:
                    #print "!", l.strip().split()
                    pass

        print proc_num, "Writing", name, n
        for year in year_grams:
            ioutils.write_pickle(year_grams[year], loc_dir + str(year) + ".pkl")

        print proc_num, "Deleting", name
        try:
            os.remove(loc_dir + name + '.gz')
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
