import requests
import urllib2
import re
import os
import subprocess
import collections
import matstore
import util
from multiprocessing import Process, Lock

DATA_DIR = '/dfs/scratch0/google_ngrams/'
SOURCE = 'eng-all'
VERSION = '20090715'#'20120701'
TYPE = '1gram'
DOWNLOAD_DIR = DATA_DIR + '/1grams/'
TMP_DIR = '/lfs/madmax4/0/will/google_ngrams/tmp/'
YEARS = range(1700, 2009)
INDEX_DIR = DATA_DIR + '/5grams_merged/'

def main():
    cdef int num_unk = 0
    merged_indices  = util.load_pickle(INDEX_DIR + "merged_index.pkl") 
    page = requests.get("http://storage.googleapis.com/books/ngrams/books/datasetsv2.html")
    pattern = re.compile('href=\'(.*%s-%s-%s-.*\.csv.zip)' % (SOURCE, TYPE, VERSION))
    urls = pattern.findall(page.text)
    del page

    year_freqs = {}
    for year in YEARS:
        year_freqs[year] = {}

    print "Start loop"
    for url in urls:
        name = re.search('%s-(.*).csv.zip' % VERSION, url).group(1)

        print "Name", name
        loc_dir = TMP_DIR + "/" + name + "/"
        util.mkdir(loc_dir)

        print  "Downloading", name

        success = False
        while not success:
            with open(loc_dir + name + '.csv.zip', 'w') as f:
                try:
                    f.write(urllib2.urlopen(url, timeout=60).read())
                    success = True
                except:
                    continue

        print  "Unzipping", name
        subprocess.call(['unzip', '-o', loc_dir + name + '.csv.zip', '-d', loc_dir])
        subprocess.call(['mv', loc_dir + 'googlebooks-' + SOURCE + '-' +  TYPE + '-' + VERSION + '-' + name + '.csv', loc_dir + name])

        print  "Going through", name
        with open(loc_dir + name) as f:
            for l in f:
                split = l.strip().split('\t')
                word = split[0].decode('utf-8').lower()
                year = int(split[1])
                count = int(split[2])
                doc_count = int(split[4])
                word_id = util.word_to_static_id_pass(word, merged_indices)
                if word_id == -1:
                    num_unk += 1
                    continue
                if not year in YEARS:
                    continue
                if not word_id in year_freqs[year]:
                    year_freqs[year][word_id] = (count, doc_count)
                else:
                    old_counts = year_freqs[year][word_id]
                    year_freqs[year][word_id] = (old_counts[0] + count, old_counts[1] + count)

        print "Deleting", name
        try:
            os.remove(loc_dir + name)
            os.remove(loc_dir + name + '.csv.zip')
        except:
            pass

    print "Writing..."
    for year in YEARS:
        util.write_pickle(year_freqs[year], DOWNLOAD_DIR + str(year) + "-freqs.pkl")

    print "Finished. Number of unmapped words: ", num_unk

def run():
    util.mkdir(TMP_DIR)
    main()
