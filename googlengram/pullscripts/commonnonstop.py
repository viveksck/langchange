import collections
import argparse

from nltk.corpus import stopwords

import ioutils

START_YEAR = 1800
END_YEAR = 2000
AVG_THRESH = 7
MIN_THRESH = 8

def get_sorted_words(years, out_pref, in_dir, avg_thresh, min_thresh):
    stop_set = set(stopwords.words('english'))
    word_freqs = collections.defaultdict(float)
    word_mins = collections.defaultdict(lambda : 1.0)
    for year in years:
        print "Processing year", year
        year_freqs = ioutils.load_pickle(in_dir + str(year) + "-freqs.pkl")
        sum = 0.0
        for _, counts in year_freqs.iteritems():
            sum += counts[0]  
        for word, counts in year_freqs.iteritems():
            if not word.isalpha() or word in stop_set or len(word) == 1:
                continue
            year_freq = float(counts[0]) / sum 
            word_freqs[word] += year_freq
            word_mins[word] = min(word_mins[word], year_freq)
    print "Writing data"
    sorted_list = sorted(word_freqs.keys(), key = lambda key : word_freqs[key], reverse=True)
    sorted_list = [word for word in sorted_list 
            if (word_freqs[word] / float(len(years)) > avg_thresh and word_mins[word] > min_thresh)]
    out_fp = open(out_pref + ".txt", "w")
    for word in sorted_list:
        out_fp.write(word.encode('utf-8') + " " + str(word_freqs[word] / float(len(years))) + "\n")
    ioutils.write_pickle(sorted_list, out_pref + ".pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get sorted list of words according to average relative frequency")
    parser.add_argument("out_dir", help="output directory")
    parser.add_argument("in_dir", help="directory with 1 grams")
    parser.add_argument("--start-year", type=int, default=START_YEAR, help="start year (inclusive)")
    parser.add_argument("--end-year", type=int, default=END_YEAR, help="end year (inclusive)")
    parser.add_argument("--avg-thresh", type=int, default=AVG_THRESH, help="end year (inclusive)")
    parser.add_argument("--min-thresh", type=int, default=MIN_THRESH, help="end year (inclusive)")
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    out_pref = args.out_dir + "/commonnonstop-" + str(years[0]) + "-" + str(years[-1]) + "-" + str(args.min_thresh) + "-" + str(args.avg_thresh)
    avg_thresh = 10.0 ** (-1.0 * float(args.avg_thresh))
    if args.min_thresh == 0:
        min_thresh = 0
    else:
        min_thresh = 10.0 ** (-1.0 * float(args.min_thresh))
    get_sorted_words(years, out_pref , args.in_dir + "/", avg_thresh, min_thresh)
 
