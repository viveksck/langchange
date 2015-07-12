import collections
import argparse

import ioutils

START_YEAR = 1800
END_YEAR = 2000

def get_sorted_words(years, out_dir, in_dir):
    word_freqs = collections.defaultdict(float)
    for year in years:
        print "Processing year", year
        year_freqs = ioutils.load_pickle(in_dir + str(year) + "-freqs.pkl")
        sum = 0.0
        for _, counts in year_freqs.iteritems():
            sum += counts[0]  
        for word, counts in year_freqs.iteritems():
            if not word.isalpha():
                continue
            word_freqs[word] += float(counts[0]) / sum 
    print "Writing data"
    sorted_list = sorted(word_freqs.keys(), key = lambda key : word_freqs[key], reverse=True)
    out_pref = out_dir + "sortedwords-" + str(years[0]) + "-" + str(years[-1]) 
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
    args = parser.parse_args()

    years = range(args.start_year, args.end_year + 1)
    get_sorted_words(years, args.out_dir + "/" , args.in_dir + "/")
        





