import argparse
import os
import ioutils

import numpy as np

def merge_bootstrap(out_pref):
    dir = "/".join(out_pref.split("/")[0:-1])
    bootfiles = os.listdir(dir)
    word_stat_lists = {}
    first_file = True
    file_num = 0
    for file in bootfiles:
        bootstats = ioutils.load_pickle(dir + "/" + file)
        print "Processing file", file
        for stat, stat_vals in bootstats.iteritems():
            if first_file:
                word_stat_lists[stat] = {}
            for word, val in stat_vals.iteritems():
                year_vals = stat_vals[word]
                if first_file:
                    word_stat_lists[stat][word] = {}
                for year, val in year_vals.iteritems():
                    if type(val) == float and np.isnan(val):
                        word_stat_lists[stat][word][year] = float('nan')
                    else:
                        if first_file:
                            word_stat_lists[stat][word][year] = np.empty((val.shape[0] * len(bootfiles)))
                        word_stat_lists[stat][word][year][file_num * val.shape[0]:(file_num + 1) * val.shape[0]] = val[:]
        first_file = False
        file_num += 1
    print "Making means and stds"
    word_stat_means = {}
    word_stat_stds = {}
    for stat, stat_vals in word_stat_lists.iteritems():
        word_stat_means[stat] = {}
        word_stat_stds[stat] = {}
        for word, year_vals in stat_vals.iteritems():
            word_stat_means[stat][word] = {}
            word_stat_stds[stat][word] = {}
            for year, val in year_vals.iteritems():
                if type(val) == float and np.isnan(val):
                    word_stat_means[stat][word][year] = float('nan')
                    word_stat_stds[stat][word][year] = float('nan')
                else:
                    word_stat_means[stat][word][year] = val.mean()
                    word_stat_stds[stat][word][year] = val.std()
    print "Writing data"
    for stat, mean_vals in word_stat_means.iteritems():
        ioutils.write_pickle(mean_vals, out_pref + "-" + stat + "-mean.pkl")
    for stat, std_vals in word_stat_stds.iteritems():
        ioutils.write_pickle(std_vals, out_pref + "-" + stat + "-std.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges bootstrap net data.")
    parser.add_argument("datapref", help="prefix of bootstrap data to be merged")
    args = parser.parse_args()
    merge_bootstrap(args.datapref)
  
