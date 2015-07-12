import argparse
import os
import ioutils

import numpy as np

def merge_bootstrap(out_pref):
    dir = "/".join(out_pref.split("/")[0:-1])
    bootfiles = os.listdir(dir)
    word_stat_lists = {}
    for file in bootfiles:
        first_file = True
        if "-".join(file.split("-")[:-1]) != out_pref.split("/")[-1]:
            print "skipping file", file
            continue
        bootstats = ioutils.load_pickle(dir + "/" + file)
        for stat, stat_vals in bootstats.iteritems():
            word_stat_lists[stat] = {}
            for word, val in stat_vals.iteritems():
                year_vals = stat_vals[word]
                word_stat_lists[stat][word] = {}
                for year, val in year_vals.iteritems():
                    if np.isnan(val):
                        word_stat_lists[stat][word][year] = float('nan')
                    else:
                        if first_file:
                            word_stat_lists[stat][word][year] = val.tolist()
                        else:
                            word_stat_lists[stat][word][year].extend(val)
        first_file = False
    word_stat_means = {}
    word_stat_stds = {}
    for stat, stat_vals in word_stat_lists.iteritems():
        word_stat_means[stat] = {}
        word_stat_stds[stat] = {}
        for word, year_vals in stat_vals.iteritems():
            word_stat_means[stat][word] = {}
            word_stat_stds[stat][word] = {}
            for year, val in year_vals.iteritems():
                if np.isnan(val):
                    word_stat_means[stat][word][year] = float('nan')
                    word_stat_stds[stat][word][year] = float('nan')
                else:
                    word_stat_means[stat][word][year] = np.mean(val)
                    word_stat_stds[stat][word][year] = np.std(val)

    for stat, mean_vals in word_stat_means.iteritems():
        ioutils.write_pickle(mean_vals, out_pref + "-" + stat + "-mean.pkl")
    for stat, std_vals in word_stat_stds.iteritems():
        ioutils.write_pickle(std_vals, out_pref + "-" + stat + "-std.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges bootstrap net data.")
    parser.add_argument("datapref", help="prefix of bootstrap data to be merged")
    args = parser.parse_args()
    merge_bootstrap(args.datapref)
  
