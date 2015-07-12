import argparse

from cooccurrence.makenulllang import make_null_language
from cooccurrence import matstore

START_YEAR = 1900
END_YEAR = 2000

def load_year_freqs(in_dir, years):
    year_freqs = {}
    year_sample_sizes = {}
    for year in years:
        mat = matstore.retrieve_mat_as_coo(in_dir + str(year) + ".bin")
        mat = mat.tocsr()
        year_sum = mat.sum() 
        mat = mat / year_sum
        year_sample_sizes[year] = year_sum / 4.0
        year_freqs[year] = {}
        for i in xrange(mat.shape[0]):
            year_freqs[year][i] = mat[i, :].sum() 
        print "Loaded year", year
    return year_freqs, year_sample_sizes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merges years of raw 5gram data.")
    parser.add_argument("out_dir", help="directory where data will be stored")
    parser.add_argument("freq_dir", help="path to unmerged data")
    args = parser.parse_args()
    years = range(START_YEAR, END_YEAR + 1)
    print "Loading year freqs.."
    year_freqs, sample_sizes = load_year_freqs(args.freq_dir + "/", years)
    ## FOR TESTING
    ##
    print "Making null language.."
    make_null_language(year_freqs, sample_sizes, args.out_dir)
