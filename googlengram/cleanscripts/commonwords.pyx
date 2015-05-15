import util
import collections

DATA_DIR = '/dfs/scratch0/google_ngrams/'
UNMERGED_DIR = DATA_DIR + '/5grams_fixed/'
OUTPUT_DIR = DATA_DIR + '/info/'

THRESHOLD = 0.9
YEARS = range(2000, 2001)

def main():
    word_year_counts = collections.defaultdict(float)
    for year in YEARS:
        print "Processing year", year
        year_list = util.load_pickle(UNMERGED_DIR + str(year) + "-list.pkl")
        for word in year_list:
            word_year_counts[word] += 1.0

    common_words = []
    for word, count in word_year_counts.iteritems():
        if count / float(len(YEARS)) >= THRESHOLD:
            common_words.append(word)

    util.write_pickle(common_words, OUTPUT_DIR + "commonwords-" + str(int(100*THRESHOLD)) + ".pkl")

if __name__ == "__main__":
    main()

            
