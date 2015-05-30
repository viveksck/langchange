import collections
from .. import util
import heapq
from nltk.corpus import stopwords 


DATA_DIR = '/dfs/scratch0/google_ngrams/'
FREQ_DIR = DATA_DIR + "/1grams/"
OUTPUT_DIR = DATA_DIR + '/info/'
OUTPREFIX = OUTPUT_DIR + "interestingwords"

NUM_KEEP = 50000
YEARS = range(1900, 2000)
INDEX = list(util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl"))
STOP_WORDS = set(stopwords.words("english"))

def get_interesting_words(years, input_dir):
    word_freqs = collections.defaultdict(float)
    for year in years:
        print "Processing year", year
        year_freqs = util.load_pickle(input_dir + str(year) + "-freqs.pkl")
        sum = 0.0
        for _, counts in year_freqs.iteritems():
            sum += counts[0]  
        for word, counts in year_freqs.iteritems():
            word_word = INDEX[word]
            if word_word in STOP_WORDS or not word_word.isalpha() or len(word_word) <= 2:
                continue
            word_freqs[word] += float(counts[0]) / sum 
    top_words = heapq.nlargest(NUM_KEEP, word_freqs.iteritems(), key = lambda item : item[1])  
    top_words_list = []
    print "Writing year", year
    out_fp = open(OUTPREFIX + ".txt", "w")
    for word, freq_sum in top_words:
        out_fp.write(INDEX[word].encode('utf-8') + " " + str(freq_sum) + "\n")
        top_words_list.append(INDEX[word])
    util.write_pickle(top_words_list, OUTPREFIX + ".pkl")

if __name__ == "__main__":
    get_interesting_words(YEARS, FREQ_DIR)
        





