import util
import operator

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + '/1grams/'
OUTPUT_FILE = DATA_DIR + "/info/stopwords"
COMMON_WORD_FILE = DATA_DIR + "info/commonwords-90.pkl"
NUM_STOP_WORDS = 5000
MERGED_INDEX = list(util.load_pickle(DATA_DIR + "5grams_merged/merged_index.pkl"))

def main():
      freqs = util.load_pickle(INPUT_DIR + '2000-freqs.pkl') 
      sorted_f = sorted(freqs.items(), key= lambda (k, v): v[0], reverse=True)
      out_fp = open(OUTPUT_FILE, "w")
      for i in range(NUM_STOP_WORDS):
          out_fp.write(MERGED_INDEX[sorted_f[i][0]].encode('utf-8') + "\n")

if __name__ == '__main__':
    main()
