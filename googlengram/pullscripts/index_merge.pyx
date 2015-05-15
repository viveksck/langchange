import util
import collections

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INDEX_DIR = DATA_DIR + '/5grams_fixed/'

def run():
    index = collections.OrderedDict()
    cdef int year
    cdef int i
    for year in xrange(1700, 2009):
        print "Merging year", year
        year_list = util.load_pickle(INDEX_DIR + str(year) + "-list.pkl")
        i = 0
        for i in xrange(len(year_list)):
            word = year_list[i]
            util.word_to_cached_id(word, index)

    util.write_pickle(index, INDEX_DIR + "merged_index.pkl") 
    util.write_pickle(list(index), INDEX_DIR + "merged_list.pkl") 
