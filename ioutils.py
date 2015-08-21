import os
import collections
import cPickle as pickle
from cooccurrence.indexing import get_word_indices

def load_year_index_infos(index_dir, years, word_file, num_words=-1):
    year_index_infos = collections.defaultdict(dict)
    word_lists = load_year_words(word_file, years)
    for year, word_list in word_lists.iteritems():
        year_index = load_pickle(index_dir + "/" + str(year) + "-index.pkl") 
        year_index_infos[year]["index"] = year_index
        if num_words != -1:
            word_list = word_list[:num_words]
        word_list, word_indices = get_word_indices(word_list, year_index)
        year_index_infos[year]["list"] = word_list
        year_index_infos[year]["indices"] = word_indices
    return year_index_infos

def load_year_index_infos_common(common_index, years, word_file, num_words=-1):
    year_index_infos = collections.defaultdict(dict)
    word_lists = load_year_words(word_file, years)
    for year, word_list in word_lists.iteritems():
        year_index = common_index
        year_index_infos[year]["index"] = year_index
        if num_words != -1:
            word_list = word_list[:num_words]
        word_list, word_indices = get_word_indices(word_list, year_index)
        year_index_infos[year]["list"] = word_list
        year_index_infos[year]["indices"] = word_indices
    return year_index_infos

def load_year_words(word_file, years):
    word_pickle = load_pickle(word_file)
    word_lists = {}
    if not years[0] in word_pickle:
        for year in years:
            word_lists[year] = word_pickle
    else:
        for year in years:
            word_lists[year] = word_pickle[year]
    return word_lists

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) 

def write_pickle(data, filename):
    fp = open(filename, "wb")
    pickle.dump(data, fp)

def load_pickle(filename):
    fp = open(filename, "rb")
    return pickle.load(fp)

def load_word_list(filename):
    fp = open(filename, "r")
    words = []
    for line in fp:
        words.append(line.strip())
    fp.close()
    return words
