import os
import cPickle as pickle

### DICTIONARY / INDEX STUFF ###
def word_to_id(word, index):
    word = word.decode('utf-8').lower()
    try:
        return index[word]
    except KeyError:
        id_ = len(index)
        index[word] = id_
        return id_

def word_to_cached_id(word, index):
    try:
        return index[word]
    except KeyError:
        id_ = len(index)
        index[word] = id_
        return id_

def word_to_fixed_id(word, index):
    word = word.strip("\"")
    try:
        return index[word]
    except KeyError:
        id_ = len(index)
        index[word] = id_
        return id_

def word_to_static_id(word, index):
    return index[word]

def word_to_static_id_pass(word, index):
    try:
        return index[word]
    except KeyError:
        return -1;

## IO STUFF ##

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

## DATA HELPERS ##

def get_neighbours_ii(csr_mat, word_i):
    row = csr_mat[word_i, :]
    neighbours = row.nonzero()[1]
    return neighbours

def get_neighbours_iw(csr_mat, word_i, index):
    neighbours_i = get_neighbours_ii(csr_mat, word_i)
    neighbours = []
    index = list(index)
    for i in neighbours_i:
        neighbours.append(index[i])
    return neighbours

def get_neighbours_ww(csr_mat, word, index):
    return get_neighbours_iw(csr_mat, index[word], index)
        
     

