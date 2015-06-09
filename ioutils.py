import os
import cPickle as pickle

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
