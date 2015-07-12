import numpy as np
import ioutils
from cooccurrence import matstore
from scipy.sparse import csr_matrix

def save_matrix(f, m):
    np.savez_compressed(f, data=m.data, indices=m.indices, indptr=m.indptr, shape=m.shape)

def load_matrix(f, thresh=None):
    if f.endswith('.bin'):
        if thresh == None:
            return matstore.retrieve_mat_as_coo(f, min_size=250000).tocsr()
        else:
            return matstore.retrieve_mat_as_coo_thresh(f, thresh, min_size=250000).tocsr()
    if not f.endswith('.npz'):
        f += '.npz'
    loader = np.load(f)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

def save_vocabulary(path, vocab):
    with open(path, 'w') as f:
        for w in vocab:
            print >>f, w

def load_vocabulary(mat, path):
    shared_vocab = list(ioutils.load_pickle(path.split(".")[0] + "-index.pkl"))
    iw = shared_vocab[:mat.shape[0]]
    ic = shared_vocab[:mat.shape[1]]
    return iw, ic

def load_shared_vocabulary(mat, mat_file):
    vocab_file = ""
    i = 0
    path = mat_file.split("/")
    while True:
        if "nppmi" in path[i]:
            break
        vocab_file += "/" + path[i]
        i += 1
    vocab_file += "/5grams/merged_list.pkl"
    shared_vocab = ioutils.load_pickle(vocab_file)
    iw = shared_vocab[:mat.shape[0]]
    ic = shared_vocab[:mat.shape[1]]
    return iw, ic

def save_count_vocabulary(path, vocab):
    with open(path, 'w') as f:
        for w, c in vocab:
            print >>f, w, c

def load_count_vocabulary(path):
    with open(path) as f:
        # noinspection PyTypeChecker
        vocab = dict([line.strip().split() for line in f if len(line) > 0])
    return vocab
