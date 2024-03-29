import heapq

from scipy.sparse import csr_matrix
from sklearn import preprocessing
import numpy as np

from vecanalysis.representations.matrix_serializer import load_vocabulary, load_matrix


class Explicit:
    """
    Base class for explicit representations. Assumes that the serialized input is (P)PMI.
    """
    
    def __init__(self, mat, word_vocab, context_vocab, normalize=True, restricted_context=None):
        self.m = mat
        self.iw = word_vocab
        self.ic = context_vocab
        self.wi = {w:i for i,w in enumerate(self.iw)}
        self.ci = {c:i for i,c in enumerate(self.ic)}
        self.normal = normalize
        if restricted_context != None:
            self.restrict_context(restricted_context)
        if normalize:
            self.normalize()

    @classmethod
    def load(cls, path, normalize=True, restricted_context=None, thresh=None):
        mat = load_matrix(path, thresh)
        word_vocab, context_vocab = load_vocabulary(mat, path)
        return cls(mat, word_vocab, context_vocab, normalize, restricted_context)

    def get_subembed(self, word_list):
        new_w_indices = np.array([self.wi[word] for word in word_list])
        c_set = set(self.ic)
        new_c_indices = np.array([self.ci[word] for word in word_list if word in c_set])
        new_m = self.m[new_w_indices, :]
        new_m = new_m[:, new_c_indices]
        return Explicit(new_m, word_list, word_list)

    def restrict_context(self, rel_words):
        rel_words = [word for word in rel_words if word in self.ci]
        rel_indices = np.array([self.ci[rel_word] for rel_word in rel_words])
        self.m = self.m[:, rel_indices]
        self.ic = rel_words
        self.ci = {c:i for i,c in enumerate(self.ic)}

    def normalize(self):
        preprocessing.normalize(self.m, copy=False)

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            return csr_matrix((1, len(self.ic)))
    
    def similarity_first_order(self, w, c):
        if self.oov(w) or self.oov(c):
            return 0.0
        return self.m[self.wi[w], self.ci[c]]
    
    def oov(self, w):
        return (not w in self.wi)

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        if self.oov(w1) or self.oov(w2):
            return 0.0
        return self.represent(w1).dot(self.represent(w2).T)[0, 0]
    
    def closest_contexts(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.represent(w)
        return heapq.nlargest(n, zip(scores.data, [self.ic[i] for i in scores.indices]))
    
    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        if self.oov(w):
            return []
        scores = self.m.dot(self.represent(w).T).T.tocsr()
        return heapq.nlargest(n, zip(scores.data, [self.iw[i] for i in scores.indices]))

    def closest_first_order(self, w, n=10):
        if self.oov(w):
            return []
        scores = self.m[self.wi[w], :]
        return heapq.nlargest(n, zip(scores.data, [self.iw[i] for i in scores.indices]))

class PositiveExplicit(Explicit):
    """
    Positive PMI (PPMI) with negative sampling (neg).
    Negative samples shift the PMI matrix before truncation.
    """
    
    def __init__(self, path, normalize=True, neg=1):
        Explicit.__init__(self, path, False)
        self.m.data -= np.log(neg)
        self.m.data[self.m.data < 0] = 0
        self.m.eliminate_zeros()
        if normalize:
            self.normalize()
