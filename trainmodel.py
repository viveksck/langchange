from vecanalysis.simplenetpred import SimpleDenseEncodeDecode
import numpy as np
from vecanalysis.representations.embedding import Embedding
from vecanalysis import alignment
from googlengram import util

DATA_DIR = '/dfs/scratch0/google_ngrams/'
OUTPUT_FILE = DATA_DIR + "info/simplemodel-5-5000000-5-100000-500.pkl"
COST_OUTPUT_FILE = DATA_DIR + "info/simplemodelcost-5-5000000-5-100000-500.pkl"

def rand_minibatch_sched(X, k, Nk):
        num = 0
        while num < Nk:
            yield np.random.randint(0, len(X), (k)).tolist()
            num += 1

if __name__ == '__main__':
    base = Embedding.load(DATA_DIR + "/sglove-vecs-smallrel-np/1980-300vecs")
    delta = Embedding.load(DATA_DIR + "/sglove-vecs-smallrel-np/1990-300vecs")
    base, delta = alignment.intersection_align(base, delta)
                
    X = base.m
    Y = delta.m
    np.random.seed(10)
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:80000], indices[80000:]
    trainX, testX = X[training_idx,:], X[test_idx,:]
    trainY, testY = Y[training_idx,:], Y[test_idx,:]
    test = SimpleDenseEncodeDecode(dims=[300,500,300], alpha=0.01)

    cost = test.train_sgd(trainX, trainY, idxiter=rand_minibatch_sched(trainX, 5, 1000000),alphaiter=SimpleDenseEncodeDecode.annealiter(0.5, epoch=100000), printevery=10000, costevery=10000)
    print "Writing model..."
    util.write_pickle(test, OUTPUT_FILE)
    util.write_pickle(cost, COST_OUTPUT_FILE)
