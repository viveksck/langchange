import time

from googlengram.statpullscripts.bootstrapping import bootstrap_count_mat 
from googlengram import matstore

if __name__ == '__main__':
    test_mat = matstore.retrieve_cooccurrence_as_coo("/dfs/scratch0/google_ngrams/5grams_merged/2008.bin")
    print "Starting test bootstrap..."
    starttime = time.clock()
    bootstrapped = bootstrap_count_mat(test_mat)
    endtime = time.clock()
    print "Finished. Bootstrap took", endtime - starttime, "seconds"

